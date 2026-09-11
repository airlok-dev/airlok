//! Completion and hints for the prompt. Typing `/` shows the matching
//! commands with their descriptions under the line; `@` offers paths from
//! the working directory, gitignore-aware and fuzzy. Tab takes the first
//! match (Tab again cycles), and a completed path goes in as plain text,
//! without the `@`.

use std::borrow::Cow;
use std::cell::OnceCell;
use std::path::{Path, PathBuf};

use airlok_core::repl::COMMANDS;
use rustyline::completion::{Completer, Pair};
use rustyline::highlight::Highlighter;
use rustyline::hint::{Hint, Hinter};
use rustyline::validate::Validator;
use rustyline::{Context, Helper};

/// Rows of the menu under the line: enough for every command, so a bare
/// `/` lists them all.
const MENU_ROWS: usize = 12;
/// Paths offered for one `@`.
const PATH_MATCHES: usize = 10;
/// Entries read from the working directory for `@`, at most.
const INDEX_LIMIT: usize = 20_000;

pub struct Prompt {
    cwd: PathBuf,
    /// Dim the menu; off when NO_COLOR is set.
    dim: bool,
    /// Paths under `cwd`, read on the first `@` after a refresh.
    index: OnceCell<Vec<String>>,
}

/// A menu under the line. Only a command name can be completed from it
/// with the right arrow; a path needs Tab, which also removes the `@`.
pub struct MenuHint {
    display: String,
    completion: Option<String>,
}

impl Hint for MenuHint {
    fn display(&self) -> &str {
        &self.display
    }

    fn completion(&self) -> Option<&str> {
        self.completion.as_deref()
    }
}

impl Prompt {
    pub fn new(cwd: PathBuf, dim: bool) -> Self {
        Self {
            cwd,
            dim,
            index: OnceCell::new(),
        }
    }

    /// Forgets the paths read so far, so the next `@` sees the files the
    /// last turn created.
    pub fn refresh(&mut self) {
        self.index.take();
    }

    /// Replacement start and candidates for Tab at `pos`.
    fn candidates(&self, line: &str, pos: usize) -> (usize, Vec<Pair>) {
        let pairs = |names: Vec<String>| {
            names
                .into_iter()
                .map(|name| Pair {
                    display: name.clone(),
                    replacement: name,
                })
                .collect()
        };
        if let Some((start, query)) = at_word(line, pos) {
            return (start, pairs(self.paths_matching(query)));
        }
        if let Some(typed) = command_word(line, pos) {
            let names = commands_matching(typed)
                .map(|(name, _)| name.to_string())
                .collect();
            return (0, pairs(names));
        }
        (pos, Vec::new())
    }

    /// The menu for the line so far, shown only with the cursor at the end.
    fn menu(&self, line: &str, pos: usize) -> Option<MenuHint> {
        if pos < line.len() {
            return None;
        }
        if let Some(typed) = command_word(line, pos) {
            let matches: Vec<_> = commands_matching(typed).collect();
            let (first, _) = matches.first()?;
            let rest = first[1 + typed.len()..].to_string();
            let mut display = rest.clone();
            for (name, what) in matches.iter().take(MENU_ROWS) {
                display.push_str(&format!("\n  {name:<12} {what}"));
            }
            return Some(MenuHint {
                display,
                completion: Some(rest),
            });
        }
        let (_, query) = at_word(line, pos)?;
        let paths = self.paths_matching(query);
        if paths.is_empty() {
            return None;
        }
        let mut display = String::new();
        for path in paths.iter().take(MENU_ROWS) {
            display.push_str(&format!("\n  {path}"));
        }
        Some(MenuHint {
            display,
            completion: None,
        })
    }

    /// The best matches for `query` among the paths under `cwd`.
    fn paths_matching(&self, query: &str) -> Vec<String> {
        let index = self.index.get_or_init(|| read_index(&self.cwd));
        let mut scored: Vec<(i64, &String)> = index
            .iter()
            .filter_map(|path| fuzzy_score(query, path).map(|score| (score, path)))
            .collect();
        scored.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then(a.1.len().cmp(&b.1.len()))
                .then(a.1.cmp(b.1))
        });
        scored
            .into_iter()
            .take(PATH_MATCHES)
            .map(|(_, path)| path.clone())
            .collect()
    }
}

impl Completer for Prompt {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        Ok(self.candidates(line, pos))
    }
}

impl Hinter for Prompt {
    type Hint = MenuHint;

    fn hint(&self, line: &str, pos: usize, _ctx: &Context<'_>) -> Option<MenuHint> {
        self.menu(line, pos)
    }
}

impl Highlighter for Prompt {
    fn highlight_hint<'h>(&self, hint: &'h str) -> Cow<'h, str> {
        if self.dim {
            Cow::Owned(format!("\x1b[2m{hint}\x1b[0m"))
        } else {
            Cow::Borrowed(hint)
        }
    }
}

impl Validator for Prompt {}

impl Helper for Prompt {}

/// The `@` word that ends at the cursor: where it starts (at the `@`) and
/// what follows the `@`.
fn at_word(line: &str, pos: usize) -> Option<(usize, &str)> {
    let before = line.get(..pos)?;
    let start = before
        .char_indices()
        .rev()
        .find(|(_, c)| c.is_whitespace())
        .map_or(0, |(i, c)| i + c.len_utf8());
    before[start..]
        .strip_prefix('@')
        .map(|query| (start, query))
}

/// The command name being typed: the line starts with `/` and the cursor
/// is still in its first word.
fn command_word(line: &str, pos: usize) -> Option<&str> {
    let typed = line.get(..pos)?.strip_prefix('/')?;
    (!typed.contains(char::is_whitespace)).then_some(typed)
}

/// Commands whose name starts with `typed`, in menu order: the first is
/// what Enter runs.
fn commands_matching(
    typed: &str,
) -> impl Iterator<Item = &'static (&'static str, &'static str)> + '_ {
    COMMANDS
        .iter()
        .filter(move |(name, _)| name[1..].starts_with(typed))
}

/// Paths under `cwd`, relative, directories ending in `/`. Gitignored
/// paths are left out; dotfiles are in, `.git` never is.
fn read_index(cwd: &Path) -> Vec<String> {
    ignore::WalkBuilder::new(cwd)
        .hidden(false)
        .filter_entry(|entry| entry.file_name() != ".git")
        .build()
        .filter_map(Result::ok)
        .filter(|entry| entry.depth() > 0)
        .take(INDEX_LIMIT)
        .filter_map(|entry| {
            let relative = entry.path().strip_prefix(cwd).ok()?.to_str()?.to_string();
            let is_dir = entry.file_type().is_some_and(|t| t.is_dir());
            Some(if is_dir {
                format!("{relative}/")
            } else {
                relative
            })
        })
        .collect()
}

/// How well `path` matches `query`: every query character must appear in
/// order, ignoring case. A path that starts with the query ranks first;
/// then matches at the start of a path segment or word, and runs of
/// adjacent matches, add to the score. `None` when it does not match.
fn fuzzy_score(query: &str, path: &str) -> Option<i64> {
    let chars: Vec<char> = path.chars().collect();
    let mut score = 0i64;
    let mut next = 0;
    let mut last: Option<usize> = None;
    for wanted in query.chars().map(|c| c.to_ascii_lowercase()) {
        let found = (next..chars.len()).find(|&i| chars[i].to_ascii_lowercase() == wanted)?;
        score += 1;
        if found == 0 || matches!(chars[found - 1], '/' | '_' | '-' | '.' | ' ') {
            score += 8;
        }
        if last.is_some_and(|l| l + 1 == found) {
            score += 5;
        }
        last = Some(found);
        next = found + 1;
    }
    if path
        .to_ascii_lowercase()
        .starts_with(&query.to_ascii_lowercase())
    {
        score += 100;
    }
    Some(score)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    use airlok_tests::TempDir;

    /// A git repository with a few files and an ignored build directory.
    fn project() -> TempDir {
        let dir = TempDir::new("complete");
        let root = dir.path();
        for path in [
            "src/main.rs",
            "src/render.rs",
            "README.md",
            "target/debug/airlok",
        ] {
            let path = root.join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, "x").unwrap();
        }
        std::fs::write(root.join(".gitignore"), "target/\n").unwrap();
        let status = Command::new("git")
            .args(["init", "-q"])
            .current_dir(root)
            .status()
            .unwrap();
        assert!(status.success());
        dir
    }

    fn replacements(pairs: &[Pair]) -> Vec<&str> {
        pairs.iter().map(|p| p.replacement.as_str()).collect()
    }

    #[test]
    fn at_completes_a_path_as_plain_text() {
        let dir = project();
        let prompt = Prompt::new(dir.path().to_path_buf(), true);
        let line = "look at @src/ma";
        let (start, pairs) = prompt.candidates(line, line.len());
        assert_eq!(start, 8, "replaces from the @");
        assert_eq!(replacements(&pairs)[0], "src/main.rs");
        let (_, pairs) = prompt.candidates("@rend", 5);
        assert_eq!(replacements(&pairs)[0], "src/render.rs", "fuzzy");
    }

    #[test]
    fn at_leaves_out_gitignored_paths_and_git_itself() {
        let dir = project();
        let prompt = Prompt::new(dir.path().to_path_buf(), true);
        let (_, pairs) = prompt.candidates("@", 1);
        let offered = replacements(&pairs);
        assert!(offered.contains(&"README.md"), "{offered:?}");
        assert!(offered.contains(&".gitignore"), "{offered:?}");
        assert!(
            !offered
                .iter()
                .any(|p| p.starts_with("target") || p.starts_with(".git/")),
            "{offered:?}"
        );
    }

    #[test]
    fn an_at_inside_a_word_is_not_a_path() {
        assert_eq!(at_word("mail me@example.com", 19), None);
        assert_eq!(at_word("see @", 5), Some((4, "")));
    }

    #[test]
    fn slash_shows_the_matching_commands_and_completes_the_first() {
        let prompt = Prompt::new(PathBuf::from("."), true);
        let menu = prompt.menu("/co", 3).unwrap();
        let lines: Vec<&str> = menu.display().lines().collect();
        assert_eq!(lines[0], "st", "the rest of /cost, inline");
        assert_eq!(lines.len(), 4, "{lines:?}");
        assert!(lines[1].starts_with("  /cost ") && lines[1].contains("tokens used"));
        assert!(lines[2].starts_with("  /compact "));
        assert!(lines[3].starts_with("  /config "));
        assert_eq!(menu.completion(), Some("st"));
        let (start, pairs) = prompt.candidates("/co", 3);
        assert_eq!(start, 0);
        assert_eq!(replacements(&pairs), ["/cost", "/compact", "/config"]);
        let all = prompt.menu("/", 1).unwrap();
        assert_eq!(
            all.display().lines().count(),
            1 + COMMANDS.len(),
            "a bare / lists every command"
        );
        assert!(prompt.menu("/model gpt", 10).is_none(), "not past the name");
        assert!(prompt.menu("/co", 1).is_none(), "cursor not at the end");
    }
}
