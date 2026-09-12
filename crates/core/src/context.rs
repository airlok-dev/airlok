//! The context block prepended to the system prompt: environment, git
//! state, a file tree, and project instructions. Built once per run, and
//! redacted with the rest of the system prompt before it leaves.

use std::path::{Path, PathBuf};
use std::process::Command;

use ignore::WalkBuilder;

/// How deep the file tree goes below the repository root.
pub const TREE_DEPTH: usize = 4;
/// Entries listed before the tree is cut and only top-level dirs remain.
pub const TREE_MAX_ENTRIES: usize = 200;
/// Project instruction files, in order of preference.
pub const INSTRUCTION_FILES: &[&str] = &["AIRLOK.md", "CLAUDE.md", "AGENTS.md"];

pub struct ContextInput<'a> {
    pub cwd: &'a Path,
    /// The user-level instructions file, normally `~/.config/airlok/AIRLOK.md`.
    pub user_instructions: Option<&'a Path>,
    /// Upper bound on the rendered block. The tree is cut first, then the
    /// instructions.
    pub max_bytes: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ContextBlock {
    pub text: String,
    /// Instruction files that were loaded, in order.
    pub instruction_files: Vec<PathBuf>,
    /// How many bytes of `text` are instruction sections, after any
    /// truncation. `/context` reports this as their share of the block.
    pub instruction_bytes: usize,
}

/// The current git branch of `cwd`, when it is in a repository at all.
pub fn branch(cwd: &Path) -> Option<String> {
    GitInfo::detect(cwd).map(|git| git.branch)
}

pub fn build(input: &ContextInput<'_>) -> ContextBlock {
    let git = GitInfo::detect(input.cwd);
    let root = git
        .as_ref()
        .map(|g| g.toplevel.clone())
        .unwrap_or_else(|| input.cwd.to_path_buf());
    let mut instruction_files = Vec::new();
    let project = project_instructions(&root);
    if let Some((path, _, _)) = &project {
        instruction_files.push(path.clone());
    }
    let user = input.user_instructions.and_then(|p| {
        std::fs::read_to_string(p)
            .ok()
            .map(|text| (p.to_path_buf(), text))
    });
    if let Some((path, _)) = &user {
        instruction_files.push(path.clone());
    }

    let head = environment_section(input.cwd, git.as_ref());
    let full_tree = tree_section(&root, false);
    let (text, instruction_bytes) = assemble(&head, &full_tree, &project, &user, input.max_bytes)
        .unwrap_or_else(|| {
            let short_tree = tree_section(&root, true);
            assemble(&head, &short_tree, &project, &user, input.max_bytes).unwrap_or_else(|| {
                assemble_truncated(&head, &short_tree, &project, &user, input.max_bytes)
            })
        });
    ContextBlock {
        text,
        instruction_files,
        instruction_bytes,
    }
}

type Instructions = Option<(PathBuf, String, String)>;

/// Everything but the instructions fits by construction; returns None when
/// the whole block would exceed the budget.
fn assemble(
    head: &str,
    tree: &str,
    project: &Instructions,
    user: &Option<(PathBuf, String)>,
    max_bytes: usize,
) -> Option<(String, usize)> {
    let instructions = format!(
        "{}{}",
        project
            .as_ref()
            .map(|(_, label, body)| section(label, body))
            .unwrap_or_default(),
        user.as_ref()
            .map(|(path, body)| section(&format!("User instructions ({})", path.display()), body))
            .unwrap_or_default()
    );
    let text = format!("{head}{tree}{instructions}");
    (text.len() <= max_bytes).then_some((text, instructions.len()))
}

/// Last resort: the short tree stays, the instructions are cut to fit.
fn assemble_truncated(
    head: &str,
    tree: &str,
    project: &Instructions,
    user: &Option<(PathBuf, String)>,
    max_bytes: usize,
) -> (String, usize) {
    const MARK: &str = "\n[instructions truncated to fit context.max_bytes]\n";
    let mut text = format!("{head}{tree}");
    let without_instructions = text.len();
    let mut budget = max_bytes.saturating_sub(text.len());
    let parts: Vec<(String, &str)> = project
        .iter()
        .map(|(_, label, body)| (label.clone(), body.as_str()))
        .chain(user.iter().map(|(path, body)| {
            (
                format!("User instructions ({})", path.display()),
                body.as_str(),
            )
        }))
        .collect();
    for (label, body) in parts {
        let header = format!("\n## {label}\n\n");
        if budget <= header.len() + MARK.len() {
            break;
        }
        let room = budget - header.len() - MARK.len();
        if body.len() <= room {
            let piece = format!("{header}{body}\n");
            budget -= piece.len();
            text.push_str(&piece);
        } else {
            let cut = floor_char_boundary(body, room);
            text.push_str(&format!("{header}{}{MARK}", &body[..cut]));
            break;
        }
    }
    let instructions = text.len() - without_instructions;
    (text, instructions)
}

fn section(label: &str, body: &str) -> String {
    format!("\n## {label}\n\n{}\n", body.trim_end())
}

fn floor_char_boundary(s: &str, mut index: usize) -> usize {
    index = index.min(s.len());
    while !s.is_char_boundary(index) {
        index -= 1;
    }
    index
}

struct GitInfo {
    toplevel: PathBuf,
    branch: String,
    status: String,
    commits: Vec<String>,
}

impl GitInfo {
    fn detect(cwd: &Path) -> Option<Self> {
        let toplevel = PathBuf::from(git(cwd, &["rev-parse", "--show-toplevel"])?.trim());
        let branch = git(cwd, &["branch", "--show-current"])
            .map(|b| b.trim().to_string())
            .filter(|b| !b.is_empty())
            .unwrap_or_else(|| "(detached HEAD)".to_string());
        let status = git(cwd, &["status", "--short"]).unwrap_or_default();
        let commits = git(cwd, &["log", "-5", "--format=%s"])
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .collect();
        Some(Self {
            toplevel,
            branch,
            status,
            commits,
        })
    }
}

fn git(cwd: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

fn environment_section(cwd: &Path, git: Option<&GitInfo>) -> String {
    let shell = std::env::var("SHELL")
        .ok()
        .and_then(|s| {
            Path::new(&s)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "unknown".to_string());
    let mut out = format!(
        "# Context\n\n## Environment\n\n- cwd: {}\n- os: {}\n- shell: {shell}\n",
        cwd.display(),
        std::env::consts::OS
    );
    match git {
        None => out.push_str("- git: not a repository\n"),
        Some(git) => {
            out.push_str(&format!(
                "- git: repository at {}, branch {}\n",
                git.toplevel.display(),
                git.branch
            ));
            out.push_str("\n## Git status\n\n");
            if git.status.trim().is_empty() {
                out.push_str("clean\n");
            } else {
                out.push_str(&format!("```\n{}```\n", git.status));
            }
            out.push_str("\n## Recent commits\n\n");
            for subject in &git.commits {
                out.push_str(&format!("- {subject}\n"));
            }
        }
    }
    out
}

/// The tree respects .gitignore, includes dotfiles other than `.git`, and
/// is sorted. `top_level_only` is the shape used when space is short.
fn tree_section(root: &Path, top_level_only: bool) -> String {
    let depth = if top_level_only { 1 } else { TREE_DEPTH };
    let mut entries: Vec<(PathBuf, bool)> = WalkBuilder::new(root)
        .hidden(false)
        .max_depth(Some(depth))
        .sort_by_file_name(|a, b| a.cmp(b))
        .filter_entry(|e| e.file_name() != ".git")
        .build()
        .filter_map(Result::ok)
        .filter(|e| e.depth() > 0)
        .map(|e| {
            let is_dir = e.file_type().is_some_and(|t| t.is_dir());
            (
                e.path()
                    .strip_prefix(root)
                    .unwrap_or(e.path())
                    .to_path_buf(),
                is_dir,
            )
        })
        .collect();
    let total = entries.len();
    let cap = if top_level_only {
        usize::MAX
    } else {
        TREE_MAX_ENTRIES
    };
    let truncated = total > cap;
    if truncated {
        entries.truncate(cap);
    }
    let mut out = format!(
        "\n## Files\n\n{}\n```\n",
        if top_level_only {
            format!(
                "Top level of {} (tree cut to fit context.max_bytes):",
                root.display()
            )
        } else {
            format!(
                "Tree of {} (depth {TREE_DEPTH}, .gitignore respected):",
                root.display()
            )
        }
    );
    for (path, is_dir) in &entries {
        let indent = "  ".repeat(path.components().count().saturating_sub(1));
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy())
            .unwrap_or_default();
        out.push_str(&format!(
            "{indent}{name}{}\n",
            if *is_dir { "/" } else { "" }
        ));
    }
    out.push_str("```\n");
    if truncated {
        let top_dirs: Vec<String> = WalkBuilder::new(root)
            .hidden(false)
            .max_depth(Some(1))
            .sort_by_file_name(|a, b| a.cmp(b))
            .filter_entry(|e| e.file_name() != ".git")
            .build()
            .filter_map(Result::ok)
            .filter(|e| e.depth() == 1 && e.file_type().is_some_and(|t| t.is_dir()))
            .map(|e| format!("{}/", e.file_name().to_string_lossy()))
            .collect();
        out.push_str(&format!(
            "Tree truncated: {} of {total} entries shown. Top-level directories: {}\n",
            cap,
            top_dirs.join(", ")
        ));
    }
    out
}

/// AIRLOK.md at the root, else CLAUDE.md, else AGENTS.md.
fn project_instructions(root: &Path) -> Instructions {
    for (i, name) in INSTRUCTION_FILES.iter().enumerate() {
        let path = root.join(name);
        if let Ok(body) = std::fs::read_to_string(&path) {
            let label = if i == 0 {
                format!("Project instructions ({name})")
            } else {
                format!(
                    "Project instructions ({name}, loaded because {} is absent)",
                    INSTRUCTION_FILES[0]
                )
            };
            return Some((path, label, body));
        }
    }
    None
}
