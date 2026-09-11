//! Golden test for the context block against a fixture repository.

use std::path::Path;
use std::process::Command;

use airlok_core::context::{build, ContextInput};
use airlok_tests::TempDir;

fn git(dir: &Path, args: &[&str]) {
    let status = Command::new("git")
        .args([
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=t",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap()
        .status;
    assert!(status.success(), "git {args:?} failed");
}

/// A small repo: two commits, an ignored build dir, a dotfile, one dirty file.
fn fixture(label: &str) -> TempDir {
    let dir = TempDir::new(label);
    let root = dir.path();
    git(root, &["init", "-q", "-b", "main"]);
    std::fs::write(root.join(".gitignore"), "target/\n").unwrap();
    std::fs::write(root.join("Cargo.toml"), "[package]\nname = \"fixture\"\n").unwrap();
    std::fs::create_dir_all(root.join("src/deep/deeper/deepest")).unwrap();
    std::fs::write(root.join("src/main.rs"), "fn main() {}\n").unwrap();
    std::fs::write(root.join("src/deep/deeper/deepest/leaf.rs"), "").unwrap();
    std::fs::create_dir_all(root.join("target")).unwrap();
    std::fs::write(root.join("target/ignored.o"), "").unwrap();
    std::fs::write(root.join("CLAUDE.md"), "Use two-space indents.\n").unwrap();
    git(root, &["add", "."]);
    git(root, &["commit", "-q", "-m", "init"]);
    std::fs::write(root.join("src/lib.rs"), "").unwrap();
    git(root, &["add", "."]);
    git(root, &["commit", "-q", "-m", "add lib"]);
    std::fs::write(root.join("src/main.rs"), "fn main() { println!(); }\n").unwrap();
    dir
}

fn normalise(text: &str, root: &Path, home: &Path) -> String {
    let shell = std::env::var("SHELL")
        .ok()
        .and_then(|s| {
            Path::new(&s)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "unknown".into());
    let canonical = root.canonicalize().unwrap();
    text.replace(&canonical.display().to_string(), "<root>")
        .replace(&root.display().to_string(), "<root>")
        .replace(&home.display().to_string(), "<home>")
        .replace(&format!("- os: {}\n", std::env::consts::OS), "- os: <os>\n")
        .replace(&format!("- shell: {shell}\n"), "- shell: <shell>\n")
}

#[test]
fn golden_context_block() {
    let dir = fixture("golden");
    // The user-level file lives outside the repo, as ~/.config would.
    let home = TempDir::new("golden-home");
    let user = home.path().join("AIRLOK.md");
    std::fs::write(&user, "Always answer in British English.\n").unwrap();

    let block = build(&ContextInput {
        cwd: dir.path(),
        user_instructions: Some(&user),
        max_bytes: 32 * 1024,
    });

    let expected = "\
# Context

## Environment

- cwd: <root>
- os: <os>
- shell: <shell>
- git: repository at <root>, branch main

## Git status

```
 M src/main.rs
```

## Recent commits

- add lib
- init

## Files

Tree of <root> (depth 4, .gitignore respected):
```
.gitignore
CLAUDE.md
Cargo.toml
src/
  deep/
    deeper/
      deepest/
  lib.rs
  main.rs
```

## Project instructions (CLAUDE.md, loaded because AIRLOK.md is absent)

Use two-space indents.

## User instructions (<home>/AIRLOK.md)

Always answer in British English.
";
    assert_eq!(normalise(&block.text, dir.path(), home.path()), expected);
    // git reports the canonical toplevel, which is where instructions are read.
    let root = dir.path().canonicalize().unwrap();
    assert_eq!(block.instruction_files, vec![root.join("CLAUDE.md"), user]);
}

#[test]
fn airlok_md_wins_over_claude_md_and_a_plain_dir_has_no_git() {
    let dir = fixture("precedence");
    std::fs::write(dir.path().join("AIRLOK.md"), "airlok rules\n").unwrap();
    let block = build(&ContextInput {
        cwd: dir.path(),
        user_instructions: None,
        max_bytes: 32 * 1024,
    });
    assert!(block
        .text
        .contains("## Project instructions (AIRLOK.md)\n\nairlok rules\n"));
    assert!(!block.text.contains("CLAUDE.md, loaded"));
    assert_eq!(
        block.instruction_files,
        vec![dir.path().canonicalize().unwrap().join("AIRLOK.md")]
    );

    let plain = TempDir::new("plain");
    std::fs::write(plain.path().join("notes.txt"), "").unwrap();
    let block = build(&ContextInput {
        cwd: plain.path(),
        user_instructions: Some(&plain.path().join("missing.md")),
        max_bytes: 32 * 1024,
    });
    assert!(block.text.contains("- git: not a repository\n"));
    assert!(!block.text.contains("## Git status"));
    assert!(block.text.contains("notes.txt\n"));
    assert!(block.instruction_files.is_empty());
}

#[test]
fn truncation_cuts_the_tree_before_the_instructions() {
    let dir = fixture("truncate");
    // Enough nested files that the full tree is several hundred bytes
    // larger than the top-level listing.
    for i in 0..30 {
        std::fs::write(dir.path().join(format!("src/deep/file{i:02}.rs")), "").unwrap();
    }
    let long_instructions = "rule line\n".repeat(60);
    std::fs::write(dir.path().join("AIRLOK.md"), &long_instructions).unwrap();

    let full = build(&ContextInput {
        cwd: dir.path(),
        user_instructions: None,
        max_bytes: 32 * 1024,
    });
    assert!(full.text.contains("(depth 4, .gitignore respected)"));

    // Tight enough that the full tree does not fit, loose enough that the
    // short tree plus the whole instructions do.
    let tight = full.text.len() - 100;
    let cut_tree = build(&ContextInput {
        cwd: dir.path(),
        user_instructions: None,
        max_bytes: tight,
    });
    assert!(cut_tree.text.len() <= tight);
    assert!(cut_tree
        .text
        .contains("(tree cut to fit context.max_bytes)"));
    assert!(!cut_tree.text.contains("deeper/"));
    assert!(cut_tree
        .text
        .contains(&long_instructions.trim_end().to_string()));
    assert!(!cut_tree.text.contains("[instructions truncated"));

    // Tighter still: the instructions are cut, with a marker.
    let tighter = cut_tree.text.len() - 200;
    let cut_both = build(&ContextInput {
        cwd: dir.path(),
        user_instructions: None,
        max_bytes: tighter,
    });
    assert!(
        cut_both.text.len() <= tighter,
        "{} > {tighter}",
        cut_both.text.len()
    );
    assert!(cut_both
        .text
        .contains("(tree cut to fit context.max_bytes)"));
    assert!(cut_both
        .text
        .contains("[instructions truncated to fit context.max_bytes]"));
    assert!(cut_both.text.contains("rule line\n"));
}

#[test]
fn tree_is_capped_at_200_entries() {
    let dir = TempDir::new("big");
    git(dir.path(), &["init", "-q", "-b", "main"]);
    std::fs::create_dir_all(dir.path().join("many")).unwrap();
    std::fs::create_dir_all(dir.path().join("other")).unwrap();
    for i in 0..250 {
        std::fs::write(dir.path().join(format!("many/f{i:03}.txt")), "").unwrap();
    }
    let block = build(&ContextInput {
        cwd: dir.path(),
        user_instructions: None,
        max_bytes: 1024 * 1024,
    });
    assert!(
        block.text.contains(
            "Tree truncated: 200 of 252 entries shown. Top-level directories: many/, other/\n"
        ),
        "{}",
        block.text
    );
    assert_eq!(
        block.text.matches("f0").count() + block.text.matches("f1").count(),
        199 - 1 + 1,
        "{}",
        block.text
    );
}
