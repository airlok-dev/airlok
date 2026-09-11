//! The read-only tools: gitignore handling, caps, paging, binary refusal,
//! and result truncation, all driven through the agent.

use std::path::Path;
use std::process::Command;

use airlok_llm::ContentBlock;
use airlok_tests::{agent, reply, tool_call, MockProvider, RecordingOutput, Shown, TempDir};
use serde_json::{json, Value};

fn git_init(dir: &Path) {
    let status = Command::new("git")
        .args(["init", "-q", "-b", "main"])
        .current_dir(dir)
        .status()
        .unwrap();
    assert!(status.success());
}

/// Runs one tool call through the agent and returns the result content
/// and error flag the model received, plus what the user saw.
async fn call(dir: &Path, tool: &str, input: Value) -> (String, bool, Vec<Shown>) {
    let provider = MockProvider::scripted(vec![tool_call("toolu_1", tool, input), reply("ok")]);
    let mut out = RecordingOutput::default();
    agent(provider.clone(), dir)
        .run("go", &mut out)
        .await
        .unwrap();
    let requests = provider.requests();
    match &requests[1].messages.last().unwrap().content[0] {
        ContentBlock::ToolResult {
            content, is_error, ..
        } => (content.clone(), *is_error, out.events),
        other => panic!("expected a tool result, got {other:?}"),
    }
}

fn fixture(label: &str) -> TempDir {
    let dir = TempDir::new(label);
    git_init(dir.path());
    std::fs::write(dir.path().join(".gitignore"), "build/\n").unwrap();
    std::fs::create_dir_all(dir.path().join("src")).unwrap();
    std::fs::create_dir_all(dir.path().join("build")).unwrap();
    std::fs::write(
        dir.path().join("src/main.rs"),
        "fn main() {\n    needle();\n}\n",
    )
    .unwrap();
    std::fs::write(dir.path().join("src/lib.rs"), "pub fn needle() {}\n").unwrap();
    std::fs::write(dir.path().join("README.md"), "needle in docs\n").unwrap();
    std::fs::write(dir.path().join("build/out.rs"), "needle in build output\n").unwrap();
    dir
}

#[tokio::test]
async fn grep_respects_gitignore_and_include_and_never_asks() {
    let dir = fixture("grep");
    let (content, is_error, shown) = call(dir.path(), "grep", json!({"pattern": "needle"})).await;
    assert!(!is_error, "{content}");
    assert_eq!(
        content,
        "README.md:1:needle in docs\nsrc/lib.rs:1:pub fn needle() {}\nsrc/main.rs:2:    needle();\n"
    );
    assert!(!shown
        .iter()
        .any(|e| matches!(e, Shown::ConfirmCommand { .. } | Shown::ConfirmWrite { .. })));

    let (content, _, _) = call(
        dir.path(),
        "grep",
        json!({"pattern": "needle", "include": "*.rs"}),
    )
    .await;
    assert_eq!(
        content,
        "src/lib.rs:1:pub fn needle() {}\nsrc/main.rs:2:    needle();\n"
    );

    let (content, _, _) = call(
        dir.path(),
        "grep",
        json!({"pattern": "needle", "path": "src/lib.rs"}),
    )
    .await;
    assert_eq!(content, "lib.rs:1:pub fn needle() {}\n");

    let (content, is_error, _) = call(dir.path(), "grep", json!({"pattern": "nothing-here"})).await;
    assert!(!is_error);
    assert_eq!(content, "no matches for `nothing-here`");

    let (content, is_error, _) = call(dir.path(), "grep", json!({"pattern": "("})).await;
    assert!(is_error);
    assert!(content.contains("invalid regex"), "{content}");
}

#[tokio::test]
async fn grep_caps_at_200_lines_and_skips_binaries() {
    let dir = fixture("grep-cap");
    let many: String = (0..300).map(|i| format!("needle {i}\n")).collect();
    std::fs::write(dir.path().join("many.txt"), many).unwrap();
    std::fs::write(dir.path().join("blob.bin"), b"needle\0binary").unwrap();

    let (content, _, _) = call(
        dir.path(),
        "grep",
        json!({"pattern": "needle", "include": "*.txt"}),
    )
    .await;
    assert_eq!(
        content
            .lines()
            .filter(|l| l.starts_with("many.txt:"))
            .count(),
        200
    );
    assert!(
        content.ends_with(
            "[truncated: 200 of 300 matching lines shown; narrow the pattern, path, or include]\n"
        ),
        "{content}"
    );

    let (content, _, _) = call(
        dir.path(),
        "grep",
        json!({"pattern": "needle", "include": "*.bin"}),
    )
    .await;
    assert_eq!(content, "no matches for `needle`");
}

#[tokio::test]
async fn glob_respects_gitignore_and_caps_at_500() {
    let dir = fixture("glob");
    let (content, is_error, shown) = call(dir.path(), "glob", json!({"pattern": "**/*.rs"})).await;
    assert!(!is_error, "{content}");
    assert_eq!(content, "src/lib.rs\nsrc/main.rs\n");
    assert!(!shown
        .iter()
        .any(|e| matches!(e, Shown::ConfirmCommand { .. } | Shown::ConfirmWrite { .. })));

    let (content, _, _) = call(
        dir.path(),
        "glob",
        json!({"pattern": "*.rs", "path": "src"}),
    )
    .await;
    assert_eq!(content, "lib.rs\nmain.rs\n");

    std::fs::create_dir_all(dir.path().join("gen")).unwrap();
    for i in 0..600 {
        std::fs::write(dir.path().join(format!("gen/f{i:04}.txt")), "").unwrap();
    }
    let (content, _, _) = call(dir.path(), "glob", json!({"pattern": "gen/*.txt"})).await;
    assert_eq!(
        content.lines().filter(|l| l.starts_with("gen/")).count(),
        500
    );
    assert!(content
        .ends_with("[truncated: 500 of 600 matches shown; use a narrower pattern or path]\n"));

    let (content, _, _) = call(dir.path(), "glob", json!({"pattern": "*.nothing"})).await;
    assert_eq!(content, "no files match `*.nothing`");
}

#[tokio::test]
async fn list_dir_shows_type_and_size() {
    let dir = fixture("list");
    let (content, is_error, _) = call(dir.path(), "list_dir", json!({})).await;
    assert!(!is_error, "{content}");
    assert_eq!(
        content,
        "dir         .git/\nfile       7  .gitignore\nfile      15  README.md\ndir         build/\ndir         src/\n"
    );
    let (content, is_error, _) = call(dir.path(), "list_dir", json!({"path": "missing"})).await;
    assert!(is_error);
    assert!(content.starts_with("error: "), "{content}");
}

#[tokio::test]
async fn read_file_pages_and_refuses_binaries() {
    let dir = fixture("read");
    let text: String = (1..=10).map(|i| format!("line {i}\n")).collect();
    std::fs::write(dir.path().join("ten.txt"), &text).unwrap();
    std::fs::write(dir.path().join("blob.bin"), b"\x7fELF\0\0\0").unwrap();

    let (content, _, _) = call(dir.path(), "read_file", json!({"path": "ten.txt"})).await;
    assert_eq!(content, text);
    let (content, _, shown) = call(
        dir.path(),
        "read_file",
        json!({"path": "ten.txt", "offset": 4, "limit": 3}),
    )
    .await;
    assert_eq!(content, "[lines 4-6 of 10]\nline 4\nline 5\nline 6\n");
    assert!(shown.contains(&Shown::ToolCall {
        name: "read_file".into(),
        summary: "ten.txt (from line 4, 3 lines)".into()
    }));

    let (content, is_error, _) = call(dir.path(), "read_file", json!({"path": "blob.bin"})).await;
    assert!(is_error);
    assert!(content.contains("binary file"), "{content}");
    assert!(content.contains("refusing"), "{content}");
}

#[tokio::test]
async fn oversized_results_are_truncated_with_a_paging_hint() {
    let dir = fixture("big");
    let big: String = (0..3000)
        .map(|i| format!("{i:05} {}\n", "x".repeat(20)))
        .collect();
    assert!(big.len() > 50 * 1024);
    std::fs::write(dir.path().join("big.txt"), &big).unwrap();

    let (content, is_error, _) = call(dir.path(), "read_file", json!({"path": "big.txt"})).await;
    assert!(!is_error);
    assert!(content.len() < big.len());
    assert!(content.starts_with("00000 "));
    let marker = format!("[output truncated: showing 51200 of {} bytes.", big.len());
    assert!(
        content.contains(&marker),
        "{}",
        &content[content.len() - 300..]
    );
    assert!(content.contains("read_file with `offset` and `limit`"));
}
