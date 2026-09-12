//! Child processes do not outlive airlok. Its own test binary, because
//! killing every MCP child is global to the process.

use std::path::Path;
use std::time::Duration;

use airlok_core::config::{Config, ConfigFile};
use airlok_core::mcp;

const SERVER: &str = env!("CARGO_BIN_EXE_mock-mcp-server");

/// The process state from `ps`: empty when it is gone, `Z` once it has
/// exited and is waiting to be reaped. Neither is running.
fn running(pid: u32) -> bool {
    let output = std::process::Command::new("ps")
        .args(["-o", "state=", "-p", &pid.to_string()])
        .output()
        .expect("ps");
    let state = String::from_utf8_lossy(&output.stdout).trim().to_string();
    !state.is_empty() && !state.starts_with('Z')
}

async fn gone(pid: u32) -> bool {
    for _ in 0..40 {
        if !running(pid) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    false
}

#[tokio::test]
async fn a_stdio_child_is_killed_on_the_way_out() {
    let text = format!("[[mcp]]\nname = \"mock\"\ncommand = \"{SERVER}\"\n");
    let file = ConfigFile::parse(&text, Path::new("test.toml")).unwrap();
    let config = Config::resolve(file, std::env::temp_dir());

    let connection = mcp::connect(&config.mcp[0]).await.unwrap();
    let pid = connection.child.expect("a stdio server runs a child");
    assert!(running(pid), "the server should be running");

    // What a signal handler calls, where no destructor runs.
    mcp::kill_children();
    assert!(gone(pid).await, "the child outlived kill_children");

    drop(connection);
}
