//! Without a controlling terminal and without --yes, airlok must refuse
//! to start rather than hang on a prompt or run unconfirmed.

use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::Command;

fn scratch(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "airlok-nontty-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Runs the binary in a new session so `/dev/tty` cannot be opened, with
/// an empty config home and no provider key in the environment.
fn run_detached(cwd: &PathBuf, args: &[&str]) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_airlok"));
    command
        .args(args)
        .current_dir(cwd)
        .env("XDG_CONFIG_HOME", cwd)
        .env_remove("ANTHROPIC_API_KEY")
        .env_remove("OPENAI_API_KEY")
        .env_remove("AZURE_OPENAI_API_KEY");
    // SAFETY: setsid only detaches the child from the controlling terminal
    // and runs before exec in the forked child.
    unsafe {
        command.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }
    command.output().unwrap()
}

#[test]
fn refuses_to_run_without_a_terminal_unless_yes_is_passed() {
    let dir = scratch("refuse");
    let output = run_detached(&dir, &["create hello.txt containing hello"]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        stderr.contains("confirmations are on but no terminal is available"),
        "{stderr}"
    );
    assert!(stderr.contains("--yes"), "{stderr}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn yes_passes_the_terminal_check() {
    let dir = scratch("yes");
    let output = run_detached(&dir, &["--yes", "create hello.txt containing hello"]);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        stderr.contains("warning: confirmations are off"),
        "{stderr}"
    );
    // It got past the terminal check and failed on the missing key instead.
    assert!(!stderr.contains("no terminal is available"), "{stderr}");
    assert!(stderr.contains("ANTHROPIC_API_KEY is not set"), "{stderr}");
    std::fs::remove_dir_all(dir).unwrap();
}
