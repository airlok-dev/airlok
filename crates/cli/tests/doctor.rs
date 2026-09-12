//! `airlok doctor` reports each check on its own and exits non-zero when
//! any of them fails, so it is usable as a CI gate.

use std::path::PathBuf;
use std::process::Command;

fn scratch(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "airlok-doctor-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// No key anywhere, and a config home of its own, so the key check fails
/// without reaching the network: the request check is skipped when there
/// is nothing to send.
fn run_doctor(cwd: &PathBuf) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_airlok"))
        .arg("doctor")
        .current_dir(cwd)
        .env("XDG_CONFIG_HOME", cwd)
        .env("XDG_DATA_HOME", cwd)
        .env_remove("ANTHROPIC_API_KEY")
        .env_remove("OPENAI_API_KEY")
        .env_remove("AZURE_OPENAI_API_KEY")
        .output()
        .expect("airlok doctor runs")
}

#[test]
fn checks_pass_and_fail_independently_and_a_failure_exits_non_zero() {
    let dir = scratch("nokey");
    let out = run_doctor(&dir);
    let text = String::from_utf8_lossy(&out.stdout).into_owned();

    assert!(
        text.lines().any(|l| l.starts_with("pass ")),
        "some checks still pass: {text}"
    );
    assert!(
        text.lines().any(|l| l.starts_with("FAIL provider key")),
        "the missing key is reported as its own failure: {text}"
    );
    assert!(
        text.contains("not attempted: no key to send"),
        "the live request is skipped rather than failing for the wrong reason: {text}"
    );
    assert!(
        !out.status.success(),
        "a failing check exits non-zero so CI catches it"
    );

    // Never the value, only where it looked.
    assert!(
        text.contains("ANTHROPIC_API_KEY"),
        "the source is named: {text}"
    );
}
