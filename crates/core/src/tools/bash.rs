use std::path::{Path, PathBuf};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::process::Command;

use super::{required_str, Plan, Tool, ToolError};

pub struct Bash {
    cwd: PathBuf,
    timeout: Duration,
}

impl Bash {
    pub fn new(cwd: &Path, timeout: Duration) -> Self {
        Self {
            cwd: cwd.to_path_buf(),
            timeout,
        }
    }
}

#[async_trait]
impl Tool for Bash {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Run a shell command in the working directory and return its stdout, stderr, and exit code."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "The command to run with sh -c" }
            },
            "required": ["command"]
        })
    }

    fn summary(&self, input: &Value) -> String {
        input["command"].as_str().unwrap_or("?").to_string()
    }

    async fn plan(&self, input: &Value) -> Result<Plan, ToolError> {
        Ok(Plan::Command {
            command: required_str(input, "command")?.to_string(),
        })
    }

    async fn execute(&self, input: Value) -> Result<String, ToolError> {
        let command = required_str(&input, "command")?;
        let child = Command::new("sh")
            .arg("-c")
            .arg(command)
            .current_dir(&self.cwd)
            .kill_on_drop(true)
            .output();
        let output = tokio::time::timeout(self.timeout, child)
            .await
            .map_err(|_| ToolError::Timeout(self.timeout))??;

        let mut report = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stderr.trim().is_empty() {
            report.push_str("\n[stderr]\n");
            report.push_str(&stderr);
        }
        if !output.status.success() {
            report.push_str(&format!(
                "\n[exit code {}]",
                output.status.code().unwrap_or(-1)
            ));
        }
        if report.trim().is_empty() {
            report.push_str("(no output)");
        }
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn captures_stdout_stderr_and_exit_code() {
        let bash = Bash::new(Path::new("."), Duration::from_secs(5));
        let out = bash
            .execute(json!({"command": "echo out; echo err >&2; exit 3"}))
            .await
            .unwrap();
        assert!(out.starts_with("out\n"));
        assert!(out.contains("[stderr]\nerr"));
        assert!(out.ends_with("[exit code 3]"));
    }

    #[tokio::test]
    async fn times_out() {
        let bash = Bash::new(Path::new("."), Duration::from_millis(100));
        let err = bash
            .execute(json!({"command": "sleep 5"}))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::Timeout(_)));
    }
}
