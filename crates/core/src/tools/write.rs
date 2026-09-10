use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::{json, Value};

use super::{required_str, resolve, unified_diff, Plan, Tool, ToolError};

pub struct WriteFile {
    cwd: PathBuf,
}

impl WriteFile {
    pub fn new(cwd: &Path) -> Self {
        Self {
            cwd: cwd.to_path_buf(),
        }
    }
}

#[async_trait]
impl Tool for WriteFile {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Create or overwrite a text file with the given content. Parent directories are created."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path of the file to write" },
                "content": { "type": "string", "description": "Full file content" }
            },
            "required": ["path", "content"]
        })
    }

    fn summary(&self, input: &Value) -> String {
        input["path"].as_str().unwrap_or("?").to_string()
    }

    async fn plan(&self, input: &Value) -> Result<Plan, ToolError> {
        let rel = required_str(input, "path")?;
        let path = resolve(&self.cwd, rel);
        let content = required_str(input, "content")?;
        let old = match tokio::fs::read_to_string(&path).await {
            Ok(old) => old,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => return Err(e.into()),
        };
        let diff = unified_diff(rel, &old, content);
        Ok(Plan::Write { path, diff })
    }

    async fn execute(&self, input: Value) -> Result<String, ToolError> {
        let path = resolve(&self.cwd, required_str(&input, "path")?);
        let content = required_str(&input, "content")?;
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&path, content).await?;
        Ok(format!(
            "wrote {} bytes to {}",
            content.len(),
            path.display()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_file_diff_snapshot() {
        let diff = unified_diff("hello.txt", "", "hello\n");
        assert_eq!(
            diff,
            "--- a/hello.txt\n\
             +++ b/hello.txt\n\
             @@ -0,0 +1 @@\n\
             +hello\n"
        );
    }

    #[test]
    fn overwrite_diff_snapshot() {
        let diff = unified_diff("notes.md", "one\ntwo\nthree\n", "one\n2\nthree\nfour\n");
        assert_eq!(
            diff,
            "--- a/notes.md\n\
             +++ b/notes.md\n\
             @@ -1,3 +1,4 @@\n\
             \x20one\n\
             -two\n\
             +2\n\
             \x20three\n\
             +four\n"
        );
    }
}
