use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::{json, Value};

use super::{required_str, resolve, Tool, ToolError};

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
