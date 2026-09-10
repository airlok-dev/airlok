use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::{json, Value};

use super::{required_str, resolve, Tool, ToolError};

pub struct ReadFile {
    cwd: PathBuf,
}

impl ReadFile {
    pub fn new(cwd: &Path) -> Self {
        Self {
            cwd: cwd.to_path_buf(),
        }
    }
}

#[async_trait]
impl Tool for ReadFile {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read a UTF-8 text file. Relative paths are resolved from the working directory."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path of the file to read" }
            },
            "required": ["path"]
        })
    }

    fn summary(&self, input: &Value) -> String {
        input["path"].as_str().unwrap_or("?").to_string()
    }

    async fn execute(&self, input: Value) -> Result<String, ToolError> {
        let path = resolve(&self.cwd, required_str(&input, "path")?);
        Ok(tokio::fs::read_to_string(path).await?)
    }
}
