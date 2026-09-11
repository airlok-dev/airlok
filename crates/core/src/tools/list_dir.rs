use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::{json, Value};

use super::{resolve, Tool, ToolError};

pub struct ListDir {
    cwd: PathBuf,
}

impl ListDir {
    pub fn new(cwd: &Path) -> Self {
        Self {
            cwd: cwd.to_path_buf(),
        }
    }
}

#[async_trait]
impl Tool for ListDir {
    fn name(&self) -> &str {
        "list_dir"
    }

    fn description(&self) -> &str {
        "List the entries of a directory with their type and size, sorted by name."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Directory to list (default: working directory)" }
            }
        })
    }

    fn summary(&self, input: &Value) -> String {
        input["path"].as_str().unwrap_or(".").to_string()
    }

    async fn execute(&self, input: Value) -> Result<String, ToolError> {
        let dir = resolve(&self.cwd, input["path"].as_str().unwrap_or("."));
        let mut entries = Vec::new();
        let mut read = tokio::fs::read_dir(&dir).await?;
        while let Some(entry) = read.next_entry().await? {
            let meta = entry.metadata().await?;
            let name = entry.file_name().to_string_lossy().into_owned();
            entries.push(if meta.is_dir() {
                (name.clone(), format!("dir         {name}/"))
            } else if meta.is_symlink() {
                (name.clone(), format!("link        {name}"))
            } else {
                (name.clone(), format!("file {:>7}  {name}", meta.len()))
            });
        }
        entries.sort();
        if entries.is_empty() {
            return Ok(format!("{} is empty", dir.display()));
        }
        Ok(entries.into_iter().map(|(_, line)| line + "\n").collect())
    }
}
