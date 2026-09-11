use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::{json, Value};

use super::{required_str, resolve, Tool, ToolError};

/// Bytes inspected for NUL to decide a file is binary.
const BINARY_PROBE: usize = 8 * 1024;

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
        "Read a UTF-8 text file. Relative paths are resolved from the working directory. \
         For large files pass `offset` (1-based first line) and `limit` (number of lines) to page."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path of the file to read" },
                "offset": { "type": "integer", "minimum": 1, "description": "First line to return, 1-based (default 1)" },
                "limit": { "type": "integer", "minimum": 1, "description": "Number of lines to return (default: all)" }
            },
            "required": ["path"]
        })
    }

    fn summary(&self, input: &Value) -> String {
        let path = input["path"].as_str().unwrap_or("?");
        match (input["offset"].as_u64(), input["limit"].as_u64()) {
            (None, None) => path.to_string(),
            (offset, limit) => format!(
                "{path} (from line {}{})",
                offset.unwrap_or(1),
                limit.map(|l| format!(", {l} lines")).unwrap_or_default()
            ),
        }
    }

    async fn execute(&self, input: Value) -> Result<String, ToolError> {
        let path = resolve(&self.cwd, required_str(&input, "path")?);
        let bytes = tokio::fs::read(&path).await?;
        if bytes.iter().take(BINARY_PROBE).any(|&b| b == 0) {
            return Err(ToolError::InvalidInput(format!(
                "{} is a binary file (contains NUL bytes); refusing to read it",
                path.display()
            )));
        }
        let text = String::from_utf8(bytes).map_err(|_| {
            ToolError::InvalidInput(format!("{} is not valid UTF-8", path.display()))
        })?;
        Ok(page(
            &text,
            input["offset"].as_u64(),
            input["limit"].as_u64(),
        ))
    }
}

/// Returns the requested window of lines with a header, or the whole text
/// when no window was asked for.
fn page(text: &str, offset: Option<u64>, limit: Option<u64>) -> String {
    if offset.is_none() && limit.is_none() {
        return text.to_string();
    }
    let lines: Vec<&str> = text.lines().collect();
    let total = lines.len();
    let start = offset.unwrap_or(1).max(1) as usize;
    let count = limit.map(|l| l as usize).unwrap_or(usize::MAX);
    let end = start.saturating_sub(1).saturating_add(count).min(total);
    if start > total {
        return format!("[lines {start}- of {total}: past the end of the file]\n");
    }
    let mut out = format!("[lines {start}-{end} of {total}]\n");
    for line in &lines[start - 1..end] {
        out.push_str(line);
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paging_windows_lines() {
        let text = "a\nb\nc\nd\ne\n";
        assert_eq!(page(text, None, None), text);
        assert_eq!(page(text, Some(2), Some(2)), "[lines 2-3 of 5]\nb\nc\n");
        assert_eq!(page(text, Some(4), None), "[lines 4-5 of 5]\nd\ne\n");
        assert_eq!(page(text, None, Some(1)), "[lines 1-1 of 5]\na\n");
        assert_eq!(
            page(text, Some(9), Some(2)),
            "[lines 9- of 5: past the end of the file]\n"
        );
    }
}
