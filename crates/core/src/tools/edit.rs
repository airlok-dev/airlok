use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::{json, Value};

use super::{required_str, resolve, unified_diff, Plan, Tool, ToolError};

pub struct EditFile {
    cwd: PathBuf,
}

impl EditFile {
    pub fn new(cwd: &Path) -> Self {
        Self {
            cwd: cwd.to_path_buf(),
        }
    }

    async fn edited(&self, input: &Value) -> Result<(PathBuf, String, String), ToolError> {
        let rel = required_str(input, "path")?;
        let path = resolve(&self.cwd, rel);
        let old = tokio::fs::read_to_string(&path).await?;
        let new = apply_edit(
            &old,
            required_str(input, "old")?,
            required_str(input, "new")?,
        )?;
        Ok((path, old, new))
    }
}

/// Replaces `old` with `new` when `old` occurs exactly once.
pub fn apply_edit(text: &str, old: &str, new: &str) -> Result<String, ToolError> {
    if old.is_empty() {
        return Err(ToolError::InvalidInput("`old` is empty".into()));
    }
    match text.matches(old).count() {
        1 => Ok(text.replacen(old, new, 1)),
        n => Err(ToolError::InvalidInput(format!(
            "found {n} matches for `old`, need exactly 1{}",
            if n == 0 {
                "; check the exact text, including whitespace"
            } else {
                "; include more surrounding context so it is unique"
            }
        ))),
    }
}

#[async_trait]
impl Tool for EditFile {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Replace one exact occurrence of `old` with `new` in an existing text file. \
         `old` must match exactly once; include enough surrounding context to make it unique. \
         Prefer this over write_file for changes to existing files."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path of the file to edit" },
                "old": { "type": "string", "description": "Exact text to find, occurring once" },
                "new": { "type": "string", "description": "Text to put in its place" }
            },
            "required": ["path", "old", "new"]
        })
    }

    fn summary(&self, input: &Value) -> String {
        input["path"].as_str().unwrap_or("?").to_string()
    }

    async fn plan(&self, input: &Value) -> Result<Plan, ToolError> {
        let (path, old, new) = self.edited(input).await?;
        let diff = unified_diff(input["path"].as_str().unwrap_or("?"), &old, &new);
        Ok(Plan::Write { path, diff })
    }

    async fn execute(&self, input: Value) -> Result<String, ToolError> {
        let (path, _, new) = self.edited(&input).await?;
        tokio::fs::write(&path, &new).await?;
        Ok(format!("edited {}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edit_must_match_exactly_once() {
        assert_eq!(apply_edit("a b a", "b", "c").unwrap(), "a c a");
        let err = apply_edit("a b a", "a", "c").unwrap_err().to_string();
        assert!(err.contains("found 2 matches"), "{err}");
        let err = apply_edit("a b a", "z", "c").unwrap_err().to_string();
        assert!(err.contains("found 0 matches"), "{err}");
        assert!(apply_edit("a", "", "c").is_err());
    }

    #[test]
    fn edit_diff_snapshot() {
        let old = "fn main() {\n    println!(\"hello\");\n    println!(\"world\");\n}\n";
        let new = apply_edit(old, "println!(\"world\");", "println!(\"there\");").unwrap();
        let diff = unified_diff("src/main.rs", old, &new);
        assert_eq!(
            diff,
            "--- a/src/main.rs\n\
             +++ b/src/main.rs\n\
             @@ -1,4 +1,4 @@\n\
             \x20fn main() {\n\
             \x20    println!(\"hello\");\n\
             -    println!(\"world\");\n\
             +    println!(\"there\");\n\
             \x20}\n"
        );
    }
}
