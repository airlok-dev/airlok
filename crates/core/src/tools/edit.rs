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
            required_str(input, "search")?,
            required_str(input, "replace")?,
        )?;
        Ok((path, old, new))
    }
}

/// Replaces `search` with `replace` when it occurs exactly once.
pub fn apply_edit(old: &str, search: &str, replace: &str) -> Result<String, ToolError> {
    if search.is_empty() {
        return Err(ToolError::InvalidInput("search text is empty".into()));
    }
    match old.matches(search).count() {
        0 => Err(ToolError::InvalidInput(
            "search text not found in the file".into(),
        )),
        1 => Ok(old.replacen(search, replace, 1)),
        n => Err(ToolError::InvalidInput(format!(
            "search text matches {n} times; include more surrounding context so it matches once"
        ))),
    }
}

#[async_trait]
impl Tool for EditFile {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Replace one exact occurrence of `search` with `replace` in a text file. \
         `search` must match exactly once; include enough context to make it unique."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path of the file to edit" },
                "search": { "type": "string", "description": "Exact text to find, occurring once" },
                "replace": { "type": "string", "description": "Text to put in its place" }
            },
            "required": ["path", "search", "replace"]
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
        assert!(err.contains("matches 2 times"), "{err}");
        let err = apply_edit("a b a", "z", "c").unwrap_err().to_string();
        assert!(err.contains("not found"), "{err}");
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
