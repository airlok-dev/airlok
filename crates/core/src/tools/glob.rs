use std::path::{Path, PathBuf};

use async_trait::async_trait;
use serde_json::{json, Value};

use super::{required_str, resolve, walk, Tool, ToolError};

pub const MAX_RESULTS: usize = 500;

pub struct Glob {
    cwd: PathBuf,
}

impl Glob {
    pub fn new(cwd: &Path) -> Self {
        Self {
            cwd: cwd.to_path_buf(),
        }
    }
}

#[async_trait]
impl Tool for Glob {
    fn name(&self) -> &str {
        "glob"
    }

    fn description(&self) -> &str {
        "List files whose path matches a glob such as `**/*.rs` or `src/*.toml`, relative to `path` \
         (default: the working directory). Respects .gitignore. Returns at most 500 paths."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Glob pattern, matched against paths relative to `path`" },
                "path": { "type": "string", "description": "Directory to search (default: working directory)" }
            },
            "required": ["pattern"]
        })
    }

    fn summary(&self, input: &Value) -> String {
        input["pattern"].as_str().unwrap_or("?").to_string()
    }

    async fn execute(&self, input: Value) -> Result<String, ToolError> {
        let pattern = required_str(&input, "pattern")?;
        let root = resolve(&self.cwd, input["path"].as_str().unwrap_or("."));
        let matcher = globset::Glob::new(pattern)
            .map_err(|e| ToolError::InvalidInput(format!("invalid glob: {e}")))?
            .compile_matcher();
        let mut paths: Vec<String> = Vec::new();
        let mut total = 0;
        for (rel, is_dir) in walk(&root, None) {
            if is_dir || !matcher.is_match(&rel) {
                continue;
            }
            total += 1;
            if paths.len() < MAX_RESULTS {
                paths.push(rel.display().to_string());
            }
        }
        if paths.is_empty() {
            return Ok(format!("no files match `{pattern}`"));
        }
        let mut out = paths.join("\n");
        out.push('\n');
        if total > MAX_RESULTS {
            out.push_str(&format!(
                "[truncated: {MAX_RESULTS} of {total} matches shown; use a narrower pattern or path]\n"
            ));
        }
        Ok(out)
    }
}
