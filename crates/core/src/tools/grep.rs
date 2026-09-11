use std::path::{Path, PathBuf};

use async_trait::async_trait;
use grep::regex::RegexMatcher;
use grep::searcher::sinks::UTF8;
use grep::searcher::{BinaryDetection, SearcherBuilder};
use serde_json::{json, Value};

use super::{required_str, resolve, walk, Tool, ToolError};

pub const MAX_LINES: usize = 200;

pub struct Grep {
    cwd: PathBuf,
}

impl Grep {
    pub fn new(cwd: &Path) -> Self {
        Self {
            cwd: cwd.to_path_buf(),
        }
    }
}

#[async_trait]
impl Tool for Grep {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search file contents with a regex under `path` (default: the working directory), \
         optionally only in files matching the `include` glob (for example `*.rs`). Respects \
         .gitignore and skips binary files. Returns `file:line:text`, at most 200 lines."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Regular expression to search for" },
                "path": { "type": "string", "description": "Directory or file to search (default: working directory)" },
                "include": { "type": "string", "description": "Only search files whose path matches this glob" }
            },
            "required": ["pattern"]
        })
    }

    fn summary(&self, input: &Value) -> String {
        let pattern = input["pattern"].as_str().unwrap_or("?");
        match input["include"].as_str() {
            Some(include) => format!("{pattern} in {include}"),
            None => pattern.to_string(),
        }
    }

    async fn execute(&self, input: Value) -> Result<String, ToolError> {
        let pattern = required_str(&input, "pattern")?;
        let root = resolve(&self.cwd, input["path"].as_str().unwrap_or("."));
        let include = input["include"]
            .as_str()
            .map(|g| {
                globset::Glob::new(g)
                    .map(|g| g.compile_matcher())
                    .map_err(|e| ToolError::InvalidInput(format!("invalid include glob: {e}")))
            })
            .transpose()?;
        let matcher = RegexMatcher::new(pattern)
            .map_err(|e| ToolError::InvalidInput(format!("invalid regex: {e}")))?;
        let mut searcher = SearcherBuilder::new()
            .line_number(true)
            .binary_detection(BinaryDetection::quit(b'\x00'))
            .build();

        let mut lines: Vec<String> = Vec::new();
        let mut total = 0usize;
        let files: Vec<PathBuf> = if root.is_file() {
            vec![PathBuf::from(
                root.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            )]
        } else {
            walk(&root, None)
                .into_iter()
                .filter(|(_, is_dir)| !is_dir)
                .map(|(rel, _)| rel)
                .collect()
        };
        let base = if root.is_file() {
            root.parent().map(Path::to_path_buf).unwrap_or_default()
        } else {
            root.clone()
        };
        for rel in files {
            if include.as_ref().is_some_and(|m| !m.is_match(&rel)) {
                continue;
            }
            let shown = rel.display().to_string();
            let result = searcher.search_path(
                &matcher,
                base.join(&rel),
                UTF8(|line_number, line| {
                    total += 1;
                    if lines.len() < MAX_LINES {
                        lines.push(format!("{shown}:{line_number}:{}", line.trim_end()));
                    }
                    Ok(true)
                }),
            );
            if let Err(e) = result {
                tracing::debug!(path = %shown, error = %e, "grep skipped file");
            }
        }
        if lines.is_empty() {
            return Ok(format!("no matches for `{pattern}`"));
        }
        let mut out = lines.join("\n");
        out.push('\n');
        if total > MAX_LINES {
            out.push_str(&format!(
                "[truncated: {MAX_LINES} of {total} matching lines shown; narrow the pattern, path, or include]\n"
            ));
        }
        Ok(out)
    }
}
