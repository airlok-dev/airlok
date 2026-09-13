//! Layered configuration.
//!
//! Precedence, highest first: CLI flags, `./airlok.toml` in the working
//! directory, the user file (`$XDG_CONFIG_HOME/airlok/config.toml`, which
//! is `~/.config/airlok/config.toml` on macOS and Linux), built-in defaults.
//!
//! [`ConfigFile`] is the on-disk shape where everything is optional;
//! [`Config`] is the resolved shape the agent runs with.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use serde::{Deserialize, Serialize};

pub const USER_CONFIG_RELATIVE: &str = "airlok/config.toml";
pub const PROJECT_CONFIG_NAME: &str = "airlok.toml";

/// `text` with its `[safety]` section replaced by `safety`, or the section
/// appended when it has none. Only that block is rewritten, so comments
/// and settings elsewhere in the file survive; comments inside the block
/// do not.
pub fn with_safety_section(text: &str, safety: &SafetyConfig) -> Result<String, String> {
    let section = SafetySection {
        confirm_writes: Some(safety.confirm_writes),
        confirm_bash: Some(safety.confirm_bash),
        confirm_mcp: Some(safety.confirm_mcp),
        confirm_images: Some(safety.confirm_images),
        bash_allowlist: Some(safety.bash_allowlist.clone()),
        bash_denylist: Some(safety.bash_denylist.clone()),
    };
    let body = toml::to_string(&section).map_err(|e| e.to_string())?;
    let block = format!("[safety]\n{body}");

    let lines: Vec<&str> = text.lines().collect();
    let start = lines.iter().position(|line| line.trim() == "[safety]");
    let Some(start) = start else {
        let mut out = text.trim_end().to_string();
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str(&block);
        return Ok(out);
    };
    let end = lines[start + 1..]
        .iter()
        .position(|line| line.trim_start().starts_with('['))
        .map(|at| start + 1 + at)
        .unwrap_or(lines.len());
    let mut out = String::new();
    for line in &lines[..start] {
        out.push_str(line);
        out.push('\n');
    }
    out.push_str(&block);
    if end < lines.len() {
        out.push('\n');
        for line in &lines[end..] {
            out.push_str(line);
            out.push('\n');
        }
    }
    Ok(out)
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("cannot read {path}: {source}")]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("cannot parse {path}: {source}")]
    Parse {
        path: PathBuf,
        source: toml::de::Error,
    },
    #[error("{0} is not set")]
    KeyEnvUnset(String),
    #[error("api_key_cmd `{command}` {problem}")]
    KeyCommand { command: String, problem: String },
    #[error("mcp server `{server}`: {what} `{command}` {problem}")]
    McpCommand {
        server: String,
        what: String,
        command: String,
        problem: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderName {
    Anthropic,
    #[serde(rename = "openai")]
    OpenAi,
}

impl ProviderName {
    pub fn default_model(self) -> &'static str {
        match self {
            ProviderName::Anthropic => airlok_llm::anthropic::DEFAULT_MODEL,
            ProviderName::OpenAi => airlok_llm::openai::DEFAULT_MODEL,
        }
    }

    /// Environment variables tried in order when `api_key_env` is unset.
    pub fn default_api_key_envs(self) -> &'static [&'static str] {
        match self {
            ProviderName::Anthropic => &["ANTHROPIC_API_KEY"],
            ProviderName::OpenAi => &["AZURE_OPENAI_API_KEY", "OPENAI_API_KEY"],
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ProviderName::Anthropic => "anthropic",
            ProviderName::OpenAi => "openai",
        }
    }
}

/// The on-disk schema. Every field is optional so layers can be merged.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConfigFile {
    pub provider: ProviderSection,
    pub agent: AgentSection,
    pub safety: SafetySection,
    pub context: ContextSection,
    pub redact: RedactSection,
    /// `[models."<id>"]`: settings for one model id, the deployment name on
    /// Azure. They apply whichever way the model was chosen, `/model` included.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub models: BTreeMap<String, ModelSection>,
    /// `[[mcp]]`: external MCP servers whose tools the model may call.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub mcp: Vec<McpSection>,
}

pub use airlok_llm::types::Api;

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelSection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api: Option<Api>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vision: Option<bool>,
}

/// One `[[mcp]]` entry on disk. `name` is required; everything else is
/// optional so layers can be merged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpSection {
    /// Unique; it prefixes every tool the server offers.
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transport: Option<McpTransport>,
    /// stdio: the program to run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub args: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env: Option<BTreeMap<String, String>>,
    /// Environment values read from a command's stdout, so a token never
    /// sits in the file. Same idea as `api_key_cmd`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env_cmd: Option<BTreeMap<String, String>>,
    /// http: the endpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<BTreeMap<String, String>>,
    /// Header values read from a command's stdout.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub header_cmd: Option<BTreeMap<String, String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_secs: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<McpTools>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trust: Option<McpTrust>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rehydrate: Option<bool>,
}

/// Where a server's definition came from. Later scopes win, and a
/// `[[mcp]]` block in a TOML config wins over all of them, since it is the
/// airlok-specific layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum McpScope {
    /// `~/.config/airlok/mcp.json`
    User,
    /// `./.mcp.json`, meant to be committed and shared
    Project,
    /// `./.airlok/mcp.json`, personal and gitignored
    Local,
    /// An `[[mcp]]` block in `config.toml` or `airlok.toml`
    Toml,
}

impl McpScope {
    pub fn as_str(self) -> &'static str {
        match self {
            McpScope::User => "user",
            McpScope::Project => "project",
            McpScope::Local => "local",
            McpScope::Toml => "toml",
        }
    }

    /// The scopes a `--scope` flag accepts, in precedence order.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "user" => Some(McpScope::User),
            "project" => Some(McpScope::Project),
            "local" => Some(McpScope::Local),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum McpTransport {
    /// A child process speaking JSON-RPC on its stdin and stdout.
    Stdio,
    /// Streamable HTTP.
    Http,
}

/// Whether calls to a server's tools are confirmed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum McpTrust {
    /// Ask before every call, like an unlisted shell command.
    Prompt,
    /// Never ask.
    Allow,
    /// Never call: the tools are listed but not offered to the model.
    Deny,
}

/// `tools = "all"`, or a list of the tool names to offer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpTools {
    All,
    Only(Vec<String>),
}

impl McpTools {
    pub fn allows(&self, tool: &str) -> bool {
        match self {
            McpTools::All => true,
            McpTools::Only(names) => names.iter().any(|name| name == tool),
        }
    }
}

impl Serialize for McpTools {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            McpTools::All => serializer.serialize_str("all"),
            McpTools::Only(names) => names.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for McpTools {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Word(String),
            List(Vec<String>),
        }
        match Raw::deserialize(deserializer)? {
            Raw::Word(word) if word == "all" => Ok(McpTools::All),
            Raw::Word(word) => Err(serde::de::Error::custom(format!(
                "tools must be \"all\" or a list of tool names, not {word:?}"
            ))),
            Raw::List(names) => Ok(McpTools::Only(names)),
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProviderSection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<ProviderName>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_key_cmd: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api: Option<Api>,
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AgentSection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bash_timeout_secs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compact_at: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_recent_turns: Option<usize>,
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SafetySection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confirm_writes: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confirm_bash: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confirm_mcp: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confirm_images: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bash_allowlist: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bash_denylist: Option<Vec<String>>,
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RedactSection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub show_secrets_in_output: Option<bool>,
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ContextSection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_bytes: Option<usize>,
}

impl ConfigFile {
    pub fn parse(text: &str, path: &Path) -> Result<Self, ConfigError> {
        toml::from_str(text).map_err(|source| ConfigError::Parse {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Reads a layer. A missing file is an empty layer, not an error.
    pub fn read(path: &Path) -> Result<Option<Self>, ConfigError> {
        match std::fs::read_to_string(path) {
            Ok(text) => Self::parse(&text, path).map(Some),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(source) => Err(ConfigError::Read {
                path: path.to_path_buf(),
                source,
            }),
        }
    }

    /// Fields set in `over` win; everything else comes from `self`.
    pub fn layer(self, over: ConfigFile) -> ConfigFile {
        ConfigFile {
            provider: ProviderSection {
                name: over.provider.name.or(self.provider.name),
                model: over.provider.model.or(self.provider.model),
                base_url: over.provider.base_url.or(self.provider.base_url),
                api_key_env: over.provider.api_key_env.or(self.provider.api_key_env),
                api_key_cmd: over.provider.api_key_cmd.or(self.provider.api_key_cmd),
                context_window: over
                    .provider
                    .context_window
                    .or(self.provider.context_window),
                api: over.provider.api.or(self.provider.api),
            },
            agent: AgentSection {
                max_turns: over.agent.max_turns.or(self.agent.max_turns),
                max_tokens: over.agent.max_tokens.or(self.agent.max_tokens),
                bash_timeout_secs: over
                    .agent
                    .bash_timeout_secs
                    .or(self.agent.bash_timeout_secs),
                compact_at: over.agent.compact_at.or(self.agent.compact_at),
                keep_recent_turns: over
                    .agent
                    .keep_recent_turns
                    .or(self.agent.keep_recent_turns),
            },
            safety: SafetySection {
                confirm_writes: over.safety.confirm_writes.or(self.safety.confirm_writes),
                confirm_bash: over.safety.confirm_bash.or(self.safety.confirm_bash),
                confirm_mcp: over.safety.confirm_mcp.or(self.safety.confirm_mcp),
                confirm_images: over.safety.confirm_images.or(self.safety.confirm_images),
                bash_allowlist: over.safety.bash_allowlist.or(self.safety.bash_allowlist),
                bash_denylist: over.safety.bash_denylist.or(self.safety.bash_denylist),
            },
            context: ContextSection {
                max_bytes: over.context.max_bytes.or(self.context.max_bytes),
            },
            redact: RedactSection {
                show_secrets_in_output: over
                    .redact
                    .show_secrets_in_output
                    .or(self.redact.show_secrets_in_output),
            },
            models: layer_models(self.models, over.models),
            mcp: layer_mcp(self.mcp, over.mcp),
        }
    }
}

/// Merges `[[mcp]]` by server name, then per key; `over` wins where both
/// set one. A name only `over` has is appended.
fn layer_mcp(mut base: Vec<McpSection>, over: Vec<McpSection>) -> Vec<McpSection> {
    for section in over {
        match base.iter().position(|s| s.name == section.name) {
            Some(at) => {
                let under = base[at].clone();
                base[at] = McpSection {
                    name: section.name,
                    transport: section.transport.or(under.transport),
                    command: section.command.or(under.command),
                    args: section.args.or(under.args),
                    env: section.env.or(under.env),
                    env_cmd: section.env_cmd.or(under.env_cmd),
                    url: section.url.or(under.url),
                    headers: section.headers.or(under.headers),
                    header_cmd: section.header_cmd.or(under.header_cmd),
                    enabled: section.enabled.or(under.enabled),
                    timeout_secs: section.timeout_secs.or(under.timeout_secs),
                    tools: section.tools.or(under.tools),
                    trust: section.trust.or(under.trust),
                    rehydrate: section.rehydrate.or(under.rehydrate),
                };
            }
            None => base.push(section),
        }
    }
    base
}

/// Merges `[models]` per model and per key; `over` wins where both set one.
fn layer_models(
    mut base: BTreeMap<String, ModelSection>,
    over: BTreeMap<String, ModelSection>,
) -> BTreeMap<String, ModelSection> {
    for (id, section) in over {
        let merged = base.remove(&id).unwrap_or_default();
        base.insert(
            id,
            ModelSection {
                reasoning_effort: section.reasoning_effort.or(merged.reasoning_effort),
                api: section.api.or(merged.api),
                vision: section.vision.or(merged.vision),
            },
        );
    }
    base
}

/// Values given on the command line. The highest layer.
#[derive(Debug, Default, Clone)]
pub struct Overrides {
    pub provider: Option<ProviderName>,
    pub model: Option<String>,
    /// `--yes`: turn every confirmation off.
    pub yes: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub provider: ProviderConfig,
    pub agent: AgentConfig,
    pub safety: SafetyConfig,
    pub context: ContextConfig,
    pub redact: RedactConfig,
    /// Per-model settings by model id.
    pub models: BTreeMap<String, ModelConfig>,
    /// External MCP servers, in the order they were configured.
    pub mcp: Vec<McpServer>,
    /// Directory the agent works in. Tools resolve relative paths against it.
    pub cwd: PathBuf,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelConfig {
    /// Sent as `reasoning_effort` by the openai provider. Not validated: the
    /// provider rejects values it does not accept.
    pub reasoning_effort: Option<String>,
    /// Overrides `[provider] api` for this model.
    pub api: Option<Api>,
    /// Whether this model takes images. A deployment name says nothing
    /// about it, so it cannot be inferred. Unset is treated as yes, and
    /// the provider's refusal names this setting.
    pub vision: Option<bool>,
}

/// A configured MCP server. Commands that produce secrets are kept as
/// commands and run when connecting, so `config show` never holds a value.
#[derive(Debug, Clone, PartialEq)]
pub struct McpServer {
    pub name: String,
    pub transport: McpTransport,
    pub command: Option<String>,
    pub args: Vec<String>,
    pub env: BTreeMap<String, String>,
    pub env_cmd: BTreeMap<String, String>,
    pub url: Option<String>,
    pub headers: BTreeMap<String, String>,
    pub header_cmd: BTreeMap<String, String>,
    pub enabled: bool,
    /// How long to wait for the server to start and answer.
    pub timeout: Duration,
    pub tools: McpTools,
    pub trust: McpTrust,
    /// Whether `Rehydrate` secrets are restored in arguments. Off by
    /// default, so a server is sent placeholders.
    pub rehydrate: bool,
    /// Which file it came from, for `airlok mcp list` and `mcp get`.
    pub scope: McpScope,
}

pub const DEFAULT_MCP_TIMEOUT_SECS: u64 = 30;

impl McpServer {
    fn from_section(section: McpSection) -> Self {
        let url = section.url;
        let transport = section.transport.unwrap_or(if url.is_some() {
            McpTransport::Http
        } else {
            McpTransport::Stdio
        });
        Self {
            name: section.name,
            transport,
            command: section.command,
            args: section.args.unwrap_or_default(),
            env: section.env.unwrap_or_default(),
            env_cmd: section.env_cmd.unwrap_or_default(),
            url,
            headers: section.headers.unwrap_or_default(),
            header_cmd: section.header_cmd.unwrap_or_default(),
            enabled: section.enabled.unwrap_or(true),
            timeout: Duration::from_secs(section.timeout_secs.unwrap_or(DEFAULT_MCP_TIMEOUT_SECS)),
            tools: section.tools.unwrap_or(McpTools::All),
            trust: section.trust.unwrap_or(McpTrust::Prompt),
            rehydrate: section.rehydrate.unwrap_or(false),
            scope: McpScope::Toml,
        }
    }

    fn to_section(&self) -> McpSection {
        McpSection {
            name: self.name.clone(),
            transport: Some(self.transport),
            command: self.command.clone(),
            args: Some(self.args.clone()),
            env: Some(self.env.clone()),
            env_cmd: Some(self.env_cmd.clone()),
            url: self.url.clone(),
            headers: Some(self.headers.clone()),
            header_cmd: Some(self.header_cmd.clone()),
            enabled: Some(self.enabled),
            timeout_secs: Some(self.timeout.as_secs()),
            tools: Some(self.tools.clone()),
            trust: Some(self.trust),
            rehydrate: Some(self.rehydrate),
        }
    }

    /// What the server can reach, when that can be told: the first
    /// argument naming a directory that exists, or the url for an http
    /// server. Shown in `airlok mcp list` and in the confirmation, so a
    /// path outside the project is visible before a call runs.
    pub fn root(&self) -> Option<String> {
        if self.transport == McpTransport::Http {
            return self.url.clone();
        }
        self.args
            .iter()
            .map(PathBuf::from)
            .find(|arg| arg.is_dir())
            .map(|dir| {
                dir.canonicalize()
                    .unwrap_or(dir)
                    .to_string_lossy()
                    .into_owned()
            })
    }

    /// The child process environment, running each `env_cmd` once. The
    /// values are secrets: never log them.
    pub fn resolved_env(&self) -> Result<BTreeMap<String, String>, ConfigError> {
        let mut env = self.env.clone();
        for (name, command) in &self.env_cmd {
            env.insert(name.clone(), self.run(command, "env_cmd", name)?);
        }
        Ok(env)
    }

    /// The HTTP headers, running each `header_cmd` once.
    pub fn resolved_headers(&self) -> Result<BTreeMap<String, String>, ConfigError> {
        let mut headers = self.headers.clone();
        for (name, command) in &self.header_cmd {
            headers.insert(name.clone(), self.run(command, "header_cmd", name)?);
        }
        Ok(headers)
    }

    fn run(&self, command: &str, what: &str, key: &str) -> Result<String, ConfigError> {
        command_stdout(command).map_err(|problem| ConfigError::McpCommand {
            server: self.name.clone(),
            what: format!("{what} for {key}"),
            command: command.to_string(),
            problem,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProviderConfig {
    pub name: ProviderName,
    pub model: String,
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
    pub api_key_cmd: Option<String>,
    /// Tokens the model can take as input. Not looked up per model; set it
    /// when the default is wrong for yours.
    pub context_window: u64,
    /// The HTTP API the openai provider speaks, unless a model overrides it.
    pub api: Api,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AgentConfig {
    /// Upper bound on model round-trips in one run, so a confused model
    /// cannot loop forever.
    pub max_turns: usize,
    pub max_tokens: u32,
    pub bash_timeout: Duration,
    /// Compact once the last request used this fraction of the window.
    pub compact_at: f64,
    /// Turns kept verbatim after the summary.
    pub keep_recent_turns: usize,
}

impl AgentConfig {
    /// Input tokens at which a session is compacted.
    pub fn compact_threshold(&self, context_window: u64) -> u64 {
        (context_window as f64 * self.compact_at.clamp(0.0, 1.0)) as u64
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SafetyConfig {
    pub confirm_writes: bool,
    pub confirm_bash: bool,
    /// Ask before a call to an MCP server whose `trust` is `prompt`.
    pub confirm_mcp: bool,
    /// Ask before sending an image, which cannot be scanned for secrets.
    pub confirm_images: bool,
    pub bash_allowlist: Vec<String>,
    pub bash_denylist: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RedactConfig {
    /// Show secrets found in files and tool output in full in the terminal.
    /// Off by default: they are masked to the first four characters and a
    /// length. Never applies to redact-only entries such as the provider key.
    pub show_secrets_in_output: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ContextConfig {
    /// Upper bound on the context block prepended to the system prompt.
    pub max_bytes: usize,
}

/// Anthropic and OpenAI frontier models both accept at least this much.
pub const DEFAULT_CONTEXT_WINDOW: u64 = 200_000;
pub const DEFAULT_CONTEXT_MAX_BYTES: usize = 32 * 1024;

pub const DEFAULT_BASH_ALLOWLIST: &[&str] = &[
    "git status",
    "git diff",
    "ls",
    "cat",
    "pwd",
    "find",
    "grep",
    "rg",
    "cargo check",
    "cargo test",
    "cargo build",
];

pub const DEFAULT_BASH_DENYLIST: &[&str] = &["rm -rf", "git push --force", "sudo"];

/// Where each file layer was looked for, for `config show`.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    pub path: PathBuf,
    pub found: bool,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Sources {
    pub user: Option<Layer>,
    pub project: Option<Layer>,
    /// One line per `mcp.json` problem: a file that will not parse, or a
    /// `${VAR}` with nothing to put in it. The server is left out and the
    /// line is shown, rather than airlok refusing to start.
    pub mcp_problems: Vec<String>,
}

/// Where the API key comes from. Displayable without revealing the key.
#[derive(Debug, Clone, PartialEq)]
pub enum KeySource {
    Env(String),
    Command(String),
}

impl std::fmt::Display for KeySource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KeySource::Env(name) => {
                let state = if env_is_set(name) { "set" } else { "unset" };
                write!(f, "env {name} ({state})")
            }
            KeySource::Command(command) => write!(f, "command \"{command}\""),
        }
    }
}

impl Config {
    /// Built-in defaults only.
    pub fn new(cwd: PathBuf) -> Self {
        Self::resolve(ConfigFile::default(), cwd)
    }

    /// Reads the user and project layers, applies overrides, and resolves.
    pub fn load(
        user: Option<&Path>,
        project: Option<&Path>,
        overrides: &Overrides,
        cwd: PathBuf,
    ) -> Result<(Config, Sources), ConfigError> {
        let mut merged = ConfigFile::default();
        let mut sources = Sources::default();
        for (path, slot) in [(user, &mut sources.user), (project, &mut sources.project)] {
            let Some(path) = path else { continue };
            let layer = ConfigFile::read(path)?;
            *slot = Some(Layer {
                path: path.to_path_buf(),
                found: layer.is_some(),
            });
            if let Some(layer) = layer {
                merged = merged.layer(layer);
            }
        }
        merged = merged.layer(overrides.as_layer());
        let mut config = Self::resolve(merged, cwd);
        // The JSON scopes, then the TOML blocks over them: `[[mcp]]` is
        // the airlok-specific layer, so it wins a name clash.
        let (servers, problems) = crate::mcp::json::load(user.and_then(Path::parent), &config.cwd);
        sources.mcp_problems = problems;
        config.mcp = crate::mcp::json::merge(servers, std::mem::take(&mut config.mcp));
        Ok((config, sources))
    }

    pub fn resolve(file: ConfigFile, cwd: PathBuf) -> Self {
        let name = file.provider.name.unwrap_or(ProviderName::Anthropic);
        let to_strings = |items: &[&str]| items.iter().map(|s| s.to_string()).collect();
        Config {
            provider: ProviderConfig {
                name,
                model: file
                    .provider
                    .model
                    .unwrap_or_else(|| name.default_model().to_string()),
                base_url: file.provider.base_url,
                api_key_env: file.provider.api_key_env,
                api_key_cmd: file.provider.api_key_cmd,
                context_window: file
                    .provider
                    .context_window
                    .unwrap_or(DEFAULT_CONTEXT_WINDOW),
                api: file.provider.api.unwrap_or_default(),
            },
            agent: AgentConfig {
                max_turns: file.agent.max_turns.unwrap_or(50),
                max_tokens: file.agent.max_tokens.unwrap_or(8192),
                bash_timeout: Duration::from_secs(file.agent.bash_timeout_secs.unwrap_or(120)),
                compact_at: file.agent.compact_at.unwrap_or(0.75),
                keep_recent_turns: file.agent.keep_recent_turns.unwrap_or(4),
            },
            safety: SafetyConfig {
                confirm_writes: file.safety.confirm_writes.unwrap_or(true),
                confirm_bash: file.safety.confirm_bash.unwrap_or(true),
                confirm_mcp: file.safety.confirm_mcp.unwrap_or(true),
                confirm_images: file.safety.confirm_images.unwrap_or(true),
                bash_allowlist: file
                    .safety
                    .bash_allowlist
                    .unwrap_or_else(|| to_strings(DEFAULT_BASH_ALLOWLIST)),
                bash_denylist: file
                    .safety
                    .bash_denylist
                    .unwrap_or_else(|| to_strings(DEFAULT_BASH_DENYLIST)),
            },
            context: ContextConfig {
                max_bytes: file.context.max_bytes.unwrap_or(DEFAULT_CONTEXT_MAX_BYTES),
            },
            redact: RedactConfig {
                show_secrets_in_output: file.redact.show_secrets_in_output.unwrap_or(false),
            },
            models: file
                .models
                .into_iter()
                .map(|(id, m)| {
                    (
                        id,
                        ModelConfig {
                            reasoning_effort: m.reasoning_effort,
                            api: m.api,
                            vision: m.vision,
                        },
                    )
                })
                .collect(),
            mcp: file.mcp.into_iter().map(McpServer::from_section).collect(),
            cwd,
        }
    }

    /// The effective configuration in file form, for `config show`.
    pub fn to_file(&self) -> ConfigFile {
        ConfigFile {
            provider: ProviderSection {
                name: Some(self.provider.name),
                model: Some(self.provider.model.clone()),
                base_url: self.provider.base_url.clone(),
                api_key_env: self.provider.api_key_env.clone(),
                api_key_cmd: self.provider.api_key_cmd.clone(),
                context_window: Some(self.provider.context_window),
                api: Some(self.provider.api),
            },
            agent: AgentSection {
                max_turns: Some(self.agent.max_turns),
                max_tokens: Some(self.agent.max_tokens),
                bash_timeout_secs: Some(self.agent.bash_timeout.as_secs()),
                compact_at: Some(self.agent.compact_at),
                keep_recent_turns: Some(self.agent.keep_recent_turns),
            },
            safety: SafetySection {
                confirm_writes: Some(self.safety.confirm_writes),
                confirm_bash: Some(self.safety.confirm_bash),
                confirm_mcp: Some(self.safety.confirm_mcp),
                confirm_images: Some(self.safety.confirm_images),
                bash_allowlist: Some(self.safety.bash_allowlist.clone()),
                bash_denylist: Some(self.safety.bash_denylist.clone()),
            },
            context: ContextSection {
                max_bytes: Some(self.context.max_bytes),
            },
            redact: RedactSection {
                show_secrets_in_output: Some(self.redact.show_secrets_in_output),
            },
            models: self
                .models
                .iter()
                .map(|(id, m)| {
                    (
                        id.clone(),
                        ModelSection {
                            reasoning_effort: m.reasoning_effort.clone(),
                            api: m.api,
                            vision: m.vision,
                        },
                    )
                })
                .collect(),
            mcp: self.mcp.iter().map(McpServer::to_section).collect(),
        }
    }

    /// Where the key will be read from. Does not read it.
    pub fn key_source(&self) -> KeySource {
        if let Some(command) = &self.provider.api_key_cmd {
            return KeySource::Command(command.clone());
        }
        if let Some(name) = &self.provider.api_key_env {
            return KeySource::Env(name.clone());
        }
        let candidates = self.provider.name.default_api_key_envs();
        let name = candidates
            .iter()
            .find(|name| env_is_set(name))
            .or(candidates.last())
            .expect("every provider has at least one default env var");
        KeySource::Env(name.to_string())
    }

    /// Reads the key. Runs `api_key_cmd` at most once per call; the caller
    /// must not log the result.
    pub fn resolve_key(&self) -> Result<String, ConfigError> {
        match self.key_source() {
            KeySource::Env(name) => std::env::var(&name)
                .ok()
                .map(|k| k.trim().to_string())
                .filter(|k| !k.is_empty())
                .ok_or(ConfigError::KeyEnvUnset(name)),
            KeySource::Command(command) => run_key_command(&command),
        }
    }
}

impl Overrides {
    fn as_layer(&self) -> ConfigFile {
        let mut layer = ConfigFile {
            provider: ProviderSection {
                name: self.provider,
                model: self.model.clone(),
                ..Default::default()
            },
            ..Default::default()
        };
        // A provider switch without a model must not inherit a model meant
        // for the other provider; the file's model is for the file's provider.
        if let (Some(provider), None) = (self.provider, &self.model) {
            layer.provider.model = Some(provider.default_model().to_string());
        }
        if self.yes {
            layer.safety.confirm_writes = Some(false);
            layer.safety.confirm_bash = Some(false);
            layer.safety.confirm_mcp = Some(false);
            layer.safety.confirm_images = Some(false);
        }
        layer
    }
}

fn env_is_set(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| !v.trim().is_empty())
}

fn run_key_command(command: &str) -> Result<String, ConfigError> {
    command_stdout(command).map_err(|problem| ConfigError::KeyCommand {
        command: command.to_string(),
        problem,
    })
}

/// Runs `command` and returns its trimmed stdout, or why that failed. The
/// output is a secret, so it never appears in the returned problem.
fn command_stdout(command: &str) -> Result<String, String> {
    let output = Command::new("sh")
        .arg("-c")
        .arg(command)
        .output()
        .map_err(|e| format!("could not start: {e}"))?;
    if !output.status.success() {
        return Err(format!("exited with {}", output.status));
    }
    let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if value.is_empty() {
        return Err("printed nothing on stdout".to_string());
    }
    Ok(value)
}

/// `$XDG_CONFIG_HOME/airlok/config.toml`, falling back to `~/.config`.
pub fn user_config_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join(USER_CONFIG_RELATIVE))
}

pub fn project_config_path(cwd: &Path) -> PathBuf {
    cwd.join(PROJECT_CONFIG_NAME)
}

/// Written by `airlok config init`. Every key is present and commented out
/// so the file documents the schema without pinning today's defaults.
pub const TEMPLATE: &str = r##"# airlok configuration. Precedence: CLI flags > ./airlok.toml > this file > defaults.
# Every key is optional. Uncomment a line to override its default.

[provider]
# name = "anthropic"            # "anthropic" or "openai"
# model = "claude-sonnet-4-6"   # default depends on the provider (openai: "gpt-5.5")
# base_url = "https://api.openai.com/v1"   # openai only; Azure: "https://<resource>.openai.azure.com/openai/v1"
# api_key_env = "ANTHROPIC_API_KEY"        # env var holding the key; openai default tries AZURE_OPENAI_API_KEY then OPENAI_API_KEY
# api_key_cmd = "az cognitiveservices account keys list -n <resource> -g <group> --query key1 -o tsv"   # shell command whose stdout is the key; run once per process
# context_window = 200000   # input tokens the model accepts; used to decide when to compact

[agent]
# max_turns = 50           # model round-trips per run
# max_tokens = 8192        # output tokens per model reply
# bash_timeout_secs = 120  # kill a bash tool command after this long
# compact_at = 0.75        # summarise the session once a request uses this fraction of context_window
# keep_recent_turns = 4    # turns kept verbatim after the summary

[safety]
# confirm_writes = true    # show a diff and ask before write_file / edit_file
# confirm_bash = true      # ask before running a command that is not allowlisted
# confirm_mcp = true      # ask before calling a tool on an MCP server whose trust is "prompt"
# confirm_images = true   # ask before sending an image, which cannot be scanned for secrets
# bash_allowlist = ["git status", "git diff", "ls", "cat", "pwd", "find", "grep", "rg", "cargo check", "cargo test", "cargo build"]
# bash_denylist = ["rm -rf", "git push --force", "sudo"]

[context]
# max_bytes = 32768         # cap on the context block (tree is cut first, then instructions)

[redact]
# show_secrets_in_output = false   # show secrets from files in full in the terminal instead of masked

# Settings for one model id (the deployment name on Azure), applied however it is chosen, /model included.
# [models."gpt-6-astra"]
# reasoning_effort = "none"   # openai only, sent as reasoning_effort; Azure's gpt-6-astra needs "none" to use tools on Chat Completions
# api = "responses"          # openai only: "chat" (the default) or "responses"
# vision = false             # set when the model refuses images, so the refusal comes at attach time

# An MCP server whose tools the model may call, offered as <name>__<tool>. Repeat the block for more.
# [[mcp]]
# name = "files"
# transport = "stdio"   # "stdio" runs command + args; "http" posts to url
# command = "npx"
# args = ["-y", "@modelcontextprotocol/server-filesystem", "."]
# env = { NODE_ENV = "production" }        # env_cmd = { TOKEN = "..." } takes the value from a command's stdout instead
# url = "https://example.com/mcp"          # http only; headers = { ... } and header_cmd = { ... } work the same way
# enabled = true
# timeout_secs = 30        # starting the server and every call
# tools = "all"            # or a list: ["read_text_file", "list_directory"]
# trust = "prompt"         # "prompt" asks before each call, "allow" never asks, "deny" keeps the tools from the model
# rehydrate = false        # false sends placeholders in arguments; true restores the secrets first
"##;

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, text: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, text).unwrap();
        path
    }

    fn scratch(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "airlok-config-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn defaults_match_the_spec() {
        let config = Config::new(PathBuf::from("."));
        assert_eq!(config.provider.name, ProviderName::Anthropic);
        assert_eq!(config.provider.model, "claude-sonnet-4-6");
        assert_eq!(config.agent.max_turns, 50);
        assert_eq!(config.agent.max_tokens, 8192);
        assert_eq!(config.agent.bash_timeout, Duration::from_secs(120));
        assert!(config.safety.confirm_writes);
        assert!(config.safety.confirm_bash);
        assert_eq!(config.safety.bash_allowlist.len(), 11);
        assert_eq!(
            config.safety.bash_denylist,
            ["rm -rf", "git push --force", "sudo"]
        );
    }

    #[test]
    fn three_layers_merge_with_flag_over_project_over_user() {
        let dir = scratch("merge");
        let user = write(
            &dir,
            "user.toml",
            r#"
            [provider]
            name = "openai"
            model = "from-user"
            base_url = "https://user.example/v1"
            [agent]
            max_turns = 5
            [safety]
            confirm_bash = false
            bash_denylist = ["from-user"]
            "#,
        );
        let project = write(
            &dir,
            "airlok.toml",
            r#"
            [provider]
            model = "from-project"
            [agent]
            max_tokens = 1234
            [safety]
            bash_denylist = ["from-project"]
            "#,
        );
        let overrides = Overrides {
            provider: None,
            model: Some("from-flag".into()),
            yes: false,
        };

        let (config, sources) =
            Config::load(Some(&user), Some(&project), &overrides, dir.clone()).unwrap();

        assert_eq!(config.provider.model, "from-flag");
        assert_eq!(config.provider.name, ProviderName::OpenAi);
        assert_eq!(
            config.provider.base_url.as_deref(),
            Some("https://user.example/v1")
        );
        assert_eq!(config.agent.max_turns, 5);
        assert_eq!(config.agent.max_tokens, 1234);
        assert_eq!(config.agent.bash_timeout, Duration::from_secs(120));
        assert!(config.safety.confirm_writes);
        assert!(!config.safety.confirm_bash);
        assert_eq!(config.safety.bash_denylist, ["from-project"]);
        assert_eq!(config.safety.bash_allowlist.len(), 11);
        assert!(sources.user.unwrap().found);
        assert!(sources.project.unwrap().found);

        // Without the flag, project beats user.
        let (config, _) = Config::load(
            Some(&user),
            Some(&project),
            &Overrides::default(),
            dir.clone(),
        )
        .unwrap();
        assert_eq!(config.provider.model, "from-project");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn missing_files_are_empty_layers() {
        let dir = scratch("missing");
        let (config, sources) = Config::load(
            Some(&dir.join("nope.toml")),
            Some(&dir.join("airlok.toml")),
            &Overrides::default(),
            dir.clone(),
        )
        .unwrap();
        assert_eq!(config, Config::new(dir.clone()));
        assert!(!sources.user.unwrap().found);
        assert!(!sources.project.unwrap().found);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn provider_flag_without_model_uses_that_providers_default() {
        let dir = scratch("provider-flag");
        let user = write(
            &dir,
            "user.toml",
            "[provider]\nname = \"openai\"\nmodel = \"my-azure-deployment\"\n",
        );
        let overrides = Overrides {
            provider: Some(ProviderName::Anthropic),
            model: None,
            yes: false,
        };
        let (config, _) = Config::load(Some(&user), None, &overrides, dir.clone()).unwrap();
        assert_eq!(config.provider.name, ProviderName::Anthropic);
        assert_eq!(config.provider.model, "claude-sonnet-4-6");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn yes_flag_turns_confirmations_off() {
        let overrides = Overrides {
            yes: true,
            ..Default::default()
        };
        let (config, _) = Config::load(None, None, &overrides, PathBuf::from(".")).unwrap();
        assert!(!config.safety.confirm_writes);
        assert!(!config.safety.confirm_bash);
        assert!(!config.safety.confirm_mcp);
    }

    #[test]
    fn unknown_keys_are_rejected() {
        let err = ConfigFile::parse("[agent]\nmax_turn = 3\n", Path::new("x.toml")).unwrap_err();
        assert!(matches!(err, ConfigError::Parse { .. }), "{err}");
    }

    #[test]
    fn template_uncommented_equals_the_defaults() {
        let uncommented: String = TEMPLATE
            .lines()
            .map(|line| {
                line.strip_prefix("# ")
                    .filter(|l| l.contains('=') || l.starts_with('['))
                    .unwrap_or(line)
            })
            .map(|line| format!("{line}\n"))
            .collect();
        let file = ConfigFile::parse(&uncommented, Path::new("template")).unwrap();
        let mut resolved = Config::resolve(file, PathBuf::from("."));
        let expected = Config::new(PathBuf::from("."));
        // The template's [models] entry is an example; there are none by default.
        assert_eq!(
            resolved.models["gpt-6-astra"].reasoning_effort.as_deref(),
            Some("none")
        );
        resolved.models.clear();
        // So is its [[mcp]] entry, and it shows every key at its default.
        let server = resolved.mcp.remove(0);
        assert_eq!(server.name, "files");
        assert_eq!(server.transport, McpTransport::Stdio);
        assert_eq!(server.command.as_deref(), Some("npx"));
        assert!(server.enabled);
        assert_eq!(
            server.timeout,
            Duration::from_secs(DEFAULT_MCP_TIMEOUT_SECS)
        );
        assert_eq!(server.tools, McpTools::All);
        assert_eq!(server.trust, McpTrust::Prompt);
        assert!(!server.rehydrate);
        // The template shows openai-only examples for these; the defaults leave them unset.
        assert_eq!(
            resolved.provider.base_url.as_deref(),
            Some("https://api.openai.com/v1")
        );
        assert_eq!(
            resolved.provider.api_key_env.as_deref(),
            Some("ANTHROPIC_API_KEY")
        );
        assert!(resolved
            .provider
            .api_key_cmd
            .as_deref()
            .unwrap()
            .starts_with("az "));
        resolved.provider.base_url = None;
        resolved.provider.api_key_env = None;
        resolved.provider.api_key_cmd = None;
        assert_eq!(resolved, expected);
    }

    #[test]
    fn key_source_prefers_command_then_env_then_provider_default() {
        let mut config = Config::new(PathBuf::from("."));
        config.provider.name = ProviderName::OpenAi;
        assert!(matches!(config.key_source(), KeySource::Env(_)));
        config.provider.api_key_env = Some("MY_KEY".into());
        assert_eq!(config.key_source(), KeySource::Env("MY_KEY".into()));
        config.provider.api_key_cmd = Some("printf k".into());
        assert_eq!(config.key_source(), KeySource::Command("printf k".into()));
        assert_eq!(config.resolve_key().unwrap(), "k");
    }

    #[test]
    fn key_command_failures_never_include_output() {
        let mut config = Config::new(PathBuf::from("."));
        // The output must not appear in the error, so compute it rather than
        // spelling it out in the command text.
        config.provider.api_key_cmd = Some("echo $((1000 + 337)); exit 3".into());
        let err = config.resolve_key().unwrap_err().to_string();
        assert!(err.contains("exited with"), "{err}");
        assert!(!err.contains("1337"), "{err}");
        config.provider.api_key_cmd = Some("true".into());
        let err = config.resolve_key().unwrap_err().to_string();
        assert!(err.contains("printed nothing"), "{err}");
    }

    #[test]
    fn mcp_servers_merge_by_name_and_new_names_are_appended() {
        let user = ConfigFile::parse(
            r#"
            [[mcp]]
            name = "files"
            command = "npx"
            args = ["-y", "server-filesystem", "."]
            trust = "prompt"
            [[mcp]]
            name = "docs"
            url = "https://example.com/mcp"
            "#,
            Path::new("user"),
        )
        .unwrap();
        let project = ConfigFile::parse(
            r#"
            [[mcp]]
            name = "files"
            trust = "allow"
            tools = ["read_text_file"]
            [[mcp]]
            name = "tickets"
            command = "./tickets-mcp"
            "#,
            Path::new("project"),
        )
        .unwrap();

        let config = Config::resolve(user.layer(project), PathBuf::from("."));

        let names: Vec<&str> = config.mcp.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["files", "docs", "tickets"]);
        let files = &config.mcp[0];
        // The project layer wins per key; the rest comes from the user file.
        assert_eq!(files.trust, McpTrust::Allow);
        assert_eq!(files.tools, McpTools::Only(vec!["read_text_file".into()]));
        assert_eq!(files.command.as_deref(), Some("npx"));
        assert_eq!(files.args.len(), 3);
        // transport follows url when it is not given.
        assert_eq!(config.mcp[1].transport, McpTransport::Http);
        assert_eq!(config.mcp[2].transport, McpTransport::Stdio);
        // Defaults for everything unset.
        assert!(files.enabled);
        assert!(!files.rehydrate);
    }

    #[test]
    fn mcp_tools_is_all_or_a_list_and_nothing_else() {
        let parse = |text: &str| ConfigFile::parse(text, Path::new("x"));
        let all = parse("[[mcp]]\nname = \"a\"\ntools = \"all\"\n").unwrap();
        assert_eq!(all.mcp[0].tools, Some(McpTools::All));
        let list = parse("[[mcp]]\nname = \"a\"\ntools = [\"one\", \"two\"]\n").unwrap();
        assert_eq!(
            list.mcp[0].tools,
            Some(McpTools::Only(vec!["one".into(), "two".into()]))
        );
        let err = parse("[[mcp]]\nname = \"a\"\ntools = \"some\"\n").unwrap_err();
        assert!(err.to_string().contains("tools must be"), "{err}");
        // A name is required, and unknown keys are still rejected.
        assert!(parse("[[mcp]]\ncommand = \"x\"\n").is_err());
        assert!(parse("[[mcp]]\nname = \"a\"\nnope = 1\n").is_err());
        assert!(McpTools::All.allows("anything"));
        assert!(!McpTools::Only(vec!["a".into()]).allows("b"));
    }

    #[test]
    fn mcp_secrets_come_from_commands_and_failures_never_carry_the_output() {
        let file = ConfigFile::parse(
            r#"
            [[mcp]]
            name = "docs"
            url = "https://example.com/mcp"
            headers = { Accept = "application/json" }
            header_cmd = { Authorization = "printf 'Bearer %s' $((1000 + 337))" }
            env_cmd = { TOKEN = "echo $((1000 + 337)); exit 3" }
            "#,
            Path::new("x"),
        )
        .unwrap();
        let config = Config::resolve(file, PathBuf::from("."));
        let server = &config.mcp[0];

        let headers = server.resolved_headers().unwrap();
        assert_eq!(headers["Accept"], "application/json");
        assert_eq!(headers["Authorization"], "Bearer 1337");

        // The config keeps the command, never the value it produces.
        let shown = toml::to_string(&config.to_file()).unwrap();
        assert!(shown.contains("printf"), "{shown}");
        assert!(!shown.contains("1337"), "{shown}");

        let err = server.resolved_env().unwrap_err().to_string();
        assert!(err.contains("env_cmd for TOKEN"), "{err}");
        assert!(!err.contains("1337"), "{err}");
    }

    #[test]
    fn models_layer_per_model_and_per_key() {
        let user = ConfigFile::parse(
            "[models.\"a\"]\nreasoning_effort = \"low\"\n[models.\"b\"]\nreasoning_effort = \"high\"\n",
            Path::new("user"),
        )
        .unwrap();
        let project = ConfigFile::parse(
            "[models.\"a\"]\nreasoning_effort = \"none\"\n",
            Path::new("project"),
        )
        .unwrap();
        let config = Config::resolve(user.layer(project), PathBuf::from("."));
        assert_eq!(config.models["a"].reasoning_effort.as_deref(), Some("none"));
        assert_eq!(config.models["b"].reasoning_effort.as_deref(), Some("high"));
        let shown = toml::to_string(&config.to_file()).unwrap();
        assert!(
            shown.contains("[models.a]") || shown.contains("[models.\"a\"]"),
            "{shown}"
        );
        assert!(ConfigFile::parse("[models.\"a\"]\nnope = 1\n", Path::new("x")).is_err());
    }
}
