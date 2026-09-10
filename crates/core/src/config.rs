//! Layered configuration.
//!
//! Precedence, highest first: CLI flags, `./airlok.toml` in the working
//! directory, the user file (`$XDG_CONFIG_HOME/airlok/config.toml`, which
//! is `~/.config/airlok/config.toml` on macOS and Linux), built-in defaults.
//!
//! [`ConfigFile`] is the on-disk shape where everything is optional;
//! [`Config`] is the resolved shape the agent runs with.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use serde::{Deserialize, Serialize};

pub const USER_CONFIG_RELATIVE: &str = "airlok/config.toml";
pub const PROJECT_CONFIG_NAME: &str = "airlok.toml";

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
}

#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SafetySection {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confirm_writes: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confirm_bash: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bash_allowlist: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bash_denylist: Option<Vec<String>>,
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
            },
            agent: AgentSection {
                max_turns: over.agent.max_turns.or(self.agent.max_turns),
                max_tokens: over.agent.max_tokens.or(self.agent.max_tokens),
                bash_timeout_secs: over
                    .agent
                    .bash_timeout_secs
                    .or(self.agent.bash_timeout_secs),
            },
            safety: SafetySection {
                confirm_writes: over.safety.confirm_writes.or(self.safety.confirm_writes),
                confirm_bash: over.safety.confirm_bash.or(self.safety.confirm_bash),
                bash_allowlist: over.safety.bash_allowlist.or(self.safety.bash_allowlist),
                bash_denylist: over.safety.bash_denylist.or(self.safety.bash_denylist),
            },
        }
    }
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
    /// Directory the agent works in. Tools resolve relative paths against it.
    pub cwd: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProviderConfig {
    pub name: ProviderName,
    pub model: String,
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
    pub api_key_cmd: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AgentConfig {
    /// Upper bound on model round-trips in one run, so a confused model
    /// cannot loop forever.
    pub max_turns: usize,
    pub max_tokens: u32,
    pub bash_timeout: Duration,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SafetyConfig {
    pub confirm_writes: bool,
    pub confirm_bash: bool,
    pub bash_allowlist: Vec<String>,
    pub bash_denylist: Vec<String>,
}

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
        Ok((Self::resolve(merged, cwd), sources))
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
            },
            agent: AgentConfig {
                max_turns: file.agent.max_turns.unwrap_or(50),
                max_tokens: file.agent.max_tokens.unwrap_or(8192),
                bash_timeout: Duration::from_secs(file.agent.bash_timeout_secs.unwrap_or(120)),
            },
            safety: SafetyConfig {
                confirm_writes: file.safety.confirm_writes.unwrap_or(true),
                confirm_bash: file.safety.confirm_bash.unwrap_or(true),
                bash_allowlist: file
                    .safety
                    .bash_allowlist
                    .unwrap_or_else(|| to_strings(DEFAULT_BASH_ALLOWLIST)),
                bash_denylist: file
                    .safety
                    .bash_denylist
                    .unwrap_or_else(|| to_strings(DEFAULT_BASH_DENYLIST)),
            },
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
            },
            agent: AgentSection {
                max_turns: Some(self.agent.max_turns),
                max_tokens: Some(self.agent.max_tokens),
                bash_timeout_secs: Some(self.agent.bash_timeout.as_secs()),
            },
            safety: SafetySection {
                confirm_writes: Some(self.safety.confirm_writes),
                confirm_bash: Some(self.safety.confirm_bash),
                bash_allowlist: Some(self.safety.bash_allowlist.clone()),
                bash_denylist: Some(self.safety.bash_denylist.clone()),
            },
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
        }
        layer
    }
}

fn env_is_set(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| !v.trim().is_empty())
}

fn run_key_command(command: &str) -> Result<String, ConfigError> {
    let fail = |problem: String| ConfigError::KeyCommand {
        command: command.to_string(),
        problem,
    };
    let output = Command::new("sh")
        .arg("-c")
        .arg(command)
        .output()
        .map_err(|e| fail(format!("could not start: {e}")))?;
    if !output.status.success() {
        return Err(fail(format!("exited with {}", output.status)));
    }
    let key = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if key.is_empty() {
        return Err(fail("printed nothing on stdout".to_string()));
    }
    Ok(key)
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

[agent]
# max_turns = 50           # model round-trips per run
# max_tokens = 8192        # output tokens per model reply
# bash_timeout_secs = 120  # kill a bash tool command after this long

[safety]
# confirm_writes = true    # show a diff and ask before write_file / edit_file
# confirm_bash = true      # ask before running a command that is not allowlisted
# bash_allowlist = ["git status", "git diff", "ls", "cat", "pwd", "find", "grep", "rg", "cargo check", "cargo test", "cargo build"]
# bash_denylist = ["rm -rf", "git push --force", "sudo"]
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
                    .filter(|l| l.contains('='))
                    .unwrap_or(line)
            })
            .map(|line| format!("{line}\n"))
            .collect();
        let file = ConfigFile::parse(&uncommented, Path::new("template")).unwrap();
        let mut resolved = Config::resolve(file, PathBuf::from("."));
        let expected = Config::new(PathBuf::from("."));
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
}
