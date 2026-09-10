# airlok

airlok is a terminal coding agent with one differentiator: it is a privacy airlock between your code and a third-party model you do not control. Secrets, keys, and anything else you tell it to protect are replaced with placeholders before a request leaves your machine, and put back when the model's answer comes home. The model never sees the real values, and your workflow never notices the swap. More at [airlok.dev](https://airlok.dev).

## Install

```sh
cargo install airlok
# or
npm install -g airlok
# or
brew install airlok-dev/tap/airlok
```

## Usage

```sh
airlok "create hello.txt containing hello"     # run one task in the current directory
airlok -v "..."                                # debug logs on stderr
airlok -y "..."                                # no confirmations (prints a warning)
airlok --provider openai --model gpt-5.5 "..." # override the provider and model for one run
airlok --show-redactions "..."                 # list what was redacted (kind and length only)
airlok config init                             # write a commented config to the user path
airlok config show                             # print the effective config and the key source
```

Every `write_file` and `edit_file` call shows a unified diff and asks `Apply? [y/N/a]`. Every `bash` call that is not on the allow list shows the command and asks `Run? [y/N/a]`. `a` approves the rest of that kind for the run. A rejection is sent back to the model so it can adapt. Prompts are read from the terminal, not stdin, so piped input still works; without a terminal, pass `--yes` or turn the confirmations off in the config.

## Configuration

Precedence, highest first: CLI flags, `./airlok.toml` in the working directory, the user file, built-in defaults. The user file is `$XDG_CONFIG_HOME/airlok/config.toml`, which is `~/.config/airlok/config.toml` on macOS and Linux. `airlok config init` writes it with every key commented out.

Full schema with defaults:

```toml
[provider]
name = "anthropic"            # "anthropic" or "openai"
model = "claude-sonnet-4-6"   # per provider: anthropic "claude-sonnet-4-6", openai "gpt-5.5"
# base_url = "https://api.openai.com/v1"   # openai only; Azure: "https://<resource>.openai.azure.com/openai/v1"
# api_key_env = "ANTHROPIC_API_KEY"        # env var holding the key
# api_key_cmd = "..."                      # shell command whose stdout is the key

[agent]
max_turns = 50           # model round-trips per run
max_tokens = 8192        # output tokens per model reply
bash_timeout_secs = 120  # kill a bash tool command after this long

[safety]
confirm_writes = true    # show a diff and ask before write_file / edit_file
confirm_bash = true      # ask before running a command that is not allow-listed
bash_allowlist = ["git status", "git diff", "ls", "cat", "pwd", "find", "grep", "rg", "cargo check", "cargo test", "cargo build"]
bash_denylist = ["rm -rf", "git push --force", "sudo"]
```

For openai, `base_url` falls back to the `OPENAI_BASE_URL` environment variable when the config does not set it.

The key is read from exactly one place. `api_key_cmd` wins if set, then `api_key_env`, then the provider default: `ANTHROPIC_API_KEY` for anthropic, and `AZURE_OPENAI_API_KEY` then `OPENAI_API_KEY` for openai. A command runs once per process and its output is never logged. `airlok config show` prints the env var name or the command, never the value. Whatever key is in use is also added to the redactor, so it can never leave the machine inside a file or command output either.

Allow-list entries match on leading tokens (`git status` allows `git status --short`, not `git push`). Deny-list entries match anywhere, in every part of a chained command, and win over the allow list. Flags are parsed rather than compared as text: the `rm -rf` entry also catches `rm -fr`, `rm -r -f`, `rm -Rf`, and `rm --recursive --force`, and `git push --force` also catches `git push -f`. A command with shell operators (`;`, `&&`, `|`, `>`, `$(`, and so on) always asks, whatever its first word.

### Azure OpenAI

```toml
[provider]
name = "openai"
model = "<deployment-name>"
base_url = "https://<resource>.openai.azure.com/openai/v1"
api_key_cmd = "az cognitiveservices account keys list -n <resource> -g <resource-group> --query key1 -o tsv"
```

With this in the user config, plain `airlok "..."` works. The key is sent in Azure's `api-key` header, chosen from the host. No shell wrapper or exported variable is needed.

## Licence

MIT OR Apache-2.0, at your option.
