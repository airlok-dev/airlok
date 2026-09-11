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
airlok                                         # interactive session in the current directory
airlok "create hello.txt containing hello"     # run one task in the current directory
airlok --resume                                # continue the latest saved session here (REPL, or with a task)
airlok --resume=<id> "summarise what we did"   # continue a particular one
airlok sessions                                # list saved sessions for this directory
airlok sessions rm <id>                        # delete one
airlok sessions clean --older-than 30d         # delete old sessions from every directory
airlok -v "..."                                # debug logs on stderr
airlok -y "..."                                # no confirmations (prints a warning)
airlok --provider openai --model gpt-5.5 "..." # override the provider and model for one run
airlok --show-redactions "..."                 # list what was redacted (kind and length only)
airlok config init                             # write a commented config to the user path
airlok config show                             # print the effective config and the key source
airlok context                                 # print the context block sent with the system prompt, after redaction
airlok redactions                              # list what the redactor detects and how each kind is treated
```

Every `write_file` and `edit_file` call shows a unified diff and asks `Apply? [y]es / [n]o / [a]ll / [q]uit`. Every `bash` call that is not on the allow list shows the command and asks the same way. `y` applies this one, `n` sends a rejection back to the model so it can adapt, `a` approves the rest of that kind for the run, and `q` aborts the run with a non-zero exit. Prompts are read from the terminal, not stdin, so piped input still works; without a terminal, pass `--yes` or turn the confirmations off in the config.

Model output is rendered as markdown (headings, emphasis, lists, tables, highlighted code) when stdout is a terminal; set `NO_COLOR` or redirect stdout for plain text. Runs of read-only tool calls collapse to one `reading N files...` line; `-v` shows every call.

### Sessions

`airlok` with no task opens a session. The prompt shows the model and the directory (`gpt-5.5 airlok> `), each line you type is one turn, and the conversation carries across turns. Ctrl-C during a turn cancels it: the text streamed so far stays in the history marked as interrupted, tool calls that had not run are dropped, and you get the prompt back. Ctrl-D or `/exit` saves and quits.

| Command | What it does |
|---|---|
| `/help` | list the commands |
| `/clear` | start a new session; the current one stays saved |
| `/compact` | summarise older turns to free context |
| `/cost` | tokens used so far, and whether they are estimates |
| `/redactions` | what was redacted before leaving this machine (kind and length only) |
| `/config` | the effective configuration |
| `/exit` | save and quit |

Every turn, one-shot or interactive, is saved under `$XDG_DATA_HOME/airlok/sessions/<hash of the directory>/` (`~/.local/share/airlok` by default). `airlok --resume` picks up the latest session for the directory; it rebuilds the context block, so the model sees the current tree and git state, and adds a note with the resume time. `airlok sessions` lists what is there.

A session file holds the plaintext conversation and the real value behind every placeholder, because that is what rehydration on resume needs. That is why the file is written with mode 0600 in a 0700 directory, and why the provider API key is the one thing left out: it is stored blank and re-read from your config on every start.

Long sessions are compacted. Once the last request used more than `compact_at` (default 0.75) of `context_window` (default 200k tokens; set it for your model), the next turn first asks the model for a summary of everything except the last `keep_recent_turns` turns and replaces those older turns with it. The summary request goes through the redactor like every other request. You see a dim `compacted: 180k -> 22k tokens` line; `/compact` does it on demand. Token counts come from the provider; when a provider reports none, airlok estimates at chars/4 and `/cost` says so.

## Tools

| Tool | Asks? | What it does |
|---|---|---|
| `read_file(path, offset?, limit?)` | no | Read a text file, optionally a window of lines. Binary files are refused. |
| `glob(pattern, path?)` | no | List files matching a glob, gitignore-aware, at most 500. |
| `grep(pattern, path?, include?)` | no | Regex search returning `file:line:text`, gitignore-aware, at most 200 lines. |
| `list_dir(path)` | no | Entries with type and size. |
| `edit_file(path, old, new)` | diff | Replace `old` with `new`; `old` must occur exactly once, otherwise the model is told how many matches there were. |
| `write_file(path, content)` | diff | Create or overwrite a file. |
| `bash(command)` | unless allow-listed | Run a shell command in the working directory. |

Tool results over 50 KiB are cut with a marker telling the model to page with `read_file`'s `offset` and `limit`.

## Context and AIRLOK.md

Every run starts with a context block in the system prompt: the working directory, OS and shell, the git branch with `git status --short` and the last five commit subjects, a gitignore-aware file tree (four levels, at most 200 entries), and project instructions. Instructions come from `AIRLOK.md` in the repository root, or `CLAUDE.md` or `AGENTS.md` if there is no `AIRLOK.md`, followed by `~/.config/airlok/AIRLOK.md`. The block goes through the redactor like everything else, and `airlok context` prints exactly what would be sent.

A short `AIRLOK.md`:

```markdown
# airlok instructions

- Run `cargo test --workspace` before saying a change is done.
- Never touch files under `vendor/`.
- Commit messages: imperative subject, no trailing period.
```

`[context] max_bytes` (default 32768) caps the block; the tree is cut to top-level directories first, then the instructions are truncated.

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
context_window = 200000       # input tokens the model accepts; used to decide when to compact

[agent]
max_turns = 50           # model round-trips per run
max_tokens = 8192        # output tokens per model reply
bash_timeout_secs = 120  # kill a bash tool command after this long
compact_at = 0.75        # summarise the session once a request uses this fraction of context_window
keep_recent_turns = 4    # turns kept verbatim after the summary

[safety]
confirm_writes = true    # show a diff and ask before write_file / edit_file
confirm_bash = true      # ask before running a command that is not allow-listed
bash_allowlist = ["git status", "git diff", "ls", "cat", "pwd", "find", "grep", "rg", "cargo check", "cargo test", "cargo build"]
bash_denylist = ["rm -rf", "git push --force", "sudo"]

[context]
max_bytes = 32768        # cap on the context block; tree is cut first, then instructions

[redact]
show_secrets_in_output = false   # show secrets from files in full in the terminal instead of masked
```

For openai, `base_url` falls back to the `OPENAI_BASE_URL` environment variable when the config does not set it.

### What comes back, and what never does

Every placeholder belongs to one of two classes, listed by `airlok redactions`. Secrets found in your files and in tool output are `rehydrate`: the real value is restored into files and commands, so edits keep working, and shown masked in the terminal (first four characters and the length) unless `[redact] show_secrets_in_output = true`. The provider API key is `redact-only`: it is never restored anywhere. If the model echoes its placeholder it prints as `[redacted: the provider API key]`, and a tool call carrying it is refused with a message to the model. `--show-redactions` lists each placeholder's kind, length, and class, never the value.

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
