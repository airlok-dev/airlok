# Changelog

All notable changes to airlok. The format follows Keep a Changelog; versions follow SemVer.

## [0.9.0] - 2026-09-13

### Added

- `/status`: version, provider, model and effort, session id and turn count, cwd and git branch, the configuration files in effect, the MCP servers with whether each is connected, and redaction counts by class. Names, counts and sources only; no value from the key or the redaction map can reach it.
- `/context`: where the context window is going, as a bar per part for the system prompt, context block, history, tool results and tool schemas, with how near compaction is. The parts are counted the way the request estimator counts them and the total is their sum, so the breakdown always adds up.
- `/doctor`, and `airlok doctor` as a subcommand that exits non-zero on failure so it works in CI. Seven checks: the config files, whether the key resolves, a minimal live request, every MCP server, git, the terminal, and write access to the session and trust directories.
- `/diff`: what airlok changed on disk this session, diffed from what was there before its first write, using the same renderer and paging as a write confirmation. `/diff <path>` narrows to one file. Sessions now record the files `write_file` and `edit_file` touched, with the original content kept under a one megabyte budget; past it a file is reported as changed rather than shown with a misleading diff.
- `/copy`: the last reply to the clipboard through pbcopy, wl-copy, xclip or xsel. `/copy <n>` takes the Nth from the end and `/copy code` the last fenced block. With no clipboard program it says which ones it tried.
- `/permissions`: view and change what airlok asks about for the session, including both bash lists, and `/permissions save` to write them into the project config after showing a diff. Removing a deny entry says what it did rather than quietly emptying the list.
- `/init`: propose an AIRLOK.md built from the repository, as a diff to approve. Build and test commands come from the marker files that are actually present, and an existing CLAUDE.md or AGENTS.md is carried over.
- `/btw <question>`: a side question answered with the project and conversation as context but no tools. Neither the question nor the answer joins the history; one note is kept, as an assistant message so it is not counted as a turn.
- `/goal <statement>`: what the session is working toward. It goes into the system prompt every turn with an instruction to say when it is met, shows in the footer, and survives `--resume`. `/goal` shows it and `/goal clear` removes it.
- Tab completes the arguments of the new commands, as it does for `/model` and `/provider`.

### Fixed

- The renderer wraps to the terminal's current width rather than the width it had when the run started. It re-measures each turn and on SIGWINCH, so a window resized mid-session takes effect on the next block instead of leaving the terminal to wrap over-long lines mid-line.
- The status line is sized to the terminal the same way, instead of keeping the width it was built with.

## [0.8.2] - 2026-09-12

### Added

- `/effort` shows the reasoning effort in force for the model in use and where it came from: the per-model config, this session, or the provider's own default. `/effort <value>` sets it for the rest of the session, using the same picker and the same did-you-mean question as `/model`, offering what the provider accepts (`none`, `minimal`, `low`, `medium`, `high`). The value is recorded in the session, so `--resume` keeps it, and on resume the session's value wins over the config file for the model the session last used. Only the openai provider sends a reasoning effort, so on anthropic the command says so rather than offering a list.
- Tab completes `/effort` arguments, alongside `/model` and `/provider`.

### Changed

- `/e` now resolves to `/effort` rather than `/exit`. Commands are matched by prefix in the order they are listed, and `/effort` is listed beside `/model`. `/exit`, `/quit` and Ctrl-D are unchanged.

### Fixed

- Ctrl-C leaves the `/model` and `/provider` picker, the way Esc does. The picker's terminal mode kept signals on, so Ctrl-C raised SIGINT and the list stayed up until Esc. Signals come back the moment the picker closes, so Ctrl-C at the prompt still clears the line.

## [0.8.1] - 2026-09-12

### Added

- `/model` with no id opens a picker: the models airlok knows here, with arrows, type to filter, Enter to take one and Esc to leave everything alone. The list is the model in use, any named in `[models."<id>"]`, the ones used earlier in the run, and the ones saved sessions used on this provider. `/provider` with no name picks the same way, and `/model <id>` still switches directly.
- Tab completes a slash command's argument, not only its name. Commands say what their arguments could be, and `/model` and `/provider` are the first two.

### Changed

- Switching to a model id airlok has not seen says so, names the nearest ids it knows, and asks whether to use the typed one anyway. An exact match is still taken as given, and the typed id is always offered first, so pressing Enter never quietly selects a near match. Before this, an unknown id was accepted and the provider rejected it a turn later.
- A provider that refuses a request is reported as one sentence naming the likely cause and what to change, for a missing model or deployment, a rejected key, rate limiting, and a request longer than the model's context. The raw body moved to the debug log, where `-v` shows it, and the session stays open as before.

## [0.8.0] - 2026-09-12

### Changed

- A server from the project's own `.mcp.json` no longer starts until this checkout has been asked about it. On first sight of one, or of a definition that changed since the last answer, airlok lists the servers and the commands they would run and asks once; the answer is recorded per repository in `.airlok/`, which is gitignored. Until then the server is pending and nothing is started. This is a behaviour change: a project file that used to start servers on the first turn now waits for an answer. Servers from the user file, the local file, or a `[[mcp]]` block are unaffected.
- `airlok mcp list` shows a pending server and what it would run, and listing never approves anything. `airlok mcp call` refuses a server this repository has not approved.

### Added

- `airlok mcp reset-project-choices` forgets the answer, so the next run asks again.
- `airlok mcp add-json <name> '<json>'` writes a server from an entry pasted as JSON, which is how one is usually shared.

## [0.7.0] - 2026-09-12

### Changed

- Tools from an MCP server are now named `mcp__<server>__<tool>`, the convention the other clients use, in place of `<server>__<tool>`. This is a breaking change for anything that referred to the old names: a saved session's history, a note in AIRLOK.md, or a `tools = [...]` list naming a tool keeps working, since that list names the server's own tool rather than the prefixed one.

### Added

- MCP servers are configured with the `mcpServers` JSON that Claude Code, Cursor, and VS Code share, so a `.mcp.json` copied from another project works unchanged. Both forms are read: `command`/`args`/`env` for stdio, and `type`/`url`/`headers` for http. Unknown keys other tools write are ignored rather than refused.
- Three scopes, each winning over the one above it: `~/.config/airlok/mcp.json`, `./.mcp.json` meant to be committed, and `./.airlok/mcp.json` for personal overrides. A server named in more than one takes the highest definition whole. `[[mcp]]` TOML blocks keep working, merge with the JSON, and win a name clash.
- Airlok's own options live under an `airlok` key inside a JSON entry, so a plain config stays plain.
- `${VAR}` and `${VAR:-default}` expand from the environment in `command`, `args`, `env`, `url`, and `headers`. Unset or empty with no default is an error naming the variable. `env_cmd` and `header_cmd` remain the better way to hold a secret.
- `airlok mcp add`, `remove`, `get`, `import`, and `export` manage those files, and `airlok mcp list` now shows which scope each server came from.
- An approval can outlive the run: `s` at an MCP confirmation remembers that server, that tool, and the places that call named, in `.airlok/mcp-trust.json`, written 0600 and gitignored. It never covers a place outside the ones approved, and it records what the server was, so changing its command or url asks again. `airlok mcp trust list` and `trust revoke <server>` manage it.

## [0.6.1] - 2026-09-12

### Fixed

- Approving an MCP call with `a` covered the whole server for the rest of the run, so a later call to the same server ran without asking, even when it reached a different place. "All" now covers one tool on one server, and only the places that call named: a call naming anything outside them asks again. Paths are resolved before they are compared, so a different spelling of the same place is still covered and `..` cannot step outside an approval.
- The confirmation now shows what the server can reach and the resolved place each argument names, marking one outside the current project, and `airlok mcp list` shows the same root. A call that reads outside the repository is visible before it runs rather than after.
- A tool call or a note printed in the middle of a streamed line split the line in two: a bullet's bold label was rendered as a finished bullet and its text as a separate block, which is what `• Overview:` and its text landing on different lines was. Only complete markdown is flushed now, so a line still arriving keeps its block.

## [0.6.0] - 2026-09-12

### Added

- MCP servers. `[[mcp]]` blocks configure servers over stdio or streamable HTTP, and their tools are offered to the model as `<server>__<tool>`, alongside the built-ins. A server starts on the first turn that can use it; one that fails to start is reported and skipped, and the run continues without it. Tools can be limited with `tools = ["name"]`, and a built-in keeps its name if a server ever claims one. The client is the official `rmcp` SDK.
- `trust` per server decides the gate: `prompt`, the default, shows the server, the tool, and the arguments and asks with the usual `[y]es / [n]o / [a]ll / [q]uit`, where `a` covers that one server for the run; `allow` never asks; `deny` keeps the server's tools from the model. `[safety] confirm_mcp` turns the prompting off wholesale, and `--yes` includes it.
- `rehydrate` per server decides what an MCP server receives. It is off by default, so secrets found in your files leave as placeholders rather than values, and the confirmation shows exactly what will be sent. The provider API key is refused in arguments either way, as it is for every tool.
- Tool descriptions and results from a server reach the model inside markers naming the server and saying the text is data, so a server cannot instruct the model or change airlok's own confirmations and deny list.
- Secrets for a server come from commands: `env_cmd` and `header_cmd` take their values from stdout, the way `api_key_cmd` does, so no token is written in the config.
- `airlok mcp list` shows each server, whether it answers, and its tools. `airlok mcp call <server> <tool> '<json>'` calls one tool through the same gates. `/mcp` does the listing in a session, and `/mcp <name>` enables a disabled server for that session.
- Plan mode offers no MCP tools, as it offers no write tools, and starts no servers.

## [0.5.0] - 2026-09-12

### Added

- A status line during each turn: a spinner, what airlok is doing (`thinking`, `reading src/main.rs`, `running cargo test`), the seconds so far, and the tokens used this turn, redrawn in place below the output and cleared before anything else prints. It is off when stdout is not a terminal, with `NO_COLOR`, and with `-v`.
- Esc cancels a running turn as Ctrl-C does: the partial reply is kept and marked interrupted. The terminal is in cbreak mode only while a turn runs, and is restored when the turn ends, before each confirmation, on a panic, and on SIGTERM. Keys typed during a turn start the next prompt.
- Plan mode, with `/plan` or `--plan`. The model gets only `read_file`, `glob`, `grep`, and `list_dir`, the system prompt says the rest are unavailable, and it ends its reply with a plan. `/go` runs the plan as the task in normal mode; `/plan` again leaves without running it. Per-model settings are sent unchanged in both modes.
- `!command` runs a command in the working directory and adds the command and its output to the conversation. `#note` appends a list item to `./AIRLOK.md`, creating it, and rebuilds the context block.
- `@` completes paths from the working directory, fuzzy and gitignore-aware; Tab inserts the path as plain text. Typing `/` shows the matching commands with their descriptions. A prefix runs the first match, and an unknown command suggests the closest.
- Alt+Enter inserts a newline, and so does Shift+Enter in terminals that send Esc then Enter for it.
- Write confirmations show line numbers, syntax highlighting, and +/- gutters, and page at 40 lines; `v` shows the rest.
- A dim footer after each turn: model, context %, session id.
- `[models."<id>"]` config sections with `reasoning_effort`, sent by the openai provider only for that model id. Azure's `gpt-6-astra` needs `reasoning_effort = "none"` to use tools on Chat Completions; when a provider rejects its reasoning effort, airlok names the setting to add. `/model` shows the configured effort.

### Changed

- The REPL starts with one line: version, model, directory, notes such as a resumed session or confirmations being off, and `/help for commands`.
- Ctrl-C at an empty prompt does nothing; it used to print a hint.
- Runs of read-only tool calls show on the status line and end with one `read N files` line, in place of the updating `reading N files...` line.
- Slash commands are listed harmless and frequent first, since a prefix now runs the first match.

## [0.4.1] - 2026-09-11

### Added

- `/model [<id>]` shows the model or switches it for the rest of the session, without validating the id. `/provider [anthropic|openai]` switches provider when a key is available for it and otherwise names the missing key. Both are recorded in the session file, and `--resume` continues on the session's last model when it used the configured provider and no `--model` is given.

### Fixed

- Markdown lines slightly wider than the terminal no longer leave a bullet, a bold label, or a styled first word such as `airlok` alone on a line. termimad's wrap cut between style runs; airlok now wraps word by word. Paragraphs and list items are rendered per block (when a blank line or the next block marker arrives, or the reply ends), so a paragraph split across source lines reflows as one. A paragraph now appears when it completes, not line by line.
- The REPL prompt always starts on its own line: rustyline checks the cursor column before each prompt.

## [0.4.0] - 2026-09-11

### Added

- Interactive sessions: `airlok` with no task opens a REPL with a `<model> <dir>> ` prompt, `/help`, `/clear`, `/compact`, `/cost`, `/redactions`, `/config`, and `/exit`. Ctrl-C cancels the turn in progress and keeps the partial reply marked as interrupted; Ctrl-D saves and quits.
- Sessions are saved after every turn under `$XDG_DATA_HOME/airlok/sessions/<dir hash>/` (`~/.local/share/airlok` by default), mode 0600 in a 0700 directory, with the config snapshot, provider and model, plaintext history, redaction map, token counts, and timestamps. The provider API key is stored blank. `--resume` continues the latest session for the directory (`--resume=<id>` a particular one), rebuilding the context block and adding a system note with the resume time. `airlok sessions` lists them, `sessions rm <id>` deletes one, `sessions clean --older-than 30d` prunes every directory.
- Compaction: once the last request used more than `[agent] compact_at` (0.75) of `[provider] context_window` (200k), the next turn replaces everything but the last `[agent] keep_recent_turns` (4) turns with a model-written summary, requested through the redactor with no tools. Shown as a dim `compacted: X -> Y tokens` line and recorded in the session.
- Token usage from both providers: Anthropic `message_start`/`message_delta` counts, OpenAI `stream_options.include_usage`. Where a provider reports none, requests are estimated at chars/4 and `/cost` says so.

### Changed

- A turn that fails or is aborted no longer leaves its prompt in the history.

## [0.3.1] - 2026-09-11

### Security

- The provider API key could be shown in the terminal: it was an ordinary redaction entry, so when the model echoed its placeholder the display path rehydrated it. Redaction entries now have a class. `rehydrate` entries (secrets found in files and tool output) are restored into files and commands and shown masked in the terminal, with `[redact] show_secrets_in_output = true` to opt into full display. `redact-only` entries, the provider API key, are never restored anywhere: the terminal shows `[redacted: the provider API key]` and a tool call carrying the placeholder is refused. `airlok redactions` lists kinds and classes; `--show-redactions` now shows the class.

## [0.3.0] - 2026-09-11

### Added

- Context injection (#5). The system prompt carries a block with the working directory, OS, shell, git branch, status and last five commits, a gitignore-aware file tree, and project instructions from `AIRLOK.md` (or `CLAUDE.md` / `AGENTS.md`) plus `~/.config/airlok/AIRLOK.md`. `[context] max_bytes` caps it; `airlok context` prints it after redaction.
- Tools `glob`, `grep`, and `list_dir`, read-only and never prompting; `read_file` paging with `offset` and `limit` and binary refusal; tool results over 50 KiB are cut with a paging hint (#5).
- Markdown rendering in the terminal with highlighted code, buffered per block; plain output when stdout is not a terminal or `NO_COLOR` is set; dimmed, collapsing tool-call lines (#5).
- Confirmation prompt `[y]es / [n]o / [a]ll / [q]uit` with a one-time hint; `q` aborts the run with a non-zero exit (#5).

### Changed

- `edit_file` takes `old` and `new` and reports the match count when it cannot apply (#5).
- The system prompt tells the model to search with grep and glob before reading files and to prefer `edit_file` over `write_file` for existing files (#5).

## [0.2.2] - 2026-09-11

### Fixed

- The v0.2.1 release workflow was rejected by GitHub at startup because the npm publish job requested more permissions than its caller grants. No 0.2.1 release or npm package was produced. No change to the binary.

## [0.2.1] - 2026-09-11

### Changed

- npm publishing uses trusted publishing (OIDC) through a custom cargo-dist publish job instead of an `NPM_TOKEN` secret. No change to the binary.

## [0.2.0] - 2026-09-11

### Added

- Layered configuration (#3). `~/.config/airlok/config.toml` (`$XDG_CONFIG_HOME` honoured), overridden by `./airlok.toml`, overridden by flags. Sections `[provider]`, `[agent]`, `[safety]`, every key optional. `airlok config init` writes a commented template; `airlok config show` prints the merged config with each layer's path and the key source, never the key. The key can come from a named env var (`api_key_env`) or a shell command run once per process (`api_key_cmd`).
- Confirmation gates (#3). `write_file` and the new `edit_file` (exact search, must match once) show a unified diff and ask `Apply? [y/N/a]`. `bash` is checked against a deny list, then an allow list on leading tokens, and asks `Run? [y/N/a]` otherwise; chained commands never bypass the prompt. Deny entries are matched on parsed flags, so `rm -rf` also catches `rm -fr`, `rm -r -f`, `rm -Rf`, and `rm --recursive --force`. A rejection is returned to the model as an error result. `--yes` or `confirm_* = false` turns prompting off with a warning; prompts read `/dev/tty`, and without a terminal the run fails fast.
- OpenAI provider (#1). Chat Completions with streaming tool calls behind `--provider openai`. The SSE reader is shared between providers.
- Azure OpenAI (#2). `OPENAI_BASE_URL` and `AZURE_OPENAI_API_KEY` (sent as an `api-key` header, chosen from the host), tolerant of Azure's empty-choices `prompt_filter_results` chunk and per-choice `content_filter_results`. A test proves no secret reaches logs at TRACE level.

### Changed

- The provider key in use joins the redactor's known secrets, so it cannot leave the machine inside a file or command output (#3).
- `--show-redactions` lists each entry as a kind and a length only, never any part of the value (#3).

## [0.1.0] - 2026-09-10

- Initial scaffold: Cargo workspace with `airlok` (cli), `airlok-core`, and `airlok-llm`; Anthropic Messages API provider with streaming; `read_file`, `write_file`, and `bash` tools; regex redaction of common API key and token formats with stable placeholders, applied to every outbound request and reversed on every response; mock-provider tests including one proving a fake API key never reaches the provider; cargo-dist and CI configuration.
