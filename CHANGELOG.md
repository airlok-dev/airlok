# Changelog

All notable changes to airlok. The format follows Keep a Changelog; versions follow SemVer.

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
