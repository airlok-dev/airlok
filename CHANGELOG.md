# Changelog

All notable changes to airlok. The format follows Keep a Changelog; versions follow SemVer.

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
