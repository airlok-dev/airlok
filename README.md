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

Environment variables:

| Variable | Used when | Meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | `--provider anthropic` (default) | Anthropic API key |
| `OPENAI_API_KEY` | `--provider openai` | OpenAI API key, sent as a bearer token |
| `AZURE_OPENAI_API_KEY` | `--provider openai` | Azure OpenAI key, sent as an `api-key` header. Takes precedence over `OPENAI_API_KEY` |
| `OPENAI_BASE_URL` | `--provider openai` | Base URL, default `https://api.openai.com/v1`. For Azure: `https://<resource>.openai.azure.com/openai/v1`, and pass the deployment name as `--model` |

## Usage

```sh
airlok "create hello.txt containing hello"     # run one task in the current directory
airlok -v "..."                                # debug logs on stderr
airlok --model claude-opus-5 "..."             # override the model (default claude-sonnet-4-6)
airlok --provider openai "..."                 # use OpenAI instead (default model gpt-5.5)
airlok --show-redactions "..."                 # list what was redacted after the run
```

Version 0.1 ships two providers (Anthropic, OpenAI), three tools (`read_file`, `write_file`, `bash`), and redaction of common API key and token formats. See the `TODO(stage N)` comments for what comes next.

## Licence

MIT OR Apache-2.0, at your option.
