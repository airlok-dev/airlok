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

Set `ANTHROPIC_API_KEY` in your environment.

## Usage

```sh
airlok "create hello.txt containing hello"     # run one task in the current directory
airlok -v "..."                                # debug logs on stderr
airlok --model claude-opus-5 "..."             # override the model (default claude-sonnet-4-6)
airlok --show-redactions "..."                 # list what was redacted after the run
```

Version 0.1 ships one provider (Anthropic), three tools (`read_file`, `write_file`, `bash`), and redaction of common API key and token formats. See the `TODO(stage N)` comments for what comes next.

## Licence

MIT OR Apache-2.0, at your option.
