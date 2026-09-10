# evals

Not built yet. This directory will hold the evaluation suite for airlok.

Two questions it must answer, with numbers:

1. Does the airlock leak? Run the agent over corpora seeded with known secrets, PII, and internal hostnames, capture every byte sent to the provider, and count what got through. The target is zero, and any regression fails CI.
2. Does redaction hurt the agent? Run the same coding tasks with redaction on and off against a fixed model, and compare task completion and turn count. Placeholders must not make the model measurably worse at the job.

Until this exists, the redaction guarantee rests on the unit tests in `crates/core/src/redact` and the integration tests in `tests/redaction.rs`.
