//! Server-sent events plumbing shared by providers: splits the byte stream
//! into `data:` payloads and hands them to a provider-specific assembler.

use std::collections::VecDeque;

use futures::stream::{self, BoxStream, StreamExt};

use crate::{LlmError, StreamEvent};

pub(crate) type ByteStream = BoxStream<'static, reqwest::Result<Vec<u8>>>;

/// Turns one `data:` payload into zero or more events.
pub(crate) trait Assembler: Send {
    fn feed(&mut self, payload: &str) -> Result<Vec<StreamEvent>, LlmError>;
    /// True once [`StreamEvent::MessageEnd`] has been emitted.
    fn ended(&self) -> bool;
}

/// Fails the request on a non-2xx status, otherwise returns the body stream.
pub(crate) async fn body_or_error(response: reqwest::Response) -> Result<ByteStream, LlmError> {
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(LlmError::Api {
            status: status.as_u16(),
            body,
        });
    }
    Ok(response
        .bytes_stream()
        .map(|chunk| chunk.map(|b| b.to_vec()))
        .boxed())
}

struct State<A> {
    bytes: ByteStream,
    buffer: String,
    pending: VecDeque<StreamEvent>,
    assembler: A,
    finished: bool,
}

pub(crate) fn events<A: Assembler + 'static>(
    bytes: ByteStream,
    assembler: A,
) -> impl futures::Stream<Item = Result<StreamEvent, LlmError>> + Send {
    let state = State {
        bytes,
        buffer: String::new(),
        pending: VecDeque::new(),
        assembler,
        finished: false,
    };
    stream::unfold(state, |mut st| async move {
        loop {
            if let Some(event) = st.pending.pop_front() {
                return Some((Ok(event), st));
            }
            if st.finished {
                return None;
            }
            match st.bytes.next().await {
                None => {
                    st.finished = true;
                    if !st.assembler.ended() {
                        return Some((
                            Err(LlmError::Protocol(
                                "stream ended before the message did".into(),
                            )),
                            st,
                        ));
                    }
                }
                Some(Err(e)) => {
                    st.finished = true;
                    return Some((Err(e.into()), st));
                }
                Some(Ok(chunk)) => {
                    st.buffer.push_str(&String::from_utf8_lossy(&chunk));
                    for payload in drain_payloads(&mut st.buffer) {
                        match st.assembler.feed(&payload) {
                            Ok(events) => st.pending.extend(events),
                            Err(e) => {
                                st.finished = true;
                                return Some((Err(e), st));
                            }
                        }
                    }
                    if st.assembler.ended() {
                        st.finished = true;
                    }
                }
            }
        }
    })
}

/// Removes every complete SSE event from `buffer` and returns their joined
/// `data:` payloads. Events are separated by a blank line.
pub(crate) fn drain_payloads(buffer: &mut String) -> Vec<String> {
    let mut payloads = Vec::new();
    while let Some((end, separator)) = find_event_boundary(buffer) {
        let event: String = buffer.drain(..end).collect();
        buffer.drain(..separator);
        let data: Vec<&str> = event
            .lines()
            .filter_map(|line| line.strip_prefix("data:"))
            .map(str::trim_start)
            .collect();
        if !data.is_empty() {
            payloads.push(data.join("\n"));
        }
    }
    payloads
}

/// Returns (index where the event text ends, length of the separator).
fn find_event_boundary(buffer: &str) -> Option<(usize, usize)> {
    let lf = buffer.find("\n\n").map(|i| (i, 2));
    let crlf = buffer.find("\r\n\r\n").map(|i| (i, 4));
    match (lf, crlf) {
        (Some(a), Some(b)) => Some(if a.0 <= b.0 { a } else { b }),
        (a, b) => a.or(b),
    }
}

/// Feeds raw SSE text through an assembler in small chunks. Test helper.
#[cfg(test)]
pub(crate) fn assemble_chunked<A: Assembler>(raw: &str, assembler: &mut A) -> Vec<StreamEvent> {
    let mut buffer = String::new();
    let mut events = Vec::new();
    for chunk in raw.as_bytes().chunks(17) {
        buffer.push_str(std::str::from_utf8(chunk).unwrap());
        for payload in drain_payloads(&mut buffer) {
            events.extend(assembler.feed(&payload).unwrap());
        }
    }
    assert!(buffer.is_empty(), "unconsumed SSE text: {buffer:?}");
    events
}
