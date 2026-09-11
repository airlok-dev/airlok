use std::time::Duration;

use airlok_core::agent::INTERRUPTED_MARKER;
use airlok_core::repl::{Line, Repl};
use airlok_core::{Interrupt, SessionStore};
use airlok_llm::ContentBlock;
use airlok_tests::{
    agent, partial, reply, MockProvider, RecordingOutput, ScriptedLines, Shown, TempDir,
    TestBackend,
};

const TOKEN_A: &str = concat!("ghp_", "abcdefghijklmnopqrstuvwxyz0123456789");
const TOKEN_B: &str = concat!("ghp_", "zyxwvutsrqponmlkjihgfedcba9876543210");

fn text_of(block: &ContentBlock) -> &str {
    match block {
        ContentBlock::Text { text } => text,
        other => panic!("expected text, got {other:?}"),
    }
}

#[tokio::test]
async fn a_scripted_session_runs_turns_commands_clear_and_exit() {
    let dir = TempDir::new("repl");
    let store = SessionStore::new(dir.path().join("data"));
    let provider = MockProvider::scripted(vec![reply("hi"), reply("hi again"), reply("never")]);
    let mut agent = agent(provider.clone(), dir.path());
    let mut repl = Repl {
        agent: &mut agent,
        store: Some(&store),
        interrupt: Interrupt::new(),
        backend: Box::new(TestBackend::default()),
    };
    let first = repl.agent.new_session();
    let mut lines = ScriptedLines::typed(&[
        &format!("hello {TOKEN_A}"),
        "/cost",
        "/compact",
        "/bogus",
        "/clear",
        &format!("again {TOKEN_B}"),
        "/help",
        "/redactions",
        "/exit",
        "this is never read",
    ]);
    let mut out = RecordingOutput::default();

    let last = repl.run(first.clone(), &mut lines, &mut out).await;

    let requests = provider.requests();
    assert_eq!(requests.len(), 2, "/exit stops before the trailing line");
    assert_eq!(
        text_of(&requests[0].messages[0].content[0]),
        "hello <<SECRET_1>>"
    );
    // /clear started a fresh session with a fresh redactor: history and
    // placeholder numbering both start over.
    assert_eq!(requests[1].messages.len(), 1);
    assert_eq!(
        text_of(&requests[1].messages[0].content[0]),
        "again <<SECRET_1>>"
    );
    assert_ne!(last.id, first.id);
    assert_eq!(
        last.first_prompt(),
        Some(format!("again {TOKEN_B}").as_str())
    );

    let statuses = out.statuses();
    assert!(
        statuses.iter().any(|s| s.starts_with("input ")),
        "{statuses:?}"
    );
    assert!(statuses.iter().any(|s| s.contains("estimated at chars/4")));
    assert!(statuses.iter().any(|s| s.starts_with("nothing to compact")));
    assert!(statuses
        .iter()
        .any(|s| s.starts_with("unknown command /bogus")));
    assert!(statuses.iter().any(|s| s.starts_with("new session ")));
    assert!(statuses.iter().any(|s| s.starts_with("/exit")));
    assert!(statuses.iter().any(|s| s.contains("github token")));
    assert!(!statuses
        .iter()
        .any(|s| s.contains(TOKEN_A) || s.contains(TOKEN_B)));
    assert_eq!(out.text(), "hihi again");
    assert_eq!(
        out.events.iter().filter(|e| **e == Shown::EndTurn).count(),
        2
    );
    assert!(lines.prompts[0].ends_with(&format!(
        " {}> ",
        dir.path().file_name().unwrap().to_string_lossy()
    )));

    // Both sessions were saved: the cleared one and the one at /exit.
    let saved = store.list(dir.path()).unwrap();
    assert_eq!(saved.len(), 2);
    assert_eq!(saved[0].id, last.id);
    assert_eq!(
        saved.iter().map(|s| s.turns).collect::<Vec<_>>(),
        vec![1, 1]
    );
}

#[tokio::test]
async fn ctrl_c_keeps_the_partial_reply_marked_interrupted() {
    let dir = TempDir::new("repl-interrupt");
    let store = SessionStore::new(dir.path().join("data"));
    let provider = MockProvider::scripted(vec![partial("The answer is"), reply("resumed fine")]);
    let mut agent = agent(provider.clone(), dir.path());
    let interrupt = Interrupt::new();
    let trigger = interrupt.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        trigger.trigger();
    });
    let mut repl = Repl {
        agent: &mut agent,
        store: Some(&store),
        interrupt,
        backend: Box::new(TestBackend::default()),
    };
    let session = repl.agent.new_session();
    let mut lines = ScriptedLines::new(vec![
        Line::Text("go".into()),
        Line::Interrupt,
        Line::Text("carry on".into()),
        Line::Eof,
    ]);
    let mut out = RecordingOutput::default();

    let session = repl.run(session, &mut lines, &mut out).await;

    assert_eq!(session.messages.len(), 4);
    assert_eq!(
        text_of(&session.messages[1].content[0]),
        format!("The answer is\n{INTERRUPTED_MARKER}")
    );
    assert!(!session.interrupted, "the following turn completed");
    assert!(out.text().starts_with("The answer is"));
    let statuses = out.statuses();
    assert!(
        statuses.contains(&"interrupted".to_string()),
        "{statuses:?}"
    );
    assert!(statuses.contains(&"(use /exit or Ctrl-D to quit)".to_string()));
    // The next request carried the marked partial reply.
    let second = &provider.requests()[1];
    assert_eq!(second.messages.len(), 3);
    assert!(text_of(&second.messages[1].content[0]).ends_with(INTERRUPTED_MARKER));
    // Ctrl-D saved it.
    let saved = store.load(dir.path(), None).unwrap().unwrap();
    assert_eq!(saved.messages, session.messages);
}

#[tokio::test]
async fn model_and_provider_switch_for_the_rest_of_the_session() {
    let dir = TempDir::new("repl-switch");
    let store = SessionStore::new(dir.path().join("data"));
    let first = MockProvider::scripted(vec![reply("one"), reply("two")]);
    let second = MockProvider::scripted(vec![reply("three")]);
    let mut agent = agent(first.clone(), dir.path());
    let mut backend = TestBackend::default();
    backend.switches.insert("openai", Ok(second.clone()));
    let mut repl = Repl {
        agent: &mut agent,
        store: Some(&store),
        interrupt: Interrupt::new(),
        backend: Box::new(backend),
    };
    let session = repl.agent.new_session();
    let mut lines = ScriptedLines::typed(&[
        &format!("remember {TOKEN_A}"),
        "/model",
        "/model my-new-model",
        "second",
        "/provider anthropic",
        "/provider nonsense",
        "/provider openai",
        "third",
        "/exit",
    ]);
    let mut out = RecordingOutput::default();

    let session = repl.run(session, &mut lines, &mut out).await;

    let default_model = airlok_llm::anthropic::DEFAULT_MODEL;
    let before = first.requests();
    assert_eq!(before.len(), 2);
    assert_eq!(before[0].model, default_model);
    assert_eq!(
        before[1].model, "my-new-model",
        "/model applies to the next turn"
    );
    let after = second.requests();
    assert_eq!(
        after.len(),
        1,
        "the turn after /provider went to the new provider"
    );
    assert_eq!(after[0].model, airlok_llm::openai::DEFAULT_MODEL);
    let carried = format!("{:?}", after[0].messages[0]);
    assert!(
        carried.contains("<<SECRET_1>>"),
        "same placeholder: {carried}"
    );
    assert!(
        !carried.contains(TOKEN_A),
        "the secret stays redacted after the switch"
    );

    let statuses = out.statuses();
    assert!(
        statuses.contains(&format!("model {default_model} (anthropic)")),
        "{statuses:?}"
    );
    assert!(statuses.contains(&"model my-new-model for the rest of this session".to_string()));
    assert!(statuses.contains(&"cannot switch to anthropic: no key configured".to_string()));
    assert!(statuses.contains(&"unknown provider nonsense: anthropic or openai".to_string()));
    let dir_name = dir
        .path()
        .file_name()
        .unwrap()
        .to_string_lossy()
        .to_string();
    assert!(lines
        .prompts
        .contains(&format!("my-new-model {dir_name}> ")));
    assert!(lines.prompts.contains(&format!(
        "{} {dir_name}> ",
        airlok_llm::openai::DEFAULT_MODEL
    )));

    assert_eq!(session.provider, "openai");
    assert_eq!(session.model, airlok_llm::openai::DEFAULT_MODEL);
    let saved = store.load(dir.path(), None).unwrap().unwrap();
    assert_eq!(saved.provider, "openai");
    assert_eq!(saved.model, airlok_llm::openai::DEFAULT_MODEL);
    assert_eq!(
        saved.config.provider.model.as_deref(),
        Some(airlok_llm::openai::DEFAULT_MODEL)
    );
}

#[tokio::test]
async fn a_model_switch_is_saved_before_the_next_turn() {
    let dir = TempDir::new("repl-model-save");
    let store = SessionStore::new(dir.path().join("data"));
    let provider = MockProvider::scripted(vec![reply("one")]);
    let mut agent = agent(provider, dir.path());
    let mut repl = Repl {
        agent: &mut agent,
        store: Some(&store),
        interrupt: Interrupt::new(),
        backend: Box::new(TestBackend::default()),
    };
    let session = repl.agent.new_session();
    // Ctrl-C at the prompt ends nothing; the switch must already be on disk.
    let mut lines = ScriptedLines::new(vec![
        Line::Text("hello".into()),
        Line::Text("/model other-model".into()),
    ]);
    let mut out = RecordingOutput::default();
    let session = repl.run(session, &mut lines, &mut out).await;
    let saved = store.load(dir.path(), Some(&session.id)).unwrap().unwrap();
    assert_eq!(saved.model, "other-model");
}
