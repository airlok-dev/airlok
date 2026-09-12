use std::time::Duration;

use airlok_core::agent::Agent;
use airlok_core::agent::INTERRUPTED_MARKER;
use airlok_core::config::{ModelConfig, ProviderName};
use airlok_core::redact::{RedactionMap, Redactor, SecretRedactor};
use airlok_core::repl::{Backend, Line, Repl, Switch};
use airlok_core::tools::READ_ONLY_TOOLS;
use airlok_core::{Interrupt, SessionStore};
use airlok_llm::{ContentBlock, Request};
use airlok_tests::{
    agent, partial, reply, tool_call, with_usage, MockProvider, RecordingOutput, ScriptedLines,
    Shown, TempDir, TestBackend,
};
use serde_json::json;

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
        used: Vec::new(),
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
        used: Vec::new(),
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
    assert!(
        !statuses.iter().any(|s| s.contains("quit")),
        "Ctrl-C at the prompt prints nothing: {statuses:?}"
    );
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
    // The bare /model opens the picker, which this cancels. my-new-model is
    // an id airlok has not seen here, so switching to it is questioned, and
    // the second answer takes the typed id.
    let mut backend = TestBackend {
        choices: vec![None, Some("my-new-model".into())].into(),
        ..TestBackend::default()
    };
    backend.switches.insert("openai", Ok(second.clone()));
    let mut repl = Repl {
        agent: &mut agent,
        store: Some(&store),
        interrupt: Interrupt::new(),
        backend: Box::new(backend),
        used: Vec::new(),
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
        // other-model is not an id airlok has seen, so it asks before
        // switching; this answers that question with the typed id.
        backend: Box::new(TestBackend {
            choices: vec![Some("other-model".into())].into(),
            ..TestBackend::default()
        }),
        used: Vec::new(),
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

fn user_texts(request: &Request) -> Vec<&str> {
    request
        .messages
        .iter()
        .filter(|m| m.role == airlok_llm::Role::User)
        .filter_map(|m| match m.content.first() {
            Some(ContentBlock::Text { text }) => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

fn tool_names(request: &Request) -> Vec<String> {
    let mut names: Vec<String> = request.tools.iter().map(|t| t.name.clone()).collect();
    names.sort();
    names
}

#[tokio::test]
async fn slash_prefixes_run_the_first_match_and_typos_get_a_suggestion() {
    let dir = TempDir::new("repl-slash");
    let provider = MockProvider::scripted(vec![]);
    let mut agent = agent(provider.clone(), dir.path());
    let mut repl = Repl {
        agent: &mut agent,
        store: None,
        interrupt: Interrupt::new(),
        backend: Box::new(TestBackend::default()),
        used: Vec::new(),
    };
    let session = repl.agent.new_session();
    let mut lines = ScriptedLines::typed(&["/co", "/modle x", "/", "/quit", "never read"]);
    let mut out = RecordingOutput::default();

    repl.run(session, &mut lines, &mut out).await;

    let statuses = out.statuses();
    assert!(
        statuses.iter().any(|s| s.starts_with("input ")),
        "/co ran /cost: {statuses:?}"
    );
    assert!(statuses
        .contains(&"unknown command /modle; did you mean /model? /help lists them".to_string()));
    assert!(
        statuses.iter().any(|s| s.starts_with("/plan ")),
        "/ alone ran /help"
    );
    assert!(provider.requests().is_empty());
    assert_eq!(lines.prompts.len(), 4, "/quit ended the session");
}

#[tokio::test]
async fn a_bang_command_runs_here_and_the_next_turn_sees_it() {
    let dir = TempDir::new("repl-bang");
    let store = SessionStore::new(dir.path().join("data"));
    let provider = MockProvider::scripted(vec![reply("It printed hello.")]);
    let mut agent = agent(provider.clone(), dir.path());
    let mut repl = Repl {
        agent: &mut agent,
        store: Some(&store),
        interrupt: Interrupt::new(),
        backend: Box::new(TestBackend::default()),
        used: Vec::new(),
    };
    let session = repl.agent.new_session();
    let mut lines = ScriptedLines::typed(&[
        "! echo hello-from-shell; pwd",
        "what did it print?",
        "/exit",
    ]);
    let mut out = RecordingOutput::default();

    let session = repl.run(session, &mut lines, &mut out).await;

    let statuses = out.statuses();
    assert!(
        statuses.contains(&"hello-from-shell".to_string()),
        "{statuses:?}"
    );
    let cwd = dir.path().canonicalize().unwrap();
    assert!(
        statuses.contains(&cwd.display().to_string()),
        "runs in the working directory: {statuses:?}"
    );
    let request = &provider.requests()[0];
    let texts = user_texts(request);
    assert_eq!(texts.len(), 2);
    assert!(
        texts[0].contains("$ echo hello-from-shell; pwd\nhello-from-shell\n"),
        "{}",
        texts[0]
    );
    assert_eq!(texts[1], "what did it print?");
    let saved = store.load(dir.path(), Some(&session.id)).unwrap().unwrap();
    assert_eq!(saved.messages, session.messages);
    assert_eq!(saved.messages.len(), 3);
}

/// A backend whose `context` returns a fixed block, standing in for the
/// rebuild after AIRLOK.md changes.
struct Reloading(&'static str);

impl Backend for Reloading {
    fn fresh_redactor(&mut self) -> Box<dyn Redactor> {
        Box::new(SecretRedactor::new())
    }

    fn switch(&mut self, _name: ProviderName, _seed: &RedactionMap) -> Result<Switch, String> {
        Err("not in this test".into())
    }

    fn context(&mut self) -> Option<String> {
        Some(self.0.to_string())
    }
}

#[tokio::test]
async fn a_hash_note_is_appended_to_airlok_md_and_applies_next_turn() {
    let dir = TempDir::new("repl-hash");
    let file = dir.path().join("AIRLOK.md");
    std::fs::write(&file, "# rules\n- existing").unwrap();
    let provider = MockProvider::scripted(vec![reply("ok")]);
    let mut agent = agent(provider.clone(), dir.path());
    let mut repl = Repl {
        agent: &mut agent,
        store: None,
        interrupt: Interrupt::new(),
        backend: Box::new(Reloading("RELOADED CONTEXT")),
        used: Vec::new(),
    };
    let session = repl.agent.new_session();
    let mut lines = ScriptedLines::typed(&["# always run the tests", "#- keep it short", "hi"]);
    let mut out = RecordingOutput::default();

    repl.run(session, &mut lines, &mut out).await;

    assert_eq!(
        std::fs::read_to_string(&file).unwrap(),
        "# rules\n- existing\n- always run the tests\n- keep it short\n"
    );
    let statuses = out.statuses();
    assert_eq!(
        statuses
            .iter()
            .filter(|s| *s == "added to ./AIRLOK.md")
            .count(),
        2
    );
    let requests = provider.requests();
    assert_eq!(requests.len(), 1, "a note is not a turn");
    assert!(requests[0].system.contains("RELOADED CONTEXT"));
}

#[tokio::test]
async fn a_hash_note_creates_airlok_md_and_says_what_it_hides() {
    let dir = TempDir::new("repl-hash-new");
    std::fs::write(dir.path().join("CLAUDE.md"), "- tabs\n").unwrap();
    let mut agent = agent(MockProvider::scripted(vec![]), dir.path());
    let mut repl = Repl {
        agent: &mut agent,
        store: None,
        interrupt: Interrupt::new(),
        backend: Box::new(TestBackend::default()),
        used: Vec::new(),
    };
    let session = repl.agent.new_session();
    let mut lines = ScriptedLines::typed(&["# prefer spaces"]);
    let mut out = RecordingOutput::default();

    repl.run(session, &mut lines, &mut out).await;

    assert_eq!(
        std::fs::read_to_string(dir.path().join("AIRLOK.md")).unwrap(),
        "- prefer spaces\n"
    );
    assert_eq!(
        out.statuses(),
        vec!["added to ./AIRLOK.md (new file, read instead of CLAUDE.md from now on)"]
    );
}

#[tokio::test]
async fn plan_then_go_researches_read_only_then_runs_with_every_tool() {
    const MODEL: &str = "gpt-6-astra";
    let dir = TempDir::new("repl-plan");
    let provider = MockProvider::scripted(vec![
        tool_call("t1", "glob", json!({"pattern": "*"})),
        reply("1. Create hello.txt containing hello."),
        tool_call(
            "t2",
            "write_file",
            json!({"path": "hello.txt", "content": "hello\n"}),
        ),
        reply("Created hello.txt."),
    ]);
    let mut agent = agent(provider.clone(), dir.path());
    agent.config_mut().provider.model = MODEL.into();
    agent.config_mut().models.insert(
        MODEL.into(),
        ModelConfig {
            reasoning_effort: Some("none".into()),
        },
    );
    let mut repl = Repl {
        agent: &mut agent,
        store: None,
        interrupt: Interrupt::new(),
        backend: Box::new(TestBackend::default()),
        used: Vec::new(),
    };
    let session = repl.agent.new_session();
    let mut lines =
        ScriptedLines::typed(&["/go", "/plan", "/go", "add a greeting file", "/go", "/exit"]);
    let mut out = RecordingOutput::default();

    repl.run(session, &mut lines, &mut out).await;

    let requests = provider.requests();
    assert_eq!(requests.len(), 4);
    let mut read_only: Vec<String> = READ_ONLY_TOOLS.iter().map(|n| n.to_string()).collect();
    read_only.sort();
    for planning in &requests[..2] {
        assert_eq!(tool_names(planning), read_only);
        assert!(planning.system.contains("Plan mode is on"));
    }
    for running in &requests[2..] {
        for name in ["write_file", "edit_file", "bash", "read_file"] {
            assert!(tool_names(running).contains(&name.to_string()), "{name}");
        }
        assert!(!running.system.contains("Plan mode"));
    }
    assert_eq!(
        user_texts(&requests[2]).last().copied(),
        Some("Carry out this plan:\n\n1. Create hello.txt containing hello.")
    );
    for request in &requests {
        assert_eq!(request.reasoning_effort.as_deref(), Some("none"));
    }
    assert_eq!(
        std::fs::read_to_string(dir.path().join("hello.txt")).unwrap(),
        "hello\n"
    );
    let statuses = out.statuses();
    assert!(statuses.contains(&"not in plan mode; /plan starts it".to_string()));
    assert!(statuses.contains(&"no plan yet: describe the task and wait for the plan".to_string()));
    assert!(
        lines.prompts[2].ends_with(" [plan]> "),
        "{:?}",
        lines.prompts
    );
    assert!(!lines.prompts[5].contains("[plan]"), "{:?}", lines.prompts);
}

#[tokio::test]
async fn plan_again_leaves_without_running_the_plan() {
    let dir = TempDir::new("repl-plan-leave");
    let provider = MockProvider::scripted(vec![reply("1. Do the thing.")]);
    let mut agent = agent(provider.clone(), dir.path());
    let mut repl = Repl {
        agent: &mut agent,
        store: None,
        interrupt: Interrupt::new(),
        backend: Box::new(TestBackend::default()),
        used: Vec::new(),
    };
    let session = repl.agent.new_session();
    let mut lines = ScriptedLines::typed(&["/plan", "plan it", "/plan", "/go"]);
    let mut out = RecordingOutput::default();

    repl.run(session, &mut lines, &mut out).await;

    assert_eq!(provider.requests().len(), 1);
    let statuses = out.statuses();
    assert!(statuses.contains(&"plan mode off".to_string()));
    assert!(statuses.contains(&"not in plan mode; /plan starts it".to_string()));
}

/// An agent on the openai provider, which is the only one that sends a
/// reasoning effort, with `model` as the model in use.
fn on_openai(provider: std::sync::Arc<MockProvider>, cwd: &std::path::Path, model: &str) -> Agent {
    let mut agent = agent(provider, cwd);
    agent.config_mut().provider.name = ProviderName::OpenAi;
    agent.config_mut().provider.model = model.into();
    agent
}

#[tokio::test]
async fn anthropic_says_it_does_not_take_a_reasoning_effort() {
    let dir = TempDir::new("effort-anthropic");
    let provider = MockProvider::scripted(vec![]);
    let mut agent = agent(provider, dir.path());
    let session = agent.new_session();
    let mut lines = ScriptedLines::typed(&["/effort", "/effort high"]);
    let mut out = RecordingOutput::default();
    {
        let mut repl = Repl {
            agent: &mut agent,
            store: None,
            interrupt: Interrupt::new(),
            backend: Box::new(TestBackend::default()),
            used: Vec::new(),
        };
        repl.run(session, &mut lines, &mut out).await;
    }

    let statuses = out.statuses();
    assert_eq!(
        statuses
            .iter()
            .filter(|line| line.contains("does not take a reasoning effort"))
            .count(),
        2,
        "with and without an argument: {statuses:?}"
    );
    assert!(
        agent
            .config()
            .models
            .values()
            .all(|m| m.reasoning_effort.is_none()),
        "nothing was set on a provider that cannot send it"
    );
}

#[tokio::test]
async fn the_effort_in_force_says_where_it_came_from() {
    let dir = TempDir::new("effort-provenance");
    let provider = MockProvider::scripted(vec![]);
    let mut agent = on_openai(provider, dir.path(), "gpt-6-astra");
    agent.config_mut().models.insert(
        "gpt-6-astra".into(),
        ModelConfig {
            reasoning_effort: Some("none".into()),
        },
    );
    let session = agent.new_session();
    // The config file gave this model "none"; the backend knows that, so
    // the first /effort reports the config and the second the session.
    let mut backend = TestBackend {
        choices: vec![Some("high".into())].into(),
        ..TestBackend::default()
    };
    backend.efforts.insert("gpt-6-astra", "none".into());
    let mut lines = ScriptedLines::typed(&["/effort", "/effort"]);
    let mut out = RecordingOutput::default();
    {
        let mut repl = Repl {
            agent: &mut agent,
            store: None,
            interrupt: Interrupt::new(),
            backend: Box::new(backend),
            used: Vec::new(),
        };
        repl.run(session, &mut lines, &mut out).await;
    }

    let statuses = out.statuses();
    assert!(
        statuses
            .iter()
            .any(|line| line.contains("none, from the config for this model")),
        "{statuses:?}"
    );
    assert!(
        statuses
            .iter()
            .any(|line| line.contains("high, set for this session")),
        "{statuses:?}"
    );
}

#[tokio::test]
async fn an_effort_set_in_the_session_reaches_the_next_request() {
    let dir = TempDir::new("effort-sent");
    let provider = MockProvider::scripted(vec![reply("before"), reply("after")]);
    let mut agent = on_openai(provider.clone(), dir.path(), "gpt-6-astra");
    let session = agent.new_session();
    let mut lines = ScriptedLines::typed(&["first", "/effort minimal", "second"]);
    let mut out = RecordingOutput::default();
    {
        let mut repl = Repl {
            agent: &mut agent,
            store: None,
            interrupt: Interrupt::new(),
            backend: Box::new(TestBackend::default()),
            used: Vec::new(),
        };
        repl.run(session, &mut lines, &mut out).await;
    }

    let requests = provider.requests();
    assert_eq!(requests.len(), 2, "{requests:?}");
    assert_eq!(requests[0].reasoning_effort, None, "nothing set yet");
    assert_eq!(
        requests[1].reasoning_effort.as_deref(),
        Some("minimal"),
        "/effort applies to the next turn"
    );
}

#[tokio::test]
async fn an_unknown_effort_is_questioned_rather_than_taken() {
    let dir = TempDir::new("effort-unknown");
    let provider = MockProvider::scripted(vec![]);
    let mut agent = on_openai(provider, dir.path(), "gpt-6-astra");
    let session = agent.new_session();
    let backend = TestBackend::default();
    let asked = backend.questions();
    let mut lines = ScriptedLines::typed(&["/effort enormous"]);
    let mut out = RecordingOutput::default();
    {
        let mut repl = Repl {
            agent: &mut agent,
            store: None,
            interrupt: Interrupt::new(),
            backend: Box::new(backend),
            used: Vec::new(),
        };
        repl.run(session, &mut lines, &mut out).await;
    }

    let statuses = out.statuses();
    assert!(
        statuses
            .iter()
            .any(|line| line.contains("not a reasoning effort airlok has seen here")),
        "{statuses:?}"
    );
    assert!(
        statuses
            .iter()
            .any(|line| line == "left the reasoning effort alone"),
        "{statuses:?}"
    );
    assert!(
        agent
            .config()
            .models
            .get("gpt-6-astra")
            .and_then(|m| m.reasoning_effort.as_deref())
            .is_none(),
        "an unknown effort must not be taken on its own"
    );
    let questions = asked.lock().unwrap();
    assert_eq!(questions.len(), 1, "{questions:?}");
    assert_eq!(
        questions[0].1.first().map(String::as_str),
        Some("enormous"),
        "the typed value is offered first, so Enter on it is a choice"
    );
}

#[tokio::test]
async fn a_goal_reaches_the_request_and_the_footer() {
    let dir = TempDir::new("repl-goal");
    let provider = MockProvider::scripted(vec![reply("working on it")]);
    let mut agent = agent(provider.clone(), dir.path());
    let session = agent.new_session();
    let mut lines = ScriptedLines::typed(&["/goal ship 0.9.0", "do the thing"]);
    let mut out = RecordingOutput::default();
    let session = {
        let mut repl = Repl {
            agent: &mut agent,
            store: None,
            interrupt: Interrupt::new(),
            backend: Box::new(TestBackend::default()),
            used: Vec::new(),
        };
        repl.run(session, &mut lines, &mut out).await
    };

    assert_eq!(session.goal.as_deref(), Some("ship 0.9.0"));
    let requests = provider.requests();
    assert_eq!(requests.len(), 1, "{requests:?}");
    assert!(
        requests[0].system.contains("ship 0.9.0"),
        "the goal is in the system prompt: {}",
        requests[0].system
    );
    let statuses = out.statuses();
    assert!(
        statuses
            .iter()
            .any(|line| line.contains("goal: ship 0.9.0")),
        "{statuses:?}"
    );
}

#[tokio::test]
async fn a_goal_can_be_shown_and_cleared() {
    let dir = TempDir::new("repl-goal-clear");
    let provider = MockProvider::scripted(vec![]);
    let mut agent = agent(provider, dir.path());
    let session = agent.new_session();
    let mut lines = ScriptedLines::typed(&[
        "/goal",
        "/goal keep it green",
        "/goal",
        "/goal clear",
        "/goal",
    ]);
    let mut out = RecordingOutput::default();
    let session = {
        let mut repl = Repl {
            agent: &mut agent,
            store: None,
            interrupt: Interrupt::new(),
            backend: Box::new(TestBackend::default()),
            used: Vec::new(),
        };
        repl.run(session, &mut lines, &mut out).await
    };

    assert_eq!(session.goal, None, "cleared");
    let statuses = out.statuses();
    assert_eq!(
        statuses
            .iter()
            .filter(|l| l.starts_with("no goal set"))
            .count(),
        2,
        "before it was set and after it was cleared: {statuses:?}"
    );
    assert!(statuses.iter().any(|l| l == "goal cleared"), "{statuses:?}");
}

#[tokio::test]
async fn an_unknown_model_id_is_questioned_rather_than_taken() {
    let dir = TempDir::new("repl-unknown-model");
    let provider = MockProvider::scripted(vec![reply("hi")]);
    let mut agent = agent(provider, dir.path());
    let before = agent.config().provider.model.clone();
    let session = agent.new_session();
    let backend = TestBackend::default();
    let asked = backend.questions();
    let mut lines = ScriptedLines::typed(&["/model gpt-9-nonesuch"]);
    let mut out = RecordingOutput::default();
    {
        let mut repl = Repl {
            agent: &mut agent,
            store: None,
            interrupt: Interrupt::new(),
            backend: Box::new(backend),
            used: Vec::new(),
        };
        repl.run(session, &mut lines, &mut out).await;
    }

    let statuses = out.statuses();
    assert!(
        statuses
            .iter()
            .any(|line| line.contains("not a model airlok has seen here")),
        "{statuses:?}"
    );
    assert!(
        statuses.iter().any(|line| line == "left the model alone"),
        "{statuses:?}"
    );
    assert_eq!(
        agent.config().provider.model,
        before,
        "an unknown id must not switch anything on its own"
    );

    let questions = asked.lock().unwrap();
    assert_eq!(questions.len(), 1, "{questions:?}");
    assert_eq!(
        questions[0].1.first().map(String::as_str),
        Some("gpt-9-nonesuch"),
        "the typed id is offered first, so Enter on it is a choice rather than a fuzzy match"
    );
}

#[tokio::test]
async fn a_near_miss_is_named_and_can_be_taken() {
    let dir = TempDir::new("repl-near-miss");
    let provider = MockProvider::scripted(vec![reply("hi")]);
    let mut agent = agent(provider, dir.path());
    let known = agent.config().provider.model.clone();
    // One character out from the model in use.
    let typo = format!("{}x", &known[..known.len() - 1]);
    let session = agent.new_session();
    let backend = TestBackend {
        choices: vec![Some(known.clone())].into(),
        ..TestBackend::default()
    };
    let mut lines = ScriptedLines::typed(&[&format!("/model {typo}")]);
    let mut out = RecordingOutput::default();
    {
        let mut repl = Repl {
            agent: &mut agent,
            store: None,
            interrupt: Interrupt::new(),
            backend: Box::new(backend),
            used: Vec::new(),
        };
        repl.run(session, &mut lines, &mut out).await;
    }

    let statuses = out.statuses();
    assert!(
        statuses
            .iter()
            .any(|line| line.starts_with("did you mean:") && line.contains(&known)),
        "{statuses:?}"
    );
    // Picking the offered id switches to it, rather than to the typo.
    assert_eq!(agent.config().provider.model, known);
}

#[tokio::test]
async fn a_provider_failure_reads_as_a_sentence_and_the_session_stays_open() {
    let dir = TempDir::new("repl-404");
    let provider = MockProvider::failing(
        404,
        r#"{"error":{"code":"DeploymentNotFound","message":"The API deployment for this resource does not exist"}}"#,
    );
    let mut agent = agent(provider, dir.path());
    let session = agent.new_session();
    let mut lines = ScriptedLines::typed(&["hello", "/cost"]);
    let mut out = RecordingOutput::default();
    {
        let mut repl = Repl {
            agent: &mut agent,
            store: None,
            interrupt: Interrupt::new(),
            backend: Box::new(TestBackend::default()),
            used: Vec::new(),
        };
        repl.run(session, &mut lines, &mut out).await;
    }

    let statuses = out.statuses();
    assert!(
        statuses
            .iter()
            .any(|line| line.contains("has no model called")),
        "{statuses:?}"
    );
    assert!(
        !statuses
            .iter()
            .any(|line| line.contains("DeploymentNotFound")),
        "the raw body belongs in the debug log, not on the prompt: {statuses:?}"
    );
    // The loop carried on: the command after the failed turn ran.
    assert!(
        statuses.iter().any(|line| line.starts_with("input ")),
        "the session should still be open: {statuses:?}"
    );
}

#[tokio::test]
async fn a_footer_follows_each_turn() {
    let dir = TempDir::new("repl-footer");
    let provider = MockProvider::scripted(vec![with_usage(50_000, 10, reply("hi"))]);
    let mut agent = agent(provider, dir.path());
    let mut repl = Repl {
        agent: &mut agent,
        store: None,
        interrupt: Interrupt::new(),
        backend: Box::new(TestBackend::default()),
        used: Vec::new(),
    };
    let session = repl.agent.new_session();
    let mut lines = ScriptedLines::typed(&["hello"]);
    let mut out = RecordingOutput::default();

    let session = repl.run(session, &mut lines, &mut out).await;

    assert_eq!(
        out.statuses().last().cloned(),
        Some(format!(
            "{} · context 25% · session {}",
            airlok_llm::anthropic::DEFAULT_MODEL,
            session.id
        ))
    );
}
