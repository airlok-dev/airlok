use std::time::Duration;

use airlok_core::redact::{Class, Redactor, SecretRedactor};
use airlok_core::{Session, SessionStore};
use airlok_llm::{ContentBlock, Role};
use airlok_tests::{agent, agent_with, reply, MockProvider, RecordingOutput, TempDir};

const KEY: &str = "sk-ant-api03-provider-key-that-never-hits-disk-0000000000";
const TOKEN: &str = concat!("ghp_", "abcdefghijklmnopqrstuvwxyz0123456789");

#[cfg(unix)]
fn mode(path: &std::path::Path) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path).unwrap().permissions().mode() & 0o777
}

/// One saved turn whose prompt carries a file secret, run with a provider
/// key registered as redact-only.
async fn saved_session(dir: &TempDir, store: &SessionStore) -> Session {
    let provider = MockProvider::scripted(vec![reply("noted")]);
    let redactor = SecretRedactor::new().with_known("the provider API key", KEY, Class::RedactOnly);
    let mut agent =
        agent_with(provider.clone(), dir.path(), redactor).with_context("old context".into());
    let mut session = agent.new_session();
    let mut out = RecordingOutput::default();
    agent
        .turn(&mut session, &format!("the token is {TOKEN}"), &mut out)
        .await
        .unwrap();
    let sent = &provider.requests()[0].messages[0];
    assert!(
        !format!("{sent:?}").contains(TOKEN),
        "token left the machine"
    );
    store.save(&session).unwrap();
    session
}

#[tokio::test]
async fn round_trip_keeps_history_and_file_secrets_but_never_the_provider_key() {
    let dir = TempDir::new("session-roundtrip");
    let store = SessionStore::new(dir.path().join("data"));
    let session = saved_session(&dir, &store).await;

    let path = store.save(&session).unwrap();
    #[cfg(unix)]
    {
        assert_eq!(mode(&path), 0o600, "session file must be private");
        assert_eq!(mode(path.parent().unwrap()), 0o700);
        assert_eq!(mode(&dir.path().join("data")), 0o700);
    }
    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(
        raw.contains(TOKEN),
        "file secrets are stored so they can be rehydrated"
    );
    assert!(
        !raw.contains(KEY),
        "the provider key must never be written to disk"
    );

    let loaded = store.load(dir.path(), None).unwrap().unwrap();
    assert_eq!(loaded.id, session.id);
    assert_eq!(loaded.messages, session.messages);
    assert_eq!(loaded.usage, session.usage);
    assert_eq!(loaded.config, session.config);
    let key_entry = loaded
        .redactions
        .values()
        .find(|e| e.class == Class::RedactOnly)
        .expect("the key's placeholder survives");
    assert_eq!(key_entry.value, "");
    let token_entry = loaded
        .redactions
        .values()
        .find(|e| e.class == Class::Rehydrate)
        .unwrap();
    assert_eq!(token_entry.value, TOKEN);
    assert_eq!(loaded, session.for_disk());
}

#[tokio::test]
async fn resume_adds_a_system_note_and_uses_the_rebuilt_context() {
    let dir = TempDir::new("session-resume");
    let store = SessionStore::new(dir.path().join("data"));
    let saved = saved_session(&dir, &store).await;

    let mut session = store
        .load(dir.path(), Some(&saved.id[..4]))
        .unwrap()
        .unwrap();
    let at = session.resume().to_string();
    let provider = MockProvider::scripted(vec![reply(&format!("still {TOKEN}"))]);
    let redactor = SecretRedactor::new()
        .with_map(&session.redactions)
        .with_known("the provider API key", KEY, Class::RedactOnly);
    let mut agent =
        agent_with(provider.clone(), dir.path(), redactor).with_context("new context".into());
    let mut out = RecordingOutput::default();
    agent
        .turn(&mut session, "what was the token?", &mut out)
        .await
        .unwrap();

    let request = &provider.requests()[0];
    assert!(request
        .system
        .contains(&format!("This session was resumed at {at}")));
    assert!(request.system.contains("new context"));
    assert!(!request.system.contains("old context"));
    // The whole earlier conversation went with the new prompt, redacted
    // with the same placeholder as before.
    assert_eq!(request.messages.len(), 3);
    let first = format!("{:?}", request.messages[0]);
    assert!(first.contains("<<SECRET_"));
    assert!(!first.contains(TOKEN));
    // And the reply rehydrated through the seeded map.
    match session.messages.last().unwrap().content.first().unwrap() {
        ContentBlock::Text { text } => assert_eq!(text, &format!("still {TOKEN}")),
        other => panic!("unexpected block {other:?}"),
    }
    assert_eq!(session.resumed_at.len(), 1);
    assert_eq!(session.messages[0].role, Role::User);
}

#[test]
fn a_seeded_redactor_reuses_saved_placeholders_and_numbers_new_ones_after() {
    let mut first =
        SecretRedactor::new().with_known("the provider API key", KEY, Class::RedactOnly);
    let (_, map) = first.redact(&format!("{KEY} {TOKEN}"));
    let mut disk = Session::new(
        std::path::Path::new("."),
        &airlok_core::Config::new(".".into()),
    );
    disk.redactions = map;
    let disk = disk.for_disk();

    let mut resumed = SecretRedactor::new().with_map(&disk.redactions);
    let (out, _) = resumed.redact(TOKEN);
    assert_eq!(out, "<<SECRET_2>>");
    let other = concat!("ghp_", "zyxwvutsrqponmlkjihgfedcba9876543210");
    let (out, map) = resumed.redact(other);
    assert_eq!(out, "<<SECRET_3>>");
    assert_eq!(map["<<SECRET_1>>"].value, "", "the old key stays blank");
    assert_eq!(map["<<SECRET_1>>"].class, Class::RedactOnly);
    assert_eq!(resumed.redact("").0, "");
}

#[tokio::test]
async fn list_rm_and_clean_manage_the_directory() {
    let dir = TempDir::new("session-list");
    let store = SessionStore::new(dir.path().join("data"));
    let provider = MockProvider::scripted(vec![reply("a"), reply("b")]);
    let mut agent = agent(provider, dir.path());
    let mut out = RecordingOutput::default();

    let mut old = agent.new_session();
    agent
        .turn(&mut old, "first prompt", &mut out)
        .await
        .unwrap();
    old.updated_at = "2020-01-01T00:00:00Z".to_string();
    store.save(&old).unwrap();
    let mut recent = agent.new_session();
    agent
        .turn(&mut recent, "second prompt", &mut out)
        .await
        .unwrap();
    store.save(&recent).unwrap();

    let listed = store.list(dir.path()).unwrap();
    assert_eq!(
        listed.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(),
        vec![recent.id.as_str(), old.id.as_str()],
        "most recently updated first"
    );
    assert_eq!(listed[0].first_prompt.as_deref(), Some("second prompt"));
    assert_eq!(listed[0].turns, 1);
    // Another directory sees nothing.
    assert!(store
        .list(&dir.path().join("elsewhere"))
        .unwrap()
        .is_empty());

    let removed = store.clean(Duration::from_secs(86_400)).unwrap();
    assert_eq!(removed.len(), 1);
    assert_eq!(store.list(dir.path()).unwrap().len(), 1);

    assert!(store.remove(dir.path(), &recent.id).unwrap());
    assert!(!store.remove(dir.path(), &recent.id).unwrap());
    assert!(store.load(dir.path(), None).unwrap().is_none());
}
