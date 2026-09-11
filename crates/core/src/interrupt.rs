//! A cancel signal for the turn in progress. The REPL triggers it on
//! Ctrl-C; the agent watches it while streaming and running tools.
//!
//! Each turn watches only triggers that happen after it started, so a
//! Ctrl-C from before the turn (queued while a prompt was open) is ignored.

use std::sync::Arc;

use tokio::sync::watch;

#[derive(Clone, Debug)]
pub struct Interrupt {
    count: Arc<watch::Sender<u64>>,
}

impl Default for Interrupt {
    fn default() -> Self {
        Self::new()
    }
}

impl Interrupt {
    pub fn new() -> Self {
        Self {
            count: Arc::new(watch::channel(0).0),
        }
    }

    pub fn trigger(&self) {
        self.count.send_modify(|n| *n += 1);
    }

    /// A watcher that fires on the next trigger from now on.
    pub fn watcher(&self) -> Watcher {
        Watcher {
            seen: *self.count.borrow(),
            rx: self.count.subscribe(),
        }
    }
}

pub struct Watcher {
    rx: watch::Receiver<u64>,
    seen: u64,
}

impl Watcher {
    /// Resolves once triggered. Never resolves if the [`Interrupt`] was
    /// dropped, so a missing signal source cannot cancel a turn.
    pub async fn triggered(&mut self) {
        let seen = self.seen;
        if self.rx.wait_for(|n| *n > seen).await.is_err() {
            std::future::pending::<()>().await;
        }
    }
}
