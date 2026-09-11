//! Keys during a turn. The terminal goes into cbreak mode: no line
//! buffering and no echo, but signals and output processing stay on, so
//! Esc arrives as a byte, Ctrl-C still raises SIGINT for the handler the
//! REPL already has, and a newline still returns the carriage. A bare Esc
//! interrupts the turn; printable keys are kept for the next prompt.
//!
//! The guard restores the mode when dropped. A panic hook and the REPL's
//! SIGTERM handler call [`restore`] for the ways out that skip it.

use std::fs::File;
use std::io::Read;
use std::os::fd::{AsRawFd, RawFd};
use std::sync::{Arc, Mutex, Once, PoisonError, TryLockError};
use std::time::Duration;

use airlok_core::Interrupt;

/// How long to wait after an Esc for the rest of an escape sequence.
const ESC_WAIT: Duration = Duration::from_millis(50);

/// The mode to put back, while cbreak is on.
static SAVED: Mutex<Option<(RawFd, libc::termios)>> = Mutex::new(None);

/// Cbreak mode on a terminal, until dropped.
pub struct Cbreak {
    fd: RawFd,
    saved: libc::termios,
}

impl Cbreak {
    pub fn enter(fd: RawFd) -> std::io::Result<Self> {
        // SAFETY: termios is plain data, filled in by tcgetattr.
        let mut saved: libc::termios = unsafe { std::mem::zeroed() };
        if unsafe { libc::tcgetattr(fd, &mut saved) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        let mut cbreak = saved;
        cbreak.c_lflag &= !(libc::ICANON | libc::ECHO);
        cbreak.c_cc[libc::VMIN] = 1;
        cbreak.c_cc[libc::VTIME] = 0;
        install_panic_hook();
        *SAVED.lock().unwrap_or_else(PoisonError::into_inner) = Some((fd, saved));
        // SAFETY: `cbreak` is a valid termios derived from the current one.
        if unsafe { libc::tcsetattr(fd, libc::TCSANOW, &cbreak) } != 0 {
            let error = std::io::Error::last_os_error();
            *SAVED.lock().unwrap_or_else(PoisonError::into_inner) = None;
            return Err(error);
        }
        Ok(Self { fd, saved })
    }
}

impl Drop for Cbreak {
    fn drop(&mut self) {
        // SAFETY: puts back the attributes read from this fd in `enter`.
        unsafe { libc::tcsetattr(self.fd, libc::TCSANOW, &self.saved) };
        *SAVED.lock().unwrap_or_else(PoisonError::into_inner) = None;
    }
}

/// Puts the terminal back if cbreak is on. Never waits for the lock, so
/// a panic hook or a signal task can call it.
pub fn restore() {
    let mut saved = match SAVED.try_lock() {
        Ok(guard) => guard,
        Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        Err(TryLockError::WouldBlock) => return,
    };
    if let Some((fd, mode)) = saved.take() {
        // SAFETY: `mode` was read from `fd` when cbreak was entered.
        unsafe { libc::tcsetattr(fd, libc::TCSANOW, &mode) };
    }
}

fn install_panic_hook() {
    static HOOK: Once = Once::new();
    HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            restore();
            previous(info);
        }));
    });
}

/// Reads the terminal during a turn: a bare Esc fires the interrupt,
/// printable input is kept for the next prompt, the rest is dropped.
pub struct Keys {
    tty: File,
    interrupt: Interrupt,
    typed: Arc<Mutex<Vec<u8>>>,
}

impl Keys {
    pub fn new(tty: File, interrupt: Interrupt, typed: Arc<Mutex<Vec<u8>>>) -> Self {
        Self {
            tty,
            interrupt,
            typed,
        }
    }

    pub fn fd(&self) -> RawFd {
        self.tty.as_raw_fd()
    }

    /// Waits up to `timeout` for input and handles whatever arrived.
    pub fn poll(&mut self, timeout: Duration) {
        if !readable(self.fd(), timeout) {
            return;
        }
        let mut buf = [0u8; 256];
        let mut bytes = match self.tty.read(&mut buf) {
            Ok(n) if n > 0 => buf[..n].to_vec(),
            // Hung up or failing: wait rather than spin on it.
            _ => {
                std::thread::sleep(timeout);
                return;
            }
        };
        // An Esc that ends the read may start a sequence whose rest has
        // not arrived yet.
        if bytes.last() == Some(&0x1b) && readable(self.fd(), ESC_WAIT) {
            if let Ok(n) = self.tty.read(&mut buf) {
                bytes.extend_from_slice(&buf[..n]);
            }
        }
        let esc = scan(
            &bytes,
            &mut self.typed.lock().unwrap_or_else(PoisonError::into_inner),
        );
        if esc {
            self.interrupt.trigger();
        }
    }
}

/// Handles the bytes of one read. Returns whether a bare Esc was among
/// them; keeps printable text in `typed`, with Backspace taking back the
/// last character. Escape sequences (arrows, function keys, Alt+key) and
/// other control keys, Enter included, are dropped.
fn scan(bytes: &[u8], typed: &mut Vec<u8>) -> bool {
    let mut esc = false;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            0x1b => match bytes.get(i + 1) {
                // Alone, or followed by another Esc: this one was a key.
                None | Some(0x1b) => {
                    esc = true;
                    i += 1;
                }
                // CSI or SS3: parameter bytes, then a final byte in @..~.
                Some(b'[' | b'O') => {
                    i += 2;
                    while i < bytes.len() && !(0x40..=0x7e).contains(&bytes[i]) {
                        i += 1;
                    }
                    i += 1;
                }
                // Alt+key.
                Some(_) => i += 2,
            },
            0x7f | 0x08 => {
                pop_char(typed);
                i += 1;
            }
            byte if byte >= 0x20 => {
                typed.push(byte);
                i += 1;
            }
            _ => i += 1,
        }
    }
    esc
}

/// Removes the last UTF-8 character.
fn pop_char(typed: &mut Vec<u8>) {
    while let Some(byte) = typed.pop() {
        if byte & 0xc0 != 0x80 {
            break;
        }
    }
}

/// Whether `fd` has input within `timeout`.
fn readable(fd: RawFd, timeout: Duration) -> bool {
    let mut poll = libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    };
    let ms = i32::try_from(timeout.as_millis()).unwrap_or(i32::MAX);
    // SAFETY: one valid pollfd, and its count.
    let ready = unsafe { libc::poll(&mut poll, 1, ms) };
    ready > 0 && poll.revents & libc::POLLIN != 0
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use std::io::Write;
    use std::os::fd::FromRawFd;
    use std::sync::MutexGuard;

    use airlok_core::interrupt::Watcher;

    /// Tests that enter cbreak share `SAVED`, so they take turns.
    static SERIAL: Mutex<()> = Mutex::new(());

    pub fn serial() -> MutexGuard<'static, ()> {
        SERIAL.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A pseudo-terminal: the controller end, and the terminal end that
    /// stands in for the user's tty.
    pub fn pty() -> (File, File) {
        let (mut controller, mut terminal) = (0, 0);
        // SAFETY: openpty fills both fds; the other arguments may be null.
        let rc = unsafe {
            libc::openpty(
                &mut controller,
                &mut terminal,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        assert_eq!(rc, 0, "openpty: {}", std::io::Error::last_os_error());
        // SAFETY: both fds are open and owned by nothing else.
        unsafe { (File::from_raw_fd(controller), File::from_raw_fd(terminal)) }
    }

    pub fn mode(fd: RawFd) -> libc::termios {
        // SAFETY: as in `Cbreak::enter`.
        let mut mode: libc::termios = unsafe { std::mem::zeroed() };
        assert_eq!(unsafe { libc::tcgetattr(fd, &mut mode) }, 0);
        mode
    }

    /// Local flags without PENDIN, which macOS sets by itself when a
    /// terminal returns to canonical mode, to reprocess pending input.
    pub fn local_flags(mode: &libc::termios) -> libc::tcflag_t {
        mode.c_lflag & !libc::PENDIN
    }

    fn fired(watcher: &mut Watcher) -> bool {
        tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap()
            .block_on(async {
                tokio::time::timeout(Duration::from_millis(20), watcher.triggered())
                    .await
                    .is_ok()
            })
    }

    #[test]
    fn a_bare_esc_interrupts_and_sequences_do_not() {
        let mut typed = Vec::new();
        assert!(scan(b"\x1b", &mut typed));
        assert!(!scan(b"\x1b[A", &mut typed), "up arrow");
        assert!(!scan(b"\x1b[1;5C", &mut typed), "ctrl+right");
        assert!(!scan(b"\x1bOP", &mut typed), "F1");
        assert!(!scan(b"\x1bf", &mut typed), "alt+f");
        assert!(scan(b"\x1b\x1b[A", &mut typed), "esc, then an arrow");
        assert!(typed.is_empty());
    }

    #[test]
    fn printable_keys_are_kept_for_the_next_prompt() {
        let mut typed = Vec::new();
        assert!(!scan("fix the té\x7fest\r\x03".as_bytes(), &mut typed));
        assert_eq!(String::from_utf8(typed).unwrap(), "fix the test");
    }

    #[test]
    fn cbreak_keeps_signals_and_newlines_and_is_undone_on_drop() {
        let _serial = serial();
        let (_controller, terminal) = pty();
        let fd = terminal.as_raw_fd();
        let before = mode(fd);
        assert_ne!(before.c_lflag & libc::ICANON, 0);
        {
            let _cbreak = Cbreak::enter(fd).unwrap();
            let on = mode(fd);
            assert_eq!(on.c_lflag & (libc::ICANON | libc::ECHO), 0);
            assert_ne!(on.c_lflag & libc::ISIG, 0, "Ctrl-C still raises SIGINT");
            assert_ne!(
                on.c_oflag & libc::OPOST,
                0,
                "\\n still returns the carriage"
            );
        }
        let after = mode(fd);
        assert_eq!(local_flags(&after), local_flags(&before));
        assert_eq!(after.c_oflag, before.c_oflag);
    }

    #[test]
    fn a_panic_anywhere_puts_the_terminal_back() {
        let _serial = serial();
        let (_controller, terminal) = pty();
        let fd = terminal.as_raw_fd();
        let before = mode(fd);
        // As if the guard's destructor never ran.
        std::mem::forget(Cbreak::enter(fd).unwrap());
        assert_eq!(mode(fd).c_lflag & libc::ICANON, 0);
        let _ = std::thread::spawn(|| panic!("a panic on another thread")).join();
        assert_eq!(local_flags(&mode(fd)), local_flags(&before));
    }

    #[test]
    fn esc_on_the_terminal_fires_the_interrupt_and_an_arrow_does_not() {
        let _serial = serial();
        let (mut controller, terminal) = pty();
        let _cbreak = Cbreak::enter(terminal.as_raw_fd()).unwrap();
        let interrupt = Interrupt::new();
        let mut watcher = interrupt.watcher();
        let typed = Arc::new(Mutex::new(Vec::new()));
        let mut keys = Keys::new(terminal, interrupt, typed.clone());
        let wait = Duration::from_millis(500);

        controller.write_all(b"\x1b[A").unwrap();
        keys.poll(wait);
        assert!(!fired(&mut watcher), "an arrow key is not Esc");
        controller.write_all(b"ls").unwrap();
        keys.poll(wait);
        assert_eq!(*typed.lock().unwrap(), b"ls");
        controller.write_all(b"\x1b").unwrap();
        keys.poll(wait);
        assert!(fired(&mut watcher), "Esc fires the interrupt");
    }
}
