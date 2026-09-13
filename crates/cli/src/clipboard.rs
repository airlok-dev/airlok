//! Reading an image from the system clipboard.
//!
//! The trait has one method and it reads. There is no write path here at
//! all, so "reading never clobbers the clipboard" is a property of the
//! shape rather than a promise about the code: a backend has nothing to
//! write with.
//!
//! Every failure names its reason. A silent no-op after Ctrl+V is the
//! thing this module exists to avoid.

use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ClipboardError {
    #[error("the clipboard holds no image")]
    NoImage,
    #[error("{0} is not installed, so the clipboard cannot be read here")]
    ToolMissing(String),
    #[error("no display server is running, so there is no clipboard to read")]
    NoDisplayServer,
    #[error("the clipboard image could not be read: {0}")]
    Unreadable(String),
    #[error("reading the clipboard is not supported on this platform")]
    Unsupported,
}

/// One way of getting an image out of the clipboard. Read only, on
/// purpose: see the module comment.
pub trait ClipboardBackend {
    /// The image and its media type, or why there is none.
    fn read_image(&self) -> Result<(Vec<u8>, &'static str), ClipboardError>;

    /// Named in `-v` output, so a failure can be traced to the tool.
    fn name(&self) -> &'static str;
}

/// Whether a program is on PATH, without running it.
fn on_path(program: &str) -> bool {
    std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).any(|dir| dir.join(program).is_file()))
        .unwrap_or(false)
}

/// macOS: AppleScript writes the PNG on the pasteboard to a file, since
/// osascript cannot hand back binary on stdout.
pub struct Osascript;

impl ClipboardBackend for Osascript {
    fn name(&self) -> &'static str {
        "osascript"
    }

    fn read_image(&self) -> Result<(Vec<u8>, &'static str), ClipboardError> {
        let dir = std::env::temp_dir();
        let file = dir.join(format!("airlok-clip-{}.png", std::process::id()));
        let script = format!(
            "set f to (open for access POSIX file \"{}\" with write permission)\n\
             try\n\
             set eof f to 0\n\
             write (the clipboard as «class PNGf») to f\n\
             close access f\n\
             on error e number n\n\
             close access f\n\
             error e number n\n\
             end try",
            file.display()
        );
        let output = Command::new("osascript")
            .arg("-e")
            .arg(&script)
            .output()
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => ClipboardError::ToolMissing("osascript".into()),
                _ => ClipboardError::Unreadable(e.to_string()),
            })?;
        let read = read_and_remove(&file);
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            // What AppleScript says when the pasteboard holds no PNG.
            if stderr.contains("into the expected type") || stderr.contains("-1700") {
                return Err(ClipboardError::NoImage);
            }
            return Err(ClipboardError::Unreadable(stderr.trim().to_string()));
        }
        match read {
            Some(bytes) if !bytes.is_empty() => Ok((bytes, "image/png")),
            _ => Err(ClipboardError::NoImage),
        }
    }
}

/// Reads the file the backend wrote and removes it either way, so no
/// copy of a screenshot is left behind.
fn read_and_remove(file: &Path) -> Option<Vec<u8>> {
    let bytes = std::fs::read(file).ok();
    let _ = std::fs::remove_file(file);
    bytes
}

/// A backend that runs a command and takes the image from its stdout.
pub struct Piped {
    program: &'static str,
    args: &'static [&'static str],
}

impl ClipboardBackend for Piped {
    fn name(&self) -> &'static str {
        self.program
    }

    fn read_image(&self) -> Result<(Vec<u8>, &'static str), ClipboardError> {
        let output = Command::new(self.program)
            .args(self.args)
            .output()
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => ClipboardError::ToolMissing(self.program.into()),
                _ => ClipboardError::Unreadable(e.to_string()),
            })?;
        if !output.status.success() || output.stdout.is_empty() {
            return Err(ClipboardError::NoImage);
        }
        Ok((output.stdout, "image/png"))
    }
}

/// Windows, where PowerShell writes the clipboard image to a file.
pub struct PowerShell;

impl ClipboardBackend for PowerShell {
    fn name(&self) -> &'static str {
        "powershell"
    }

    fn read_image(&self) -> Result<(Vec<u8>, &'static str), ClipboardError> {
        let file = std::env::temp_dir().join(format!("airlok-clip-{}.png", std::process::id()));
        let script = format!(
            "Add-Type -AssemblyName System.Windows.Forms; \
             $i = [Windows.Forms.Clipboard]::GetImage(); \
             if ($i -eq $null) {{ exit 2 }}; \
             $i.Save('{}', [System.Drawing.Imaging.ImageFormat]::Png)",
            file.display()
        );
        let output = Command::new("powershell")
            .args(["-NoProfile", "-Command", &script])
            .output()
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => ClipboardError::ToolMissing("powershell".into()),
                _ => ClipboardError::Unreadable(e.to_string()),
            })?;
        let read = read_and_remove(&file);
        if output.status.code() == Some(2) {
            return Err(ClipboardError::NoImage);
        }
        match read {
            Some(bytes) if !bytes.is_empty() => Ok((bytes, "image/png")),
            _ => Err(ClipboardError::NoImage),
        }
    }
}

/// The backend for this machine, or why there is none. The display check
/// comes first on Linux so a headless session says that rather than
/// blaming a missing tool.
pub fn backend() -> Result<Box<dyn ClipboardBackend>, ClipboardError> {
    if cfg!(target_os = "macos") {
        return Ok(Box::new(Osascript));
    }
    if cfg!(windows) {
        return Ok(Box::new(PowerShell));
    }
    if cfg!(target_os = "linux") {
        let wayland = std::env::var_os("WAYLAND_DISPLAY").is_some();
        let x11 = std::env::var_os("DISPLAY").is_some();
        if !wayland && !x11 {
            return Err(ClipboardError::NoDisplayServer);
        }
        if wayland && on_path("wl-paste") {
            return Ok(Box::new(Piped {
                program: "wl-paste",
                args: &["-t", "image/png"],
            }));
        }
        if on_path("xclip") {
            return Ok(Box::new(Piped {
                program: "xclip",
                args: &["-selection", "clipboard", "-t", "image/png", "-o"],
            }));
        }
        // xsel is often present and cannot help: it has no way to ask for
        // a target, so it can only ever return text. Saying so beats
        // running it and reporting an empty clipboard.
        if on_path("xsel") {
            return Err(ClipboardError::ToolMissing(
                "wl-paste or xclip (xsel is installed but cannot read images)".into(),
            ));
        }
        return Err(ClipboardError::ToolMissing("wl-paste or xclip".into()));
    }
    Err(ClipboardError::Unsupported)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stands in for a real clipboard. It records every call, and the
    /// trait gives it nothing to write with.
    struct Mock {
        answer: Result<(Vec<u8>, &'static str), ClipboardError>,
        calls: std::cell::RefCell<Vec<&'static str>>,
    }

    impl ClipboardBackend for Mock {
        fn name(&self) -> &'static str {
            "mock"
        }

        fn read_image(&self) -> Result<(Vec<u8>, &'static str), ClipboardError> {
            self.calls.borrow_mut().push("read_image");
            self.answer.clone()
        }
    }

    fn mock(answer: Result<(Vec<u8>, &'static str), ClipboardError>) -> Mock {
        Mock {
            answer,
            calls: std::cell::RefCell::new(Vec::new()),
        }
    }

    #[test]
    fn every_failure_names_its_reason() {
        for (error, expected) in [
            (ClipboardError::NoImage, "holds no image"),
            (
                ClipboardError::ToolMissing("wl-paste".into()),
                "is not installed",
            ),
            (ClipboardError::NoDisplayServer, "no display server"),
            (
                ClipboardError::Unreadable("boom".into()),
                "could not be read: boom",
            ),
            (
                ClipboardError::Unsupported,
                "not supported on this platform",
            ),
        ] {
            let said = error.to_string();
            assert!(said.contains(expected), "{said}");
            assert!(!said.is_empty());
        }
    }

    #[test]
    fn a_failed_read_touches_the_clipboard_once_and_writes_nothing() {
        let backend = mock(Err(ClipboardError::NoImage));
        assert_eq!(backend.read_image(), Err(ClipboardError::NoImage));
        // The only thing a backend can do is read, so a failure cannot
        // have replaced what was on the clipboard.
        assert_eq!(*backend.calls.borrow(), vec!["read_image"]);
    }

    #[test]
    fn a_successful_read_returns_bytes_and_a_media_type() {
        let backend = mock(Ok((vec![1, 2, 3], "image/png")));
        let (bytes, media_type) = backend.read_image().unwrap();
        assert_eq!(bytes, vec![1, 2, 3]);
        assert_eq!(media_type, "image/png");
        assert_eq!(*backend.calls.borrow(), vec!["read_image"]);
    }
}
