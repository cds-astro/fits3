use winit::{event::*, keyboard::KeyCode};
use winit::keyboard::{ModifiersState, PhysicalKey};

pub(crate) struct ShortKeyCommands {
    modifiers: ModifiersState,
}

impl ShortKeyCommands {
    // Create a new instance
    pub(crate) fn new() -> Self {
        Self {
            modifiers: ModifiersState::empty(),
        }
    }

    // Update the stored modifiers when the event comes in
    pub(crate) fn update_modifiers(&mut self, new_modifiers: Modifiers) {
        self.modifiers = new_modifiers.state();
    }

    /// Process a key event with a given key and callback
    /// - `key`: the key you want to trigger on (e.g., VirtualKeyCode::O)
    /// - `event`: the event
    /// - `require_ctrl`: whether Ctrl should be held
    /// - `f`: callback to call when combination matches
    pub(crate) fn process_key_event<F>(
        &self,
        key: PhysicalKey,
        event: &KeyEvent,
        require_ctrl: bool,
        f: F,
    )
    where
        F: FnOnce(),
    {
        // Only act on key press
        if event.state != ElementState::Pressed {
            return;
        }

        // Match the key
        if event.physical_key == key {
            let ctrl_pressed = if is_macos() {
                self.modifiers.super_key() // Cmd
            } else {
                self.modifiers.control_key()
            };

            if require_ctrl && !ctrl_pressed {
                return; // Ctrl required but not pressed
            }

            // Trigger the callback
            f();
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn is_macos() -> bool {
    let window = web_sys::window().unwrap();
    let navigator = window.navigator();
    let user_agent = navigator.user_agent().unwrap_or_default();

    user_agent.contains("Mac")
}

#[cfg(not(target_arch = "wasm32"))]
fn is_macos() -> bool {
    cfg!(target_os = "macos")
}