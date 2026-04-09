use glam::Vec2;
use std::collections::HashSet;
use winit::event::MouseScrollDelta;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

pub struct Input {
    keys_down: HashSet<KeyCode>,
    keys_pressed: HashSet<KeyCode>, // pressed this frame

    mouse_down: HashSet<MouseButton>,
    mouse_pressed: HashSet<MouseButton>,

    mouse_pos: Vec2,
    mouse_delta: Vec2,
    mouse_moved: bool,
    mouse_wheel: MouseScrollDelta,
    mouse_wheeled: bool,
}

impl Input {
    pub fn new() -> Self {
        Self {
            keys_down: HashSet::new(),
            keys_pressed: HashSet::new(),
            mouse_down: HashSet::new(),
            mouse_pressed: HashSet::new(),
            mouse_pos: Vec2::ZERO,
            mouse_delta: Vec2::ZERO,
            mouse_moved: false,
            mouse_wheel: MouseScrollDelta::LineDelta(0.0, 0.0),
            mouse_wheeled: false,
        }
    }

    pub fn update_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            if self.keys_down.insert(code) {
                                self.keys_pressed.insert(code);
                            }
                        }
                        ElementState::Released => {
                            self.keys_down.remove(&code);
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => match state {
                ElementState::Pressed => {
                    if self.mouse_down.insert(*button) {
                        self.mouse_pressed.insert(*button);
                    }
                }
                ElementState::Released => {
                    self.mouse_down.remove(button);
                }
            },
            WindowEvent::CursorMoved { position, .. } => {
                let position = Vec2::new(position.x as f32, position.y as f32);
                self.mouse_delta = position - self.mouse_pos;
                self.mouse_pos = position;

                self.mouse_moved = true;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                self.mouse_wheel = *delta;

                self.mouse_wheeled = true;
            }
            _ => {}
        }
    }

    pub fn end_frame(&mut self) {
        self.keys_pressed.clear();
        self.mouse_pressed.clear();
        self.mouse_moved = false;
        self.mouse_wheeled = false;
    }

    // --- queries ---
    #[allow(dead_code)]
    pub fn key_down(&self, key: KeyCode) -> bool {
        self.keys_down.contains(&key)
    }

    #[allow(dead_code)]
    pub fn key_pressed(&self, key: KeyCode) -> bool {
        self.keys_pressed.contains(&key)
    }

    pub fn mouse_down(&self, button: MouseButton) -> bool {
        self.mouse_down.contains(&button)
    }

    pub fn mouse_pressed(&self, button: MouseButton) -> bool {
        self.mouse_pressed.contains(&button)
    }

    pub fn mouse_moved(&self) -> bool {
        self.mouse_moved
    }

    pub fn mouse_pos(&self) -> &Vec2 {
        &self.mouse_pos
    }

    pub fn mouse_wheeled(&self) -> bool {
        self.mouse_wheeled
    }

    pub fn mouse_wheel_delta(&self) -> MouseScrollDelta {
        self.mouse_wheel
    }
}
