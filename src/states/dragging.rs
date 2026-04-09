use glam::Vec2;
use winit::event::MouseButton;
use winit::window::Window;

use crate::states::FiniteStateMachine;
use crate::Input;
use crate::State;

#[derive(Clone)]
pub(crate) enum Dragging {
    Init,
    Dragging { start_cursor_pos: Vec2 },
}

impl Dragging {
    pub(crate) fn new(_input: &Input) -> Self {
        Self::Init
    }
}

impl FiniteStateMachine for Dragging {
    fn on_start(&self) -> bool {
        matches!(self, Self::Init { .. })
    }

    fn verify_transitions(
        self,
        state: &mut State,
        input: &Input,
        pointer_over_egui: bool,
        _consumed_by_ui: bool,
        window: &Window,
    ) -> Self {
        match self {
            Self::Init if !pointer_over_egui && input.mouse_moved() => {
                state.set_cursor(egui::CursorIcon::Grab);

                Self::Init
            }

            Self::Init if !pointer_over_egui && input.mouse_pressed(MouseButton::Left) => {
                state.set_cursor(egui::CursorIcon::Grabbing);

                let cursor_pos = *input.mouse_pos();

                Self::Dragging {
                    start_cursor_pos: cursor_pos,
                }
            }

            Self::Dragging { start_cursor_pos } if input.mouse_moved() => {
                state.set_cursor(egui::CursorIcon::Grabbing);

                let cursor_pos = input.mouse_pos();
                let dx = (cursor_pos.x - start_cursor_pos.x) / ((state.size.width as f32) * 0.5);
                let dy = (cursor_pos.y - start_cursor_pos.y) / ((state.size.height as f32) * 0.5);

                let dtheta = 2.0 * dx as f64;
                let ddelta = dy as f64;

                let delta = (state.delta + ddelta).clamp(
                    -std::f64::consts::PI * 0.5 + 1e-3,
                    std::f64::consts::PI * 0.5 - 1e-3,
                );
                let theta = state.theta + dtheta;

                state.set_camera_position(theta, delta, window);

                state.needs_redraw_view = true;

                Self::Dragging {
                    start_cursor_pos: *cursor_pos,
                }
            }

            Self::Dragging { start_cursor_pos } if !input.mouse_down(MouseButton::Left) => {
                if pointer_over_egui {
                    state.set_cursor(egui::CursorIcon::Default);
                } else {
                    state.set_cursor(egui::CursorIcon::Grab);
                }

                Self::Init
            }

            _ => self,
        }
    }
}
