use crate::Scene;
use std::mem::offset_of;
use winit::window::Window;

use crate::states::FiniteStateMachine;
use crate::Clock;
use crate::Input;
use crate::State;
use winit::event::MouseScrollDelta;

#[derive(Clone)]
pub(crate) enum Zooming {
    Init {
        last_zoom_time: f32,
        clock: Clock,
    },
}

impl Zooming {
    pub(crate) fn new(_input: &Input) -> Self {
        Self::Init {
            last_zoom_time: 0.0,
            clock: Clock::now(),
        }
    }
}

impl FiniteStateMachine for Zooming {
    fn on_start(&self) -> bool {
        matches!(self, Self::Init { .. })
    }

    fn verify_transitions(
        self,
        state: &mut State,
        input: &Input,
        pointer_over_egui: bool,
        _consumed_by_ui: bool,
        _window: &Window,
    ) -> Self {
        match self {
            Self::Init {
                clock,
                ..
            } if !pointer_over_egui && input.mouse_wheeled() => {
                let delta_y = match input.mouse_wheel_delta() {
                    MouseScrollDelta::PixelDelta(delta) => delta.y as f32,
                    MouseScrollDelta::LineDelta(_, y) => y,
                };

                if delta_y > 0.0 {
                    state.set_cursor(egui::CursorIcon::ZoomIn);
                } else if delta_y < 0.0 {
                    state.set_cursor(egui::CursorIcon::ZoomOut);
                } else if pointer_over_egui {
                    state.set_cursor(egui::CursorIcon::Default);
                } else {
                    state.set_cursor(egui::CursorIcon::Grab);
                }

                let mut zoom_factor = state.zoom_factor;

                zoom_factor = (zoom_factor * (1.0 - delta_y * 1e-2)).clamp(0.01, 1.0);

                let last_zoom_time = clock.elapsed_as_secs();

                state.set_zoom_factor(zoom_factor);

                Self::Init {
                    last_zoom_time,
                    clock,
                }
            }

            Self::Init {
                last_zoom_time,
                clock,
            } if !input.mouse_wheeled() && (clock.elapsed_as_secs() - last_zoom_time) > 0.5 => {
                if pointer_over_egui {
                    state.set_cursor(egui::CursorIcon::Default);
                } else {
                    state.set_cursor(egui::CursorIcon::Grab);
                }

                Self::Init {
                    last_zoom_time,
                    clock,
                }
            }

            _ => self,
        }
    }
}
