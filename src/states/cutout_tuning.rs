use glam::Vec2;
use winit::event::MouseButton;
use winit::window::Window;

use crate::states::FiniteStateMachine;
use crate::Input;
use crate::State;

#[derive(Clone)]
pub(crate) enum CutoutTuning {
    Init,
    Dragging {
        start_cursor_pos: Vec2,
        start_min_cut: f32,
        start_max_cut: f32,
    },
}

impl CutoutTuning {
    pub(crate) fn new(_input: &Input) -> Self {
        Self::Init
    }
}

impl FiniteStateMachine for CutoutTuning {
    fn on_start(&self) -> bool {
        matches!(self, Self::Init)
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
            Self::Init if !pointer_over_egui && input.mouse_pressed(MouseButton::Right) => {
                state.set_cursor(egui::CursorIcon::PointingHand);

                let start_cursor_pos = *input.mouse_pos();
                let start_min_cut = state.min_cut;
                let start_max_cut = state.max_cut;

                Self::Dragging {
                    start_cursor_pos,
                    start_min_cut,
                    start_max_cut,
                }
            }

            Self::Dragging {
                start_cursor_pos,
                start_min_cut,
                start_max_cut,
            } if input.mouse_moved() => {
                state.set_cursor(egui::CursorIcon::PointingHand);
                let cursor_pos = *input.mouse_pos();
                let dx = (cursor_pos.x - start_cursor_pos.x) / ((state.size.width as f32) * 0.5);
                let dy = (cursor_pos.y - start_cursor_pos.y) / ((state.size.height as f32) * 0.5);

                // between -1 and 1

                let l = state.max_cut_default - state.min_cut_default;
                state.min_cut = start_min_cut + dx * l + dy * l;
                state.max_cut = start_max_cut + dx * l - dy * l;

                state.queue.write_buffer(
                    &state.buffers["volumetric_render_params"],
                    0,
                    bytemuck::bytes_of(&[state.min_cut, state.max_cut]),
                );

                state.needs_redraw_view = true;

                Self::Dragging {
                    start_cursor_pos,
                    start_min_cut,
                    start_max_cut,
                }
            }

            Self::Dragging {
                start_cursor_pos, ..
            } if !input.mouse_down(MouseButton::Right) => {
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
