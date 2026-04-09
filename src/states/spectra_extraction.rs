use crate::states::FiniteStateMachine;
use crate::Input;
use crate::State;
use winit::event::MouseButton;
use winit::event::MouseScrollDelta;
use winit::window::Window;

#[derive(Clone)]
pub(crate) enum SpectraExtraction {
    Init,
    Pointing { radius: f32 },
}

impl SpectraExtraction {
    pub(crate) fn new(_input: &Input) -> Self {
        Self::Init
    }
}

impl FiniteStateMachine for SpectraExtraction {
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
        let egui_wants_pointer_input = state.egui_renderer.context().wants_pointer_input();

        match self {
            Self::Init if state.show_spectra_window => {
                // Enter spectra extraction mode
                Self::Pointing { radius: 10.0 }
            }

            Self::Pointing { .. } if !state.show_spectra_window => Self::Init,

            Self::Pointing { mut radius, .. }
                if !pointer_over_egui && !egui_wants_pointer_input =>
            {
                state.set_cursor(egui::CursorIcon::Crosshair);

                if input.mouse_wheeled() {
                    let delta_y = match input.mouse_wheel_delta() {
                        MouseScrollDelta::PixelDelta(delta) => delta.y as f32,
                        MouseScrollDelta::LineDelta(_, y) => y,
                    };

                    radius = (radius - delta_y).max(1.0);
                }

                let cursor_pos = *input.mouse_pos();
                // todo draw the circle around the mouse
                let ctx = state.egui_renderer.context();
                let pointer_pos = ctx.pointer_hover_pos();
                let painter = ctx.layer_painter(egui::LayerId::new(
                    egui::Order::Foreground,
                    egui::Id::new("cursor_circle"),
                ));

                if let Some(pointer_pos) = pointer_pos {
                    painter.circle_stroke(
                        pointer_pos,
                        radius,
                        egui::Stroke::new(3.0, egui::Color32::RED),
                    );
                }

                if input.mouse_down(MouseButton::Left) {
                    if let Some(pointer_pos) = pointer_pos {
                        painter.circle_filled(
                            pointer_pos,
                            radius,
                            egui::Color32::from_rgba_premultiplied(255, 0, 0, 100),
                        );
                    }

                    state.spectra_data = state
                        .extract_spectra_at_screen_position(
                            cursor_pos.x as u32,
                            state.size.height - cursor_pos.y as u32,
                        )
                        .unwrap_or(vec![])
                        .into_iter()
                        .enumerate()
                        .map(|(i, v)| [i as f64, v as f64])
                        .collect();
                }

                Self::Pointing { radius }
            }

            _ => self,
        }
    }
}
