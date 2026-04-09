use crate::input::Input;
use crate::State;

use winit::window::Window;

mod cutout_tuning;
mod dragging;
mod spectra_extraction;
mod zooming;

use cutout_tuning::CutoutTuning;
use dragging::Dragging;
use spectra_extraction::SpectraExtraction;
use zooming::Zooming;

pub(crate) trait FiniteStateMachine {
    fn verify_transitions(
        self,
        state: &mut State,
        input: &Input,
        pointer_over_egui: bool,
        consumed_by_ui: bool,
        window: &Window,
    ) -> Self;
    fn on_start(&self) -> bool;
}

#[derive(Clone)]
pub(crate) struct Mode {
    dragging_mode: Dragging,
    spectra_extraction_mode: SpectraExtraction,
    cutout_tuning_mode: CutoutTuning,
    zooming_mode: Zooming,
}

impl Mode {
    pub(crate) fn new(input: &Input) -> Self {
        let dragging_mode = Dragging::new(input);
        let spectra_extraction_mode = SpectraExtraction::new(input);
        let cutout_tuning_mode = CutoutTuning::new(input);
        let zooming_mode = Zooming::new(input);

        Self {
            dragging_mode,
            spectra_extraction_mode,
            cutout_tuning_mode,
            zooming_mode,
        }
    }
}

impl FiniteStateMachine for Mode {
    fn on_start(&self) -> bool {
        self.dragging_mode.on_start()
    }

    fn verify_transitions(
        mut self,
        state: &mut State,
        input: &Input,
        pointer_over_egui: bool,
        consumed_by_ui: bool,
        window: &Window,
    ) -> Self {
        // If the spectra mode is on
        if !self.spectra_extraction_mode.on_start() {
            self.spectra_extraction_mode = self.spectra_extraction_mode.clone().verify_transitions(
                state,
                input,
                pointer_over_egui,
                consumed_by_ui,
                window,
            );

            // no conflicts with the cutout tuning mode
            self.cutout_tuning_mode = self.cutout_tuning_mode.clone().verify_transitions(
                state,
                input,
                pointer_over_egui,
                consumed_by_ui,
                window,
            );
        // If the cutout tuning mode is on
        } else if !self.cutout_tuning_mode.on_start() {
            self.cutout_tuning_mode = self.cutout_tuning_mode.clone().verify_transitions(
                state,
                input,
                pointer_over_egui,
                consumed_by_ui,
                window,
            );
        // Otherwise, normal behavior
        } else {
            // priority to the normal mode
            self.dragging_mode = self.dragging_mode.clone().verify_transitions(
                state,
                input,
                pointer_over_egui,
                consumed_by_ui,
                window,
            );

            if self.dragging_mode.on_start() {
                self.zooming_mode = self.zooming_mode.clone().verify_transitions(
                    state,
                    input,
                    pointer_over_egui,
                    consumed_by_ui,
                    window,
                );
            }

            // if the mode has not been consumed check for the other modes
            if self.dragging_mode.on_start() {
                self.spectra_extraction_mode = self
                    .spectra_extraction_mode
                    .clone()
                    .verify_transitions(state, input, pointer_over_egui, consumed_by_ui, window);

                if self.spectra_extraction_mode.on_start() {
                    self.cutout_tuning_mode = self.cutout_tuning_mode.clone().verify_transitions(
                        state,
                        input,
                        pointer_over_egui,
                        consumed_by_ui,
                        window,
                    );
                }
            }
        }

        self
    }
}
