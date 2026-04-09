use crate::uniform::Interaction;
use crate::Cube;
use egui_double_slider::DoubleSlider;
use std::collections::HashMap;
use crate::uniform::SurfaceRenderParams;
use crate::uniform::VolumetricRenderParams;
use std::mem::offset_of;

use crate::Scene;

#[repr(i32)]
#[derive(PartialEq, Copy, Clone, Debug)]
enum Colormap {
    Turbo,
    Viridis,
    Inferno,
    Plasma,
    Rainbow,
    Cubehelix,
}

#[repr(i32)]
#[derive(PartialEq, Copy, Clone, Debug)]
enum TransferFunc {
    Linear,
    Sqrt,
    Pow2,
    Asinh,
    Log,
}

pub struct SettingsFrame {
    pub open: bool,

    colormap: Colormap,
    transfer: TransferFunc,

    // a background color
    bg_color: [f32; 3],
}

impl SettingsFrame {
    pub fn new() -> Self {
        Self {
            open: false,
            bg_color: [0.0_f32; 3],
            colormap: Colormap::Turbo,
            transfer: TransferFunc::Linear,
        }
    }

    pub fn close(&mut self) {
        self.open = false;
    }

    pub fn render(
        &mut self,
        ui: &mut egui::Ui,
        queue: &wgpu::Queue,
        buffers: &HashMap<&'static str, wgpu::Buffer>,
        needs_redraw: &mut bool,
        min_cut: &mut f32,
        max_cut: &mut f32,
        min_cut_default: f32,
        max_cut_default: f32
    ) {
        if ui
            .add(egui::Button::new("⚙ Settings").selected(self.open))
            .on_hover_text("Show options (Ctrl+O)")
            .clicked()
        {
            self.open = !self.open;
            // render with the new mode enabled
            *needs_redraw = true;
        }

        if self.open {
            egui::Window::new("Settings")
                .resizable(false)
                .default_pos(egui::pos2(10.0, 50.0))
                .default_width(150.0)
                .show(ui.ctx(), |ui| {
                    let mut bg_color = self.bg_color;
                    //let mut min_cut = self.min_cut;
                    //let mut max_cut = self.max_cut;
                    let mut colormap = self.colormap;
                    let mut transfer = self.transfer;

                    let data_length = (max_cut_default - min_cut_default).abs();
                    let datamin = min_cut_default - data_length;
                    let datamax = max_cut_default + 5.0 * data_length;

                    let old_render_params = (
                        *min_cut,
                        *max_cut,
                        colormap,
                        transfer,
                        bg_color,
                    );

                    ui.label("Cutout parameters");
                    ui.add(
                        DoubleSlider::new(
                            min_cut,
                            max_cut,
                            datamin..=datamax,
                        )
                        .scroll_factor((datamax - datamin) / 100.0)
                        .separation_distance((datamax - datamin) / 100.0),
                    );

                    ui.horizontal(|ui| {
                        ui.add(
                            egui::Slider::new(min_cut, datamin..=datamax)
                                .text("Min cut"),
                        );
                    });
                    ui.horizontal(|ui| {
                        ui.add(
                            egui::Slider::new(max_cut, datamin..=datamax)
                                .text("Max cut"),
                        );
                    });
                    if ui.button("Reset cuts").clicked() {
                        *min_cut = min_cut_default;
                        *max_cut = max_cut_default;
                    }

                    egui::ComboBox::from_label("Select colormap")
                        .selected_text(format!("{:?}", colormap))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut colormap,
                                Colormap::Turbo,
                                "Turbo",
                            );
                            ui.selectable_value(
                                &mut colormap,
                                Colormap::Viridis,
                                "Viridis",
                            );
                            ui.selectable_value(
                                &mut colormap,
                                Colormap::Inferno,
                                "Inferno",
                            );
                            ui.selectable_value(
                                &mut colormap,
                                Colormap::Plasma,
                                "Plasma",
                            );
                            ui.selectable_value(
                                &mut colormap,
                                Colormap::Rainbow,
                                "Rainbow",
                            );
                            ui.selectable_value(
                                &mut colormap,
                                Colormap::Cubehelix,
                                "Cubehelix",
                            );
                        });

                    egui::ComboBox::from_label("Select a transfer function")
                        .selected_text(format!("{:?}", transfer))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut transfer,
                                TransferFunc::Linear,
                                "Linear",
                            );
                            ui.selectable_value(
                                &mut transfer,
                                TransferFunc::Sqrt,
                                "Sqrt",
                            );
                            ui.selectable_value(
                                &mut transfer,
                                TransferFunc::Pow2,
                                "Pow2",
                            );
                            ui.selectable_value(
                                &mut transfer,
                                TransferFunc::Asinh,
                                "Asinh",
                            );
                            ui.selectable_value(
                                &mut transfer,
                                TransferFunc::Log,
                                "Log",
                            );
                        });

                    ui.label("Background color");
                    ui.color_edit_button_rgb(&mut bg_color);

                    if old_render_params
                        != (
                            *min_cut,
                            *max_cut,
                            colormap,
                            transfer,
                            bg_color,
                        )
                    {
                        queue.write_buffer(
                            &buffers["volumetric_render_params"],
                            0,
                            bytemuck::bytes_of(&VolumetricRenderParams {
                                cut: [*min_cut, *max_cut],
                                colormap: colormap as i32,
                                transfer: transfer as i32,
                                bg_color,
                                _pad: 0.0,
                            }),
                        );

                        *needs_redraw = true;
                    }

                    self.bg_color = bg_color;
                    self.colormap = colormap;
                    self.transfer = transfer;
                    //self.min_cut = min_cut;
                    //self.max_cut = max_cut;
                });
        }
    }
}
