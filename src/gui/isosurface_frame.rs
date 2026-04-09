use crate::uniform::Interaction;
use crate::Cube;
use egui_double_slider::DoubleSlider;
use std::collections::HashMap;
use crate::uniform::SurfaceRenderParams;
use std::mem::offset_of;

use crate::Scene;

pub struct IsosurfaceFrame {
    // isosurface value
    isosurface: f32,
    // a diffuse color to show the isosurface with
    diffuse_color: [f32; 4],
    // min cut precomputed corresponding to the first 1% of data
    pub min_cut_default: f32,
    // max cut precomputed corresponding to the last 99% of data
    pub max_cut_default: f32,
    
    pub open: bool,
}

impl IsosurfaceFrame {
    pub fn new() -> Self {
        Self {
            isosurface: 0.0,
            diffuse_color: [0.0, 1.0, 0.0, 1.0],
            min_cut_default: 0.0,
            max_cut_default: 1.0,
            open: false,
        }
    }

    pub fn set_default_cuts(&mut self, min_cut_default: f32, max_cut_default: f32) {
        self.min_cut_default = min_cut_default as f32;
        self.max_cut_default = max_cut_default as f32;
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
    ) {
        if ui
            .add(egui::Button::new("Isosurface").selected(self.open))
            .on_hover_text("Isosurface rendering mode")
            .clicked()
        {
            self.open = !self.open;
            // render with the new mode enabled
            *needs_redraw = true;
        }

        if self.open {
            egui::Window::new("Isosurface options")
                .resizable(false)
                .default_pos(egui::pos2(10.0, 50.0))
                .default_width(150.0)
                .show(ui.ctx(), |ui| {
                    let mut diffuse_color = self.diffuse_color;
                    let mut isosurface = self.isosurface;

                    let min_cut_default = self.min_cut_default;
                    let max_cut_default = self.max_cut_default;

                    let old_render_params = (
                        isosurface,
                        diffuse_color,
                    );
                    
                    // Isosurface scope
                    ui.add_enabled_ui(self.open, |ui| {
                        ui.add(
                            egui::Slider::new(
                                &mut isosurface,
                                min_cut_default..=max_cut_default,
                            )
                            .text("Iso-value"),
                        );
                        ui.label("Diffuse color");
                        ui.color_edit_button_rgba_unmultiplied(&mut diffuse_color);
                    });

                    if old_render_params != (isosurface, diffuse_color) {
                        queue.write_buffer(
                            &buffers["surface_render_params"],
                            0,
                            bytemuck::bytes_of(&SurfaceRenderParams {
                                diffuse_color,
                                iso: isosurface,
                                _pad: [0.0; 3]
                            }),
                        );

                        *needs_redraw = true;
                    }

                    self.diffuse_color = diffuse_color;
                    self.isosurface = isosurface;
                });
        }
    }
}
