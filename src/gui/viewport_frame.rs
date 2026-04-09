use crate::uniform::Interaction;
use std::collections::HashMap;
use std::mem::offset_of;

use crate::Scene;

pub struct ViewportFrame {
    // perspective rendering mode
    pub perspective: bool,
    pub theta: f32,
    pub delta: f32,

    pub open: bool,
}

impl ViewportFrame {
    pub fn new() -> Self {
        Self {
            open: false,
            perspective: false,
            delta: 0.0,
            theta: std::f32::consts::PI,
        }
    }

    pub fn set_camera_position(&mut self, theta: f64, delta: f64) {
        self.theta = theta as f32;
        self.delta = delta as f32;
    }

    pub fn set_perspective(&mut self, perspective: bool) {
        self.perspective = perspective;
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
            .add(egui::Button::new("🎥 View").selected(self.open))
            .on_hover_text("Viewport tool")
            .clicked()
        {
            self.open = !self.open;
        }

        if self.open {
            egui::Window::new("Viewport options")
                .resizable(false)
                .default_pos(egui::pos2(10.0, 50.0))
                .default_width(150.0)
                .show(ui.ctx(), |ui| {
                    let mut perspective = self.perspective;
                    let mut theta = self.theta;
                    let mut delta = self.delta;

                    let old_scene_settings = (theta, delta, perspective);
                    
                    // Viewport scope
                    ui.checkbox(&mut perspective, "Perspective");

                    if ui.button("RA Dec (Front)").clicked() {
                        (theta, delta) = (std::f32::consts::PI, 0.0);
                    }

                    if ui.button("-RA Dec (Back)").clicked() {
                        (theta, delta) = (0.0, 0.0);
                    }

                    if ui.button("-V Dec (Left)").clicked() {
                        (theta, delta) = (-std::f32::consts::PI / 2.0, 0.0);
                    }

                    if ui.button("V Dec (Right)").clicked() {
                        (theta, delta) = (std::f32::consts::PI / 2.0, 0.0);
                    }

                    if ui.button("RA V (Top)").clicked() {
                        (theta, delta) =
                            (std::f32::consts::PI, std::f32::consts::PI * 0.5 - 1e-3);
                    }

                    if ui.button("RA -V (Bottom)").clicked() {
                        (theta, delta) =
                            (std::f32::consts::PI, -std::f32::consts::PI * 0.5 + 1e-3);
                    }

                    if old_scene_settings != (theta, delta, perspective) {
                        queue.write_buffer(
                            &buffers["scene"],
                            std::mem::offset_of!(Scene, origin) as wgpu::BufferAddress,
                            bytemuck::bytes_of(&[
                                theta,
                                delta,
                                0.0,
                                0.0,
                                if perspective { 1.0_f32 } else { 0.0_f32 },
                            ]),
                        );

                        *needs_redraw = true;
                    }

                    self.perspective = perspective;
                    self.theta = theta;
                    self.delta = delta;
                });
        }
    }
}
