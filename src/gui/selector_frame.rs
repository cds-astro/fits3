use crate::uniform::Interaction;
use crate::Cube;
use egui_double_slider::DoubleSlider;
use std::collections::HashMap;
use std::mem::offset_of;

#[cfg(target_arch = "wasm32")]
use crate::ONSELECT;
#[cfg(target_arch = "wasm32")]
use fitsrs::ImgXY;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsValue;

pub struct SelectorFrame {
    pub f1: f32,
    pub f2: f32,
    pub fmin: f32,
    pub fmax: f32,
    pub fov: f32,
    pub fov_min: f32,
    pub fov_max: f32,
    pub ra: f32,
    pub ra_min: f32,
    pub ra_max: f32,
    pub dec: f32,
    pub dec_min: f32,
    pub dec_max: f32,
    pub show_unique_slice: bool,
    pub slice_idx: u32,
    pub open: bool,
}

impl SelectorFrame {
    pub fn new() -> Self {
        Self {
            open: false,
            f1: 0.0,
            f2: 100.0,
            fmin: 0.0,
            fmax: 100.0,
            fov: 100.0,
            fov_min: 0.0,
            fov_max: 100.0,
            ra: 50.0,
            ra_min: 0.0,
            ra_max: 100.0,
            dec: 50.0,
            dec_min: 0.0,
            dec_max: 100.0,
            slice_idx: 0,
            show_unique_slice: false,
        }
    }

    pub fn reset(&mut self, cube: &Cube) {
        self.fov = cube.size.0 as f32;
        self.fov_min = 0.0;
        self.fov_max = cube.size.0 as f32;

        self.ra = (cube.size.0 as f32) * 0.5;
        self.ra_min = 0.0;
        self.ra_max = cube.size.0 as f32;

        self.dec = (cube.size.1 as f32) * 0.5;
        self.dec_min = 0.0;
        self.dec_max = cube.size.1 as f32;

        self.f1 = 0.0;
        self.f2 = cube.size.2 as f32;
        self.fmin = 0.0;
        self.fmax = cube.size.2 as f32;

        self.slice_idx = 0;
    }

    pub fn close(&mut self) {
        self.open = false;
    }

    pub fn render(
        &mut self,
        ui: &mut egui::Ui,
        queue: &wgpu::Queue,
        buffers: &HashMap<&'static str, wgpu::Buffer>,
        cube: &Cube,
        needs_redraw: &mut bool,
    ) {
        if ui
            .add(egui::Button::new("⛶ Select").selected(self.open))
            .on_hover_text("Selection tool")
            .clicked()
        {
            self.open = !self.open;
        }

        if self.open {
            egui::Window::new("Selection settings")
                .resizable(false)
                .default_pos(egui::pos2(10.0, 50.0))
                .default_width(150.0)
                .show(ui.ctx(), |ui| {
                    let mut f1 = self.f1;
                    let mut f2 = self.f2;
                    let mut fov = self.fov;
                    let mut ra = self.ra;
                    let mut dec = self.dec;
                    let mut ra_min = self.ra_min;
                    let mut ra_max = self.ra_max;
                    let mut dec_min = self.dec_min;
                    let mut dec_max = self.dec_max;
                    let mut fmin = self.fmin;
                    let mut fmax = self.fmax;
                    let mut fov_min = self.fov_min;
                    let mut fov_max = self.fov_max;
                    let mut show_unique_slice = self.show_unique_slice;
                    let mut slice_idx = self.slice_idx;

                    let naxis = &cube.size;

                    let old_bbox_settings = [
                        ra,
                        dec,
                        fov,
                        f1,
                        f2,
                        ra_min,
                        ra_max,
                        dec_min,
                        dec_max,
                        fov_min,
                        fov_max,
                        fmin,
                        fmax,
                        slice_idx as f32,
                    ];
                    ui.checkbox(&mut show_unique_slice, "Slice selector");
                    ui.add_enabled_ui(show_unique_slice, |ui| {
                        ui.add(
                            egui::Slider::new(&mut slice_idx, (fmin as u32)..=(fmax as u32))
                                .text("slice idx"),
                        );
                    });

                    ui.separator();

                    ui.add_enabled_ui(!show_unique_slice, |ui| {
                        ui.label("Select a frequency range");
                        ui.horizontal(|ui| {
                            ui.add(egui::DragValue::new(&mut f1).speed(1.0));
                            ui.add(
                                DoubleSlider::new(&mut f1, &mut f2, fmin..=fmax).scroll_factor(1.0),
                            );
                            ui.add(egui::DragValue::new(&mut f2).speed(1.0));
                        });

                        ui.add(egui::Slider::new(&mut fov, fov_min..=fov_max).text("Select FoV"));
                        ui.add(egui::Slider::new(&mut ra, ra_min..=ra_max).text("Select RA"));
                        ui.add(egui::Slider::new(&mut dec, dec_min..=dec_max).text("Select Dec"));

                        // f1, f2, fov, ra, dec
                        ui.horizontal(|ui| {
                            if ui.button("Select").clicked() {
                                let l = [
                                    (ra - fov * 0.5) / (naxis.0 as f32),
                                    (dec - fov * 0.5) / (naxis.1 as f32),
                                    f1 / (naxis.2 as f32),
                                ];
                                let h = [
                                    (ra + fov * 0.5) / (naxis.0 as f32),
                                    (dec + fov * 0.5) / (naxis.1 as f32),
                                    f2 / (naxis.2 as f32),
                                ];

                                queue.write_buffer(
                                    &buffers["interaction"],
                                    0,
                                    bytemuck::bytes_of(&[
                                        l[0], l[1], l[2], 0.0, h[0], h[1], h[2], 0.0,
                                    ]),
                                );

                                // set the new select limits
                                ra_min = ra - fov * 0.5;
                                ra_max = ra + fov * 0.5;
                                dec_min = dec - fov * 0.5;
                                dec_max = dec + fov * 0.5;
                                fmin = f1;
                                fmax = f2;
                                fov_min = 0.0;
                                fov_max = fov;

                                #[cfg(target_arch = "wasm32")]
                                {
                                    let x_px = ra as f64;
                                    let y_px = dec as f64;
                                    let w_px = fov as f64;

                                    let p = cube.wcs.unproj(&ImgXY::new(x_px, y_px)).unwrap();

                                    let fov = cube.wcs.field_of_view().0
                                        * ((w_px as f64) / (naxis.0 as f64));

                                    let f1 = f1 / (naxis.2 as f32);
                                    let f2 = f2 / (naxis.2 as f32);

                                    ONSELECT.with(|f| {
                                        if let Some(cb) = &*f.borrow() {
                                            use js_sys::Array;
                                            let ra = p.lon().to_degrees();
                                            let dec = p.lat().to_degrees();

                                            let args = Array::new();
                                            args.push(&JsValue::from_f64(ra));
                                            args.push(&JsValue::from_f64(dec));
                                            args.push(&JsValue::from_f64(fov));
                                            args.push(&JsValue::from_f64(f1 as f64));
                                            args.push(&JsValue::from_f64(f2 as f64));
                                            cb.apply(&JsValue::NULL, &args).unwrap();
                                        }
                                    });
                                }
                            }

                            if ui.button("Reset").clicked() {
                                fov = naxis.0 as f32;
                                ra = (naxis.0 as f32) * 0.5;
                                dec = (naxis.1 as f32) * 0.5;
                                f1 = 0.0;
                                f2 = naxis.2 as f32;

                                ra_min = 0.0;
                                ra_max = naxis.0 as f32;
                                dec_min = 0.0;
                                dec_max = naxis.1 as f32;
                                fmin = 0.0;
                                fmax = naxis.2 as f32;
                                fov_min = 0.0;
                                fov_max = naxis.0 as f32;

                                queue.write_buffer(
                                    &buffers["interaction"],
                                    0,
                                    bytemuck::bytes_of(&[
                                        0.0_f32, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0,
                                    ]),
                                );
                            }
                        });
                    });

                    if old_bbox_settings
                        != [
                            ra,
                            dec,
                            fov,
                            f1,
                            f2,
                            ra_min,
                            ra_max,
                            dec_min,
                            dec_max,
                            fov_min,
                            fov_max,
                            fmin,
                            fmax,
                            slice_idx as f32,
                        ]
                        || show_unique_slice != self.show_unique_slice
                    {
                        let (l, h) = if show_unique_slice {
                            let l = [
                                (ra - fov * 0.5 - ra_min) / (ra_max - ra_min) - 0.5,
                                (dec - fov * 0.5 - dec_min) / (dec_max - dec_min) - 0.5,
                                (slice_idx as f32 - fmin) / (fmax - fmin) - 0.5,
                            ];
                            let h = [
                                (ra + fov * 0.5 - ra_min) / (ra_max - ra_min) - 0.5,
                                (dec + fov * 0.5 - dec_min) / (dec_max - dec_min) - 0.5,
                                (slice_idx as f32 + 1.0 - fmin) / (fmax - fmin) - 0.5,
                            ];

                            (l, h)
                        } else {
                            let l = [
                                (ra - fov * 0.5 - ra_min) / (ra_max - ra_min) - 0.5,
                                (dec - fov * 0.5 - dec_min) / (dec_max - dec_min) - 0.5,
                                (f1 - fmin) / (fmax - fmin) - 0.5,
                            ];
                            let h = [
                                (ra + fov * 0.5 - ra_min) / (ra_max - ra_min) - 0.5,
                                (dec + fov * 0.5 - dec_min) / (dec_max - dec_min) - 0.5,
                                (f2 - fmin) / (fmax - fmin) - 0.5,
                            ];

                            (l, h)
                        };

                        queue.write_buffer(
                            &buffers["interaction"],
                            offset_of!(Interaction, bbox_min) as wgpu::BufferAddress,
                            bytemuck::bytes_of(&[l[0], l[1], l[2], 0.0, h[0], h[1], h[2], 0.0]),
                        );

                        *needs_redraw = true;
                    }

                    self.f1 = f1;
                    self.f2 = f2;
                    self.fov = fov;
                    self.ra = ra;
                    self.dec = dec;
                    self.ra_min = ra_min;
                    self.ra_max = ra_max;
                    self.dec_min = dec_min;
                    self.dec_max = dec_max;
                    self.fmin = fmin;
                    self.fmax = fmax;
                    self.fov_min = fov_min;
                    self.fov_max = fov_max;
                    self.show_unique_slice = show_unique_slice;
                    self.slice_idx = slice_idx;
                });
        }
    }
}
