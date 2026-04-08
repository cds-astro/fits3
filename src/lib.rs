extern crate byte_slice_cast;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use winit::event_loop::ControlFlow;
#[cfg(target_arch = "wasm32")]
extern crate console_error_panic_hook;

use std::mem::offset_of;
use std::iter;
use std::convert::TryInto;
use egui_double_slider::DoubleSlider;
use std::path::PathBuf;
#[cfg(not(target_arch = "wasm32"))]
use egui_file_dialog::FileDialog;

use egui::Color32;

#[repr(i32)]
#[derive(PartialEq, Copy, Clone)]
#[derive(Debug)]
enum Colormap {
    Turbo,
    Viridis,
    Inferno,
    Plasma,
    Rainbow,
    Cubehelix,
}

#[repr(i32)]
#[derive(PartialEq, Copy, Clone)]
#[derive(Debug)]
enum TransferFunc {
    Linear,
    Sqrt,
    Pow2,
    Asinh,
    Log,
}

use std::f32::consts::PI;
use uniform::{Scene, Volume, RenderParams, Interaction};

use winit::{
    application::ApplicationHandler,
    dpi::PhysicalPosition,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Fullscreen, Window, WindowId},
};
mod gui;
mod math;
mod texture;
mod time;
mod vertex;
mod volumetric;
mod selector;
mod moment;
mod open_file;
mod uniform;
mod user_event;
mod short_keys;
#[cfg(target_arch = "wasm32")]
mod js_api;
mod cube;

use fitsrs::card::Value;
use fitsrs::HDU;
use user_event::UserEvent;
use crate::math::Vec4;
use texture::Texture;
use time::Clock;
use vertex::{VertexNDC, Vertex};
use crate::selector::SelectorRenderer;
use crate::cube::Cube;

#[cfg(target_arch = "wasm32")]
use js_api::*;

use volumetric::VolumetricRenderer;

use fitsrs::Fits;
#[cfg(not(target_arch = "wasm32"))]
use memmap2::Mmap;
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
use std::io::Cursor;

use std::rc::Rc;
use std::cell::RefCell;

#[cfg(not(target_arch = "wasm32"))]
const CUBES_PATH: &[&'static str] = &[
    "./cubes/cutout-CDS_P_LGLBSHI16.fits",
    "./cubes/NGC_628_RO_CUBE_THINGS.FITS",
    "./cubes/cutout-CDS_P_LGLBSHI16.fits",
    "./cubes/cutout-CDS_C_GALFAHI.fits",
    "./cubes/NGC3198_cube.fits",
    "./cubes/NGC7331_cube.fits",
    "./cubes/CO_21_binned.fits",
    "./cubes/DHIGLS_DF_Tb.fits",
    "./cubes/DHIGLS_MG_Tb.fits",
    "./cubes/DHIGLS_PO_Tb.fits", //"./cubes/cosmo512-be.fits",
];

#[cfg(target_arch = "wasm32")]
use fitsrs::{ImgXY};

use std::collections::HashMap;
struct State {
    surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,

    viewport: ViewPort,
    
    is_surface_configured: bool,

    volumetric_renderer: VolumetricRenderer,
    selector_renderer: SelectorRenderer,

    // uniforms
    buffers: HashMap<&'static str, wgpu::Buffer>,
    clock: Clock,

    // NAXIS of the current loaded cube
    naxis: (u32, u32, u32),

    /// Cuts properties
    // min cut precomputed corresponding to the first 1% of data 
    min_cut_default: f32,
    // max cut precomputed corresponding to the last 99% of data
    max_cut_default: f32,
    // current min cut
    min_cut: f32,
    // current max cut
    max_cut: f32,

    // Selection
    f1: f32,
    f2: f32,
    fmin: f32,
    fmax: f32,
    fov: f32,
    fov_min: f32,
    fov_max: f32,
    ra: f32,
    ra_min: f32,
    ra_max: f32,
    dec: f32,
    dec_min: f32,
    dec_max: f32,

    #[cfg(not(target_arch = "wasm32"))]
    file_dialog: FileDialog,
    picked_file: Rc<RefCell<Option<String>>>,
    colormap: Colormap,
    transfer: TransferFunc,

    // isosurface value
    isosurface: f32,
    // a diffuse color to show the isosurface with
    diffuse_color: [f32; 4],
    // a background color
    bg_color: [f32; 3],
    // perspective rendering mode
    perspective: bool,
    // slice index
    slice_idx: u32,

    /// ui options
    show_isosurface: bool,
    show_options: bool,
    show_unique_slice: bool,
    show_shortkeys: bool,


    // The cube real data
    cube: Option<Cube>,

    // moment 0 state
    show_moment0_window: bool,
    moment0_texture: Option<egui::TextureHandle>,

    delta: f64,
    theta: f64,
    dtheta: f64,
    ddelta: f64,

    egui_renderer: gui::EguiRenderer, //egui: EguiRenderer,

    needs_redraw: bool,
}

#[derive(Debug, Clone, PartialEq)]
struct ViewPort {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
}

use crate::math::Mat4;
use crate::short_keys::ShortKeyCommands;


impl State {
    async fn new(
        window: &Window,
        instance: &wgpu::Instance,
        surface: wgpu::Surface<'static>,
    ) -> Self {
        let size = window.inner_size();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                // favor performane over the memory usage
                memory_hints: Default::default(),
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for the web, we'll have to disable some.
                required_limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits {
                        max_texture_dimension_3d: 512,
                        ..wgpu::Limits::downlevel_webgl2_defaults()
                    }
                } else {
                    wgpu::Limits::default()
                },
                label: None,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an sRGB surface texture. Using a different
        // one will result in all the colors coming out darker. If you want to support non
        // sRGB surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![surface_format.add_srgb_suffix()],
            desired_maximum_frame_latency: 2,
        };

        let buffers: HashMap<&'static str, wgpu::Buffer> = vec![
            ("scene", device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Scene uniform buffer"),
                size: std::mem::size_of::<Scene>() as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
            ("volume", device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Volume uniform buffer"),
                size: std::mem::size_of::<Volume>() as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
            ("render_params", device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Render params uniform buffer"),
                size: std::mem::size_of::<RenderParams>() as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
            ("interaction", device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Interaction uniform buffer"),
                size: std::mem::size_of::<Interaction>() as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
        ].into_iter().collect();

        // Uniform buffer
        queue.write_buffer(
            &buffers["scene"],
            0,
            bytemuck::bytes_of(&Scene {
                win_size: [1.0 as f32, 0.0, 0.0, 0.0],
                origin: [PI, 0.0, 0.0, 0.0],
                perspective: [0.0, 0.0, 0.0, 0.0],
            })
        );

        queue.write_buffer(
            &buffers["render_params"],
            0,
            bytemuck::bytes_of(&RenderParams {
                cut_iso: [1.0 as f32, 0.0, 0.0],
                colormap: 0,
                diffuse_color: [0.0 as f32, 1.0, 0.0, 1.0],
                transfer: 0,
                bg_color: [0.0_f32; 3],
            })
        );

        queue.write_buffer(
            &buffers["volume"],
            0,
            bytemuck::bytes_of(&Volume {
                cube_size: [1.0 as f32, 1.0, 1.0],
                _pad1: 0.0,
                block_size: [1.0 as f32, 1.0, 1.0],
                _pad2: 0.0,   // padding!
            })
        );

        queue.write_buffer(
            &buffers["interaction"],
            0,
            bytemuck::bytes_of(&Interaction {
                zoom_min: [0.0 as f32, 0.0, 0.0],
                _pad1: 0.0,
                zoom_max: [1.0, 1.0, 1.0],
                _pad2: 0.0,   // padding!
                bbox_min: [-0.5 as f32, -0.5, -0.5],
                _pad3: 0.0,   // padding!
                bbox_max: [0.5 as f32, 0.5, 0.5],
                _pad4: 0.0,   // padding!
            })
        );

        // Uniform buffer
        // set the initial cut values
        let clock = Clock::now();

        // Egui renderer init
        let egui_renderer = gui::EguiRenderer::new(&device, config.format, window);

        // Transfer local data for wasm
        let volumetric_renderer = VolumetricRenderer::new(&device, &queue, &config, &buffers);
        let selector_renderer = SelectorRenderer::new(&device, &config, &buffers);

        Self {
            surface,
            device,
            queue,
            config,
            size,

            is_surface_configured: false,

            // uniforms
            buffers,

            viewport: ViewPort {x: 0.0, y: 0.0, width: 1.0, height: 1.0},

            naxis: (1, 1, 1),

            min_cut_default: 0.0,
            max_cut_default: 1.0,
            min_cut: 0.0,
            max_cut: 1.0,

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

            perspective: false,
            isosurface: 0.0,
            slice_idx: 0,
            diffuse_color: [0.0, 1.0, 0.0, 1.0],
            bg_color: [0.0_f32; 3],
            show_isosurface: false,
            show_options: false,
            show_unique_slice: false,
            show_shortkeys: false,
            cube: None,

            delta: 0.0,
            theta: std::f64::consts::PI,
            dtheta: 0.0,
            ddelta: 0.0,
            #[cfg(not(target_arch = "wasm32"))]
            file_dialog: FileDialog::new()
                .add_file_filter_extensions("FITS cube", vec!["fits"])
                .default_file_filter("FITS cube"),
            picked_file: Rc::new(RefCell::new(None)),

            colormap: Colormap::Turbo,
            transfer: TransferFunc::Linear,

            clock,
            egui_renderer,
            volumetric_renderer,
            selector_renderer,

            // Moment0 window state
            show_moment0_window: false,
            moment0_texture: None,

            needs_redraw: true,
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn resize(&mut self, mut new_size: winit::dpi::PhysicalSize<u32>) {
        new_size.width = (new_size.width as f32 * 0.75_f32) as u32;
        new_size.height = (new_size.height as f32 * 0.75_f32) as u32;

        if new_size.width > 0 && new_size.height > 0 {
            new_size.width = new_size
                .width
                .min(wgpu::Limits::downlevel_webgl2_defaults().max_texture_dimension_2d);
            new_size.height = new_size
                .height
                .min(wgpu::Limits::downlevel_webgl2_defaults().max_texture_dimension_2d);

            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;

            self.queue.write_buffer(
                &self.buffers["scene"],
                0,
                bytemuck::bytes_of(&[self.size.width as f32, self.size.height as f32, 0.0, 0.0]),
            );

            self.needs_redraw = true;
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;

            self.queue.write_buffer(
                &self.buffers["scene"],
                0,
                bytemuck::bytes_of(&[self.size.width as f32, self.size.height as f32, 0.0, 0.0]),
            );

            self.needs_redraw = true;
        }
    }

    fn set_camera_position(&mut self, theta: f64, dtheta: f64, delta: f64, ddelta: f64, window: &Window) {
        self.theta = theta;
        self.dtheta = dtheta;
        self.delta = delta;
        self.ddelta = ddelta;

        self.queue.write_buffer(
            &self.buffers["scene"],
            offset_of!(Scene, origin) as wgpu::BufferAddress,
            bytemuck::bytes_of(&[self.theta as f32 + self.dtheta as f32, self.delta as f32 + self.ddelta as f32]),
        );
        
        window.request_redraw();
    }

    fn render(&mut self, window: &Window) -> Result<(), wgpu::SurfaceError> {
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        if !self.is_surface_configured {
            return Ok(());
        }

        if let Ok(frame) = self.surface.get_current_texture() {
            let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
                format: Some(self.config.format.add_srgb_suffix()),
                ..Default::default()
            });

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });


            self.volumetric_renderer.render_frame(&mut encoder, &view, self.show_isosurface, &self.viewport);
            self.selector_renderer.render_frame(&mut encoder, &view, &self.viewport);

            {
                self.egui_renderer.begin_frame(window);

                let mut isosurface = self.isosurface;
                let mut perspective = self.perspective;
                let mut diffuse_color = self.diffuse_color;
                let mut bg_color = self.bg_color;
                let mut show_isosurface = self.show_isosurface;
                let mut show_options = self.show_options;
                let mut show_shortkeys = self.show_shortkeys;
                let mut show_unique_slice = self.show_unique_slice;
                let mut min_cut = self.min_cut;
                let mut max_cut = self.max_cut;
                let min_cut_default = self.min_cut_default;
                let max_cut_default = self.max_cut_default;

                let mut theta = self.theta as f32;
                let mut delta = self.delta as f32;
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
                let mut show_moment0_window = self.show_moment0_window;

                let mut needs_redraw = self.needs_redraw;

                let picked_file = self.picked_file.clone();
                let mut file_picked = false;

                #[cfg(not(target_arch = "wasm32"))]
                let file_dialog = &mut self.file_dialog;

                let ctx = self.egui_renderer.context();
                egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.heading("FITS3");
                                                    
                        // File import
                        if ui.button("Load a file").clicked() {
                            // Open the file dialog to pick a file.
                            #[cfg(target_arch = "wasm32")]
                            open_file::open_file_dialog(picked_file.clone());

                            #[cfg(not(target_arch = "wasm32"))]
                            file_dialog.pick_file();
                        }

                        if picked_file.borrow().is_some() {
                            ui.label(format!("Picked file: {:?}", picked_file.borrow().as_ref().unwrap()));
                        }

                        ui.checkbox(&mut show_options, "Show options (Ctrl+O)")
                            .on_hover_text("Toggle with: Ctrl+O");

                        ui.checkbox(&mut show_shortkeys, "Show shortkeys (Ctrl+I)")
                            .on_hover_text("Toggle with: Ctrl+I");
                    });
                });

                #[cfg(not(target_arch = "wasm32"))]
                {
                    file_dialog.update(ctx);

                    if let Some(path) = file_dialog.take_picked() {
                        file_picked = true;
                        if path.extension().unwrap() == "fits" {
                            let file_name = path.to_path_buf().into_os_string().to_str().unwrap().to_string();
                            self.picked_file = Rc::new(RefCell::new(Some(file_name)));
                        }
                    }
                }
                
                let mut colormap = self.colormap;
                let mut transfer = self.transfer;

                let cube = self.cube.as_ref();
                let queue = &self.queue;

                let mut slice_idx = self.slice_idx;

                let data_length = (self.max_cut_default - self.min_cut_default).abs();
                let datamin = self.min_cut_default - data_length;
                let datamax = self.max_cut_default + 5.0*data_length;
                
                let buffers = &self.buffers;
                let moment0_texture = &mut self.moment0_texture;
                let naxis = &self.naxis;
                let mut old_bbox_settings = [ra, dec, fov, f1, f2, ra_min, ra_max, dec_min, dec_max, fov_min, fov_max, fmin, fmax, slice_idx as f32];
                let mut old_render_params = (show_isosurface, min_cut, max_cut, isosurface, colormap, transfer, diffuse_color, bg_color);
                let mut old_scene_settings = (theta, delta, perspective);
                if show_options {
                    egui::SidePanel::left("fits3 options")
                    .resizable(true)
                    .show(ctx, |ui| {
                        // Volumetric scope
                        ui.add_enabled_ui(!show_isosurface, |ui| {
                            ui.label("Cutout parameters");
                            ui.add(
                                DoubleSlider::new(&mut min_cut, &mut max_cut, datamin..=datamax)
                                    .scroll_factor((datamax - datamin) / 100.0)
                                    .separation_distance((datamax - datamin) / 100.0)
                            );

                            ui.horizontal(|ui| {
                                ui.add(egui::Slider::new(&mut min_cut, datamin..=datamax).text("Min cut"));
                            });
                            ui.horizontal(|ui| {
                                ui.add(egui::Slider::new(&mut max_cut, datamin..=datamax).text("Max cut"));
                            });
                            if ui.button("Reset cuts").clicked() {
                                min_cut = min_cut_default;
                                max_cut = max_cut_default;
                            }

                            egui::ComboBox::from_label("Select colormap")
                                .selected_text(format!("{:?}",colormap))
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut colormap, Colormap::Turbo, "Turbo");
                                    ui.selectable_value(&mut colormap, Colormap::Viridis, "Viridis");
                                    ui.selectable_value(&mut colormap, Colormap::Inferno, "Inferno");
                                    ui.selectable_value(&mut colormap, Colormap::Plasma, "Plasma");
                                    ui.selectable_value(&mut colormap, Colormap::Rainbow, "Rainbow");
                                    ui.selectable_value(&mut colormap, Colormap::Cubehelix, "Cubehelix");
                                });

                            egui::ComboBox::from_label("Select a transfer function")
                                .selected_text(format!("{:?}",transfer))
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut transfer, TransferFunc::Linear, "Linear");
                                    ui.selectable_value(&mut transfer, TransferFunc::Sqrt, "Sqrt");
                                    ui.selectable_value(&mut transfer, TransferFunc::Pow2, "Pow2");
                                    ui.selectable_value(&mut transfer, TransferFunc::Asinh, "Asinh");
                                    ui.selectable_value(&mut transfer, TransferFunc::Log, "Log");
                                });

                            ui.label("Background color");
                            ui.color_edit_button_rgb(&mut bg_color);
                        });

                        ui.separator();

                        ui.horizontal(|ui| {
                            ui.add(egui::Slider::new(&mut theta, -std::f32::consts::PI..=std::f32::consts::PI).text("theta"));
                        });
                        ui.horizontal(|ui| {
                            ui.add(egui::Slider::new(&mut delta, -std::f32::consts::PI..=std::f32::consts::PI).text("delta"));
                        });
                        
                        ui.separator();

                        // Isosurface scope
                        ui.checkbox(&mut show_isosurface, "Show isosurface");

                        ui.add_enabled_ui(show_isosurface, |ui| {
                            ui.add(egui::Slider::new(&mut isosurface, min_cut_default..=max_cut_default).text("Iso-value"));
                            ui.label("Diffuse color");
                            ui.color_edit_button_rgba_unmultiplied(&mut diffuse_color);
                        });

                        ui.separator();

                        // Viewport scope
                        ui.label("Viewport");
                        ui.checkbox(&mut perspective, "Perspective");

                        if ui.button("RA Dec (Front)").clicked() {
                            (theta, delta) = (std::f32::consts::PI, 0.0);
                        }

                        if ui.button("-RA Dec (Back)").clicked() {
                            (theta, delta) = (0.0, 0.0);
                        }

                        if ui.button("-V Dec (Left)").clicked() {
                            (theta, delta) = (-std::f32::consts::PI/2.0, 0.0);
                        }

                        if ui.button("V Dec (Right)").clicked() {
                            (theta, delta) = (std::f32::consts::PI/2.0, 0.0);
                        }

                        if ui.button("RA V (Top)").clicked() {
                            (theta, delta) = (std::f32::consts::PI, std::f32::consts::PI * 0.5 - 1e-3);
                        }

                        if ui.button("RA -V (Bottom)").clicked() {
                            (theta, delta) = (std::f32::consts::PI, -std::f32::consts::PI * 0.5 + 1e-3);
                        }

                        ui.separator();

                        ui.checkbox(&mut show_unique_slice, "Slice selector");
                        ui.add_enabled_ui(show_unique_slice, |ui| {
                            ui.add(egui::Slider::new(&mut slice_idx, (fmin as u32)..=(fmax as u32)).text("slice idx"));
                        });

                        ui.separator();

                        ui.add_enabled_ui(!show_unique_slice, |ui| {
                            ui.label("Select a frequency range");
                            ui.horizontal(|ui| {
                                ui.add(egui::DragValue::new(&mut f1).speed(1.0));
                                ui.add(
                                    DoubleSlider::new(&mut f1, &mut f2, fmin..=fmax)
                                        .scroll_factor(1.0)
                                );
                                ui.add(egui::DragValue::new(&mut f2).speed(1.0));
                            });

                            ui.add(egui::Slider::new(&mut fov, fov_min..=fov_max as f32).text("Select FoV"));
                            ui.add(egui::Slider::new(&mut ra, ra_min..=ra_max as f32).text("Select RA"));
                            ui.add(egui::Slider::new(&mut dec, dec_min..=dec_max as f32).text("Select Dec"));

                            ui.add_enabled_ui(cube.is_some(), |ui| {
                                // f1, f2, fov, ra, dec
                                ui.horizontal(|ui| {
                                    if ui.button("Select").clicked() {
                                        let l = [
                                            (ra - fov * 0.5) / (naxis.0 as f32),
                                            (dec - fov * 0.5) / (naxis.1 as f32),
                                            f1 / (naxis.2 as f32)
                                        ];
                                        let h = [
                                            (ra + fov * 0.5) / (naxis.0 as f32),
                                            (dec + fov * 0.5) / (naxis.1 as f32),
                                            f2 / (naxis.2 as f32)
                                        ];

                                        queue.write_buffer(
                                            &buffers["interaction"],
                                            0,
                                            bytemuck::bytes_of(&[l[0], l[1], l[2], 0.0, h[0], h[1], h[2], 0.0]),
                                        );

                                        // set the new select limits
                                        ra_min = ra - fov * 0.5;
                                        ra_max = ra + fov * 0.5;
                                        dec_min = dec - fov * 0.5;
                                        dec_max = dec + fov * 0.5;
                                        fmin = f1 as f32;
                                        fmax = f2 as f32;
                                        fov_min = 0.0;
                                        fov_max = fov;

                                        #[cfg(target_arch = "wasm32")]
                                        {
                                            let x_px = ra as f64;
                                            let y_px = dec as f64;
                                            let w_px = fov as f64;

                                            if let Some(cube) = cube {
                                                let p = cube
                                                    .wcs
                                                    .unproj(&ImgXY::new(x_px, y_px))
                                                    .unwrap();

                                                let fov = cube
                                                    .wcs.field_of_view().0 * ((w_px as f64) / (naxis.0 as f64));

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
                                            bytemuck::bytes_of(&[0.0_f32, 0.0, 0.0,
                                                0.0,
                                                1.0, 1.0, 1.0,
                                                0.0
                                            ]),
                                        );
                                    }
                                });
                            });
                        });

                        ui.separator();

                        ui.add_enabled_ui(cube.is_some(), |ui| {
                            if let Some(cube) = cube {
                                ui.horizontal(|ui| {
                                    if ui.button("Moment 0").clicked() {
                                        if moment0_texture.is_none() {
                                            let image = moment::compute_moment0(&cube);

                                            let tex = ctx
                                                .load_texture(
                                                    "moment0",
                                                    egui::ColorImage::from_rgba_unmultiplied([naxis.0 as usize, naxis.1 as usize], &image),
                                                    egui::TextureOptions::NEAREST,
                                                );

                                            *moment0_texture = Some(tex);
                                        }
                                        show_moment0_window = true;
                                    }

                                    if ui.button("Moment 1").clicked() {
                                        
                                    }
                                    if ui.button("Moment 2").clicked() {
                                        
                                    }
                                });
                            }
                        });
                        
                        if show_moment0_window {
                            egui::Window::new("Moment 0")
                                .open(&mut show_moment0_window)
                                .show(ctx, |ui| {
                                    if let Some(tex) = &moment0_texture {
                                        let size = tex.size_vec2();

                                        ui.image((tex.id(), size));
                                    }
                                });
                        }
                    });

                    if old_bbox_settings != [ra, dec, fov, f1, f2, ra_min, ra_max, dec_min, dec_max, fov_min, fov_max, fmin, fmax, slice_idx as f32] || show_unique_slice != self.show_unique_slice {
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
                                (f1 as f32 - fmin) / (fmax - fmin) - 0.5,
                            ];
                            let h = [
                                (ra + fov * 0.5 - ra_min) / (ra_max - ra_min) - 0.5,
                                (dec + fov * 0.5 - dec_min) / (dec_max - dec_min) - 0.5,
                                (f2 as f32 - fmin) / (fmax - fmin) - 0.5,
                            ];

                            (l, h)
                        };

                        queue.write_buffer(
                            &buffers["interaction"],
                            offset_of!(Interaction, bbox_min) as wgpu::BufferAddress,
                            bytemuck::bytes_of(&[
                                l[0], l[1], l[2], 0.0,
                                h[0], h[1], h[2], 0.0
                            ]),
                        );

                        needs_redraw = true;
                    }

                    if old_scene_settings != (theta, delta, perspective) {
                        self.queue.write_buffer(
                            &self.buffers["scene"],
                            std::mem::offset_of!(uniform::Scene, origin) as wgpu::BufferAddress,
                            bytemuck::bytes_of(&[
                                theta, delta, 0.0, 0.0,
                                if perspective { 1.0_f32 } else { 0.0_f32 }
                            ]),
                        );

                        needs_redraw = true;
                    }

                    if old_render_params != (show_isosurface, min_cut, max_cut, isosurface, colormap, transfer, diffuse_color, bg_color) {
                        queue.write_buffer(
                            &buffers["render_params"],
                            0,
                            bytemuck::bytes_of(&RenderParams {
                                cut_iso: [min_cut, max_cut, isosurface],
                                colormap: colormap as i32,
                                diffuse_color,
                                transfer: transfer as i32,
                                bg_color,
                            }),
                        );

                        needs_redraw = true;
                    }

                    self.theta = theta as f64;
                    self.delta = delta as f64;
                    self.dtheta = 0.0;
                    self.ddelta = 0.0;

                    self.isosurface = isosurface;
                    self.perspective = perspective;
                    self.diffuse_color = diffuse_color;
                    self.bg_color = bg_color;
                    self.show_isosurface = show_isosurface;
                    self.show_unique_slice = show_unique_slice;
                    self.colormap = colormap;
                    self.transfer = transfer;
                    self.min_cut = min_cut;
                    self.max_cut = max_cut;

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

                    self.slice_idx = slice_idx;

                    self.show_moment0_window = show_moment0_window;
                }

                self.needs_redraw = needs_redraw;
                self.show_options = show_options;

                if self.needs_redraw || ctx.has_requested_repaint() {
                    window.request_redraw();
                }

                if show_shortkeys {
                    egui::Window::new("fits3 shortkeys")
                    .resizable(true)
                    .show(ctx, |ui| {
                        ui.label("Shortkeys to display the differents faces of the cube :");
                        ui.label("Front : F | 5");
                        ui.label("Left  : L | 4");
                        ui.label("Right : R | 6");
                        ui.label("Top   : T | 8");
                        ui.label("Back  : B | 0");
                        ui.label("Bottom: 2");

                        ui.separator();

                        ui.label("Press 'SPACE' to reset the view");

                        ui.label("Use 'Arrow keys' to move around the cube.");
                    });
                }

                self.show_shortkeys = show_shortkeys;
                
                #[cfg(not(target_arch = "wasm32"))]
                let sf = window.scale_factor() as f32;
                #[cfg(target_arch = "wasm32")]
                let sf = window.scale_factor() as f32 * 0.75;

                #[cfg(not(target_arch = "wasm32"))]
                let pixels_per_point = ctx.pixels_per_point();
                #[cfg(target_arch = "wasm32")]
                let pixels_per_point = ctx.pixels_per_point() * 0.75;

                let screen_descriptor = egui_wgpu::ScreenDescriptor {
                    size_in_pixels: [self.config.width, self.config.height],
                    pixels_per_point: sf,
                };

                // Check for viewport change
                let rect = ctx.available_rect();

                let vx = (rect.min.x * ctx.pixels_per_point()).max(0.0);
                let vy = (rect.min.y * ctx.pixels_per_point()).max(0.0);
                #[cfg(target_arch = "wasm32")]
                let vw = (rect.width() * pixels_per_point)
                    .max(0.0)
                    .min(wgpu::Limits::downlevel_webgl2_defaults().max_texture_dimension_2d as f32);
                #[cfg(not(target_arch = "wasm32"))]
                let vw = (rect.width() * pixels_per_point)
                    .max(0.0);

                #[cfg(target_arch = "wasm32")]
                let vh = (rect.height() * pixels_per_point)
                    .max(0.0)
                    .min(wgpu::Limits::downlevel_webgl2_defaults().max_texture_dimension_2d as f32);
                #[cfg(not(target_arch = "wasm32"))]
                let vh = (rect.height() * pixels_per_point)
                    .max(0.0);

                let new_viewport = ViewPort { x: vx, y: vy, width: vw, height: vh };
                if self.viewport != new_viewport {
                    self.viewport = new_viewport;

                    self.queue.write_buffer(
                        &self.buffers["scene"],
                        std::mem::offset_of!(uniform::Scene, win_size) as wgpu::BufferAddress,
                        bytemuck::bytes_of(&[
                            self.viewport.width, self.viewport.height
                        ]),
                    );
                }

                #[cfg(not(target_arch = "wasm32"))]
                if file_picked {
                    if let Some(path_buf) = &*self.picked_file.clone().borrow() {
                        let file = File::open(path_buf).unwrap();
                        let mmap = unsafe { Mmap::map(&file).unwrap() };

                        let reader = Cursor::new(mmap);
                        let _ = self.visualize_cube(reader);
                        window.request_redraw();
                    }
                }

                self.egui_renderer.end_frame_and_draw(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    window,
                    &view,
                    screen_descriptor,
                );
            }

            self.queue.submit(iter::once(encoder.finish()));
            frame.present();
        }

        Ok(())
    }

    fn visualize_cube<R: AsRef<[u8]> + std::fmt::Debug>(
        &mut self,
        reader: Cursor<R>,
    ) -> Result<(), &'static str> {
        let cube = Cube::from_fits(reader, &self.device, &self.queue)?;

        // reset the cutoff values
        self.queue.write_buffer(
            &self.buffers["render_params"],
            0,
            bytemuck::bytes_of(&[cube.mincut, cube.maxcut]),
        );
        self.queue.write_buffer(
            &self.buffers["volume"],
            0,
            bytemuck::bytes_of(&Volume {
                cube_size: [cube.dim.0 as f32, cube.dim.1 as f32, cube.dim.2 as f32],
                _pad1: 0.0,
                block_size: [32.0, 32.0, ((cube.dim.2 as f32) / 64.0).clamp(16.0, 128.0)],
                _pad2: 0.0
            }),
        );

        self.volumetric_renderer.set_volume(&self.device, &self.buffers, &cube);

        self.naxis = cube.dim;

        self.ra = (cube.dim.0 as f32) * 0.5;
        self.dec = (cube.dim.1 as f32) * 0.5;
        self.f1 = 0.0;
        self.f2 = cube.dim.2 as f32;
        self.fov = cube.dim.0 as f32;
        self.fov_min = 0.0;
        self.fov_max = cube.dim.0 as f32;
        
        self.ra_min = 0.0;
        self.ra_max = cube.dim.0 as f32;
        self.dec_min = 0.0;
        self.dec_max = cube.dim.1 as f32;
        self.fmin = 0.0;
        self.fmax = cube.dim.2 as f32;

        if !self.show_unique_slice {
            self.queue.write_buffer(
                &self.buffers["interaction"],
                0,
                bytemuck::bytes_of(&Interaction {
                    zoom_min: [0.0 as f32, 0.0, 0.0],
                    _pad1: 0.0,
                    zoom_max: [1.0, 1.0, 1.0],
                    _pad2: 0.0,   // padding!
                    bbox_min: [-0.5 as f32, -0.5, -0.5],
                    _pad3: 0.0,   // padding!
                    bbox_max: [0.5 as f32, 0.5, 0.5],
                    _pad4: 0.0,   // padding!
                })
            );
        }

        // TODO: reset the zoom scaling as well to see the whole cube 

        self.min_cut_default = cube.mincut;
        self.max_cut_default = cube.maxcut;
        // by default, set the cuts to the one precalculated
        self.min_cut = cube.mincut;
        self.max_cut = cube.maxcut;

        self.cube = Some(cube);

        self.needs_redraw = true;

        Ok(())
    }
}

use std::sync::Arc;
pub struct App {
    instance: wgpu::Instance,
    state: Option<State>,
    window: Option<Arc<Window>>,

    panning: bool,
    cuts: bool,
    cursor_pos: PhysicalPosition<f64>,
    start_cursor_pos: PhysicalPosition<f64>,
    start_min_cut: f32,
    start_max_cut: f32,

    pub needs_redraw: bool,

    shortcuts: ShortKeyCommands,

    #[cfg(not(target_arch = "wasm32"))]
    i: usize,
}


impl App {
    pub fn new() -> Self {
        // The instance is a handle to our GPU
        // BackendBit::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        Self {
            instance,
            state: None,
            window: None,
            panning: false,
            cuts: false,
            cursor_pos: PhysicalPosition::new(0.0, 0.0),
            start_cursor_pos: PhysicalPosition::new(0.0, 0.0),

            start_min_cut: 0.0,
            start_max_cut: 1.0,

            needs_redraw: true,

            shortcuts: ShortKeyCommands::new(),

            #[cfg(not(target_arch = "wasm32"))]
            i: 0,
        }
    }

    async fn set_window(&mut self, window: Window) {
        let window = Arc::new(window);
        let surface = self
            .instance
            .create_surface(window.clone())
            .expect("Failed to created the wgpu surface.");

        let state = State::new(
            &window,
            &self.instance,
            surface,
        )
        .await;

        self.window.get_or_insert(window);
        self.state.get_or_insert(state);
    }
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = create_window(event_loop);
        pollster::block_on(self.set_window(window));
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::DisplayData => {
                let window = self.window.as_ref().unwrap();
                window.request_redraw();          // ✅ trigger render
            }
        }
    }

    #[allow(unused_variables)]
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let state = self.state
            .as_mut()
            .unwrap();

        let window = self.window.as_ref().unwrap();

        #[cfg(target_arch = "wasm32")]
        js_api::handle_events(state, window);

        // let egui render to process the event first
        let mut ui_taken_event = false;
        let response = state
            .egui_renderer
            .handle_input(self.window.as_ref().unwrap(), &event);

        if let egui_winit::EventResponse { consumed: true, .. } = response {
            // an action has been done on the ui so we must redraw things.
            window.request_redraw();
            ui_taken_event = true;
        }

        match event {
            WindowEvent::ModifiersChanged(new_mods) => {
                self.shortcuts.update_modifiers(new_mods);
            }
            #[cfg(not(target_arch = "wasm32"))]
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => event_loop.exit(),
            #[cfg(not(target_arch = "wasm32"))]
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::KeyA),
                        ..
                    },
                ..
            } => {
                // toggle fullscreen
                self.i = (self.i + 1) % CUBES_PATH.len();

                let file = File::open(&CUBES_PATH[self.i]).unwrap();
                let mmap = unsafe { Mmap::map(&file).unwrap() };

                let reader = Cursor::new(mmap);

                let _ = state
                    .visualize_cube(reader);

                window.request_redraw();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Enter),
                        ..
                    },
                ..
            } => {
                // toggle fullscreen
                self.window
                    .as_ref()
                    .unwrap()
                    .set_fullscreen(Some(Fullscreen::Borderless(None)));

                window.request_redraw();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Space),
                        ..
                    },
                ..
            } => {
                state.set_camera_position(std::f64::consts::PI,0.0,0.0,0.0, window);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::ArrowLeft),
                        ..
                    },
                ..
            } => {
                state.theta -= std::f64::consts::PI/4.0;
                state.set_camera_position(state.theta, 0.0, state.delta, 0.0, window);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::ArrowRight),
                        ..
                    },
                ..
            } => {
                state.theta += std::f64::consts::PI/4.0;
                state.set_camera_position(state.theta, 0.0, state.delta, 0.0, window);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::ArrowUp),
                        ..
                    },
                ..
            } => {
                state.delta = (state.delta + std::f64::consts::PI / 4.0).clamp(
                    -std::f64::consts::PI * 0.5 + 1e-3,
                    std::f64::consts::PI * 0.5 - 1e-3,
                );
                state.set_camera_position(state.theta, state.dtheta, state.delta, state.ddelta, window);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::ArrowDown),
                        ..
                    },
                ..
            } => {
                state.delta = (state.delta - std::f64::consts::PI / 4.0).clamp(
                    -std::f64::consts::PI * 0.5 + 1e-3,
                    std::f64::consts::PI * 0.5 - 1e-3,
                );
                state.set_camera_position(state.theta, state.dtheta, state.delta, state.ddelta, window);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::KeyF) | PhysicalKey::Code(KeyCode::Numpad2),
                        ..
                    },
                ..
            } => {
                state.set_camera_position(std::f64::consts::PI,0.0,0.0,0.0, window);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::KeyB) | PhysicalKey::Code(KeyCode::Numpad8),
                        ..
                    },
                ..
            } => {
                state.set_camera_position(0.0,0.0,0.0,0.0, window);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::KeyL) | PhysicalKey::Code(KeyCode::Numpad4),
                        ..
                    },
                ..
            } => {
                state.set_camera_position(-std::f64::consts::PI * 0.5,0.0,0.0,0.0, window);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::KeyR) | PhysicalKey::Code(KeyCode::Numpad6),
                        ..
                    },
                ..
            } => {
                state.set_camera_position(std::f64::consts::PI * 0.5,0.0,0.0,0.0, window);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::KeyT) | PhysicalKey::Code(KeyCode::Numpad0),
                        ..
                    },
                ..
            } => {
                state.set_camera_position(std::f64::consts::PI, 0.0, std::f64::consts::PI * 0.5 - 1e-3,0.0, window);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Numpad5),
                        ..
                    },
                ..
            } => {
                state.set_camera_position(std::f64::consts::PI,0.0,-std::f64::consts::PI * 0.5 + 1e-3,0.0, window);
            }
            WindowEvent::KeyboardInput {
                event,
                ..
            } => {
                // Ctrl+O
                self.shortcuts.process_key_event(PhysicalKey::Code(KeyCode::KeyO), &event, true, || {
                    state.show_options = !state.show_options;

                    window.request_redraw();
                });

                self.shortcuts.process_key_event(PhysicalKey::Code(KeyCode::KeyI), &event, true, || {
                    state.show_shortkeys = !state.show_shortkeys;

                    window.request_redraw();
                });
            },
            WindowEvent::Resized(physical_size) => state.resize(physical_size),
            WindowEvent::RedrawRequested => {
                let window = self.window.as_ref().unwrap();
                let _ = state.render(window);

                /*let pointer_over_egui = state
                    .egui_renderer
                    .context()
                    .is_pointer_over_area();*/

                if state.needs_redraw {
                    state.needs_redraw = false;
                    
                    window.request_redraw();
                }
            }
            // Moving
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if !ui_taken_event {
                    self.panning = true;
                    self.start_cursor_pos = self.cursor_pos;
                    state.dtheta = 0.0;
                    state.ddelta = 0.0;
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => {
                self.panning = false;
                state.theta += state.dtheta;
                state.delta += state.ddelta;

                state.delta = state.delta.clamp(
                    -std::f64::consts::PI * 0.5 + 1e-3,
                    std::f64::consts::PI * 0.5 - 1e-3,
                );
            }
            // Change cuts
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Right,
                ..
            } => {
                self.cuts = true;
                self.start_cursor_pos = self.cursor_pos;
                self.start_min_cut = self.state.as_ref().unwrap().min_cut;
                self.start_max_cut = self.state.as_ref().unwrap().max_cut;
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Right,
                ..
            } => {
                self.cuts = false;
            }
            WindowEvent::CursorEntered { .. } => {
                window.request_redraw(); // ✅ force a frame
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_pos = position;

                if self.panning {
                    let dx = (self.cursor_pos.x - self.start_cursor_pos.x)
                        / ((state.size.width as f64) * 0.5);
                    let dy = (self.cursor_pos.y - self.start_cursor_pos.y)
                        / ((state.size.height as f64) * 0.5);

                    state.dtheta = 2.0 * dx;
                    state.ddelta = dy;

                    let d = (state.delta as f32 + state.ddelta as f32).clamp(
                        -std::f32::consts::PI * 0.5 + 1e-3,
                        std::f32::consts::PI * 0.5 - 1e-3,
                    );

                    state.queue.write_buffer(
                        &state.buffers["scene"],
                        offset_of!(Scene, origin) as wgpu::BufferAddress,
                        bytemuck::bytes_of(&[state.theta as f32 + state.dtheta as f32, d]),
                    );
                } else if self.cuts {
                    let dx =
                        ((self.cursor_pos.x - self.start_cursor_pos.x) as f32) / ((state.size.width as f32) * 0.5);
                    let dy =
                        ((self.cursor_pos.y - self.start_cursor_pos.y) as f32) / ((state.size.height as f32) * 0.5);

                    // between -1 and 1

                    let l = state.max_cut_default - state.min_cut_default;
                    state.min_cut = self.start_min_cut + dx * l + dy * l;
                    state.max_cut = self.start_max_cut + dx * l - dy * l;

                    state.queue.write_buffer(
                        &state.buffers["render_params"],
                        0,
                        bytemuck::bytes_of(&[
                            state.min_cut,
                            state.max_cut,
                        ]),
                    );
                }

                window.request_redraw();
            }
            _ => {}
        }
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    #[cfg(target_arch = "wasm32")]
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    #[cfg(target_arch = "wasm32")]
    console_log::init_with_level(log::Level::Warn).expect("Couldn't initialize logger");
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();

    let event_loop = EventLoop::<UserEvent>::with_user_event().build().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);

    // Create a proxy to send events to
    let proxy = event_loop.create_proxy();

    user_event::create_proxy(&event_loop);

    let mut app = App::new();
    event_loop.run_app(&mut app).expect("Failed to run the app");
}

fn create_window(event_loop: &ActiveEventLoop) -> Window {
    #[cfg(not(target_arch = "wasm32"))]
    let win_attrs = Window::default_attributes().with_title("Astronomical cube visualizer");
    #[cfg(target_arch = "wasm32")]
    let mut win_attrs = Window::default_attributes().with_title("Astronomical cube visualizer");

    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;
        use winit::platform::web::WindowAttributesExtWebSys;
        let canvas = web_sys::window()
            .unwrap()
            .document()
            .unwrap()
            .get_element_by_id("canvas")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();

        win_attrs = win_attrs.with_canvas(Some(canvas));
    }

    event_loop.create_window(win_attrs).unwrap()
}