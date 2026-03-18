extern crate byte_slice_cast;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use winit::event_loop::ControlFlow;
#[cfg(target_arch = "wasm32")]
extern crate console_error_panic_hook;

use std::iter;
use std::convert::TryInto;
use egui_double_slider::DoubleSlider;
use std::path::PathBuf;
use egui_file_dialog::FileDialog;

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
use fitsrs::card::Value;
use fitsrs::HDU;

use crate::math::Vec4;
use texture::Texture;
use time::Clock;
use vertex::{VertexNDC, Vertex};
use crate::selector::SelectorRenderer;

use volumetric::VolumetricRenderer;

use fitsrs::Fits;
#[cfg(not(target_arch = "wasm32"))]
use memmap2::Mmap;
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
use std::io::Cursor;

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

    #[cfg(target_arch = "wasm32")]
    recv_data: async_channel::Receiver<Vec<u8>>,
    
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


    file_dialog: FileDialog,
    picked_file: Option<PathBuf>,
    // isosurface value
    isosurface: f32,
    // a diffuse color to show the isosurface with
    diffuse_color: [f32; 4],
    // perspective rendering mode
    perspective: bool,
    // slice index
    slice_idx: u32,

    /// ui options
    show_isosurface: bool,
    show_options: bool,
    show_unique_slice: bool,


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
}

use crate::math::Mat4;
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
            ("rotmat", device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rot matrix uniform"),
                size: 64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
            ("time", device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("time in secs since starting"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
            ("size", device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Cube size"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
            ("isosurface", device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Isosurface max value"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
            ("diffuse_color", device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Diffuse color"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
            ("perspective", device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("perspective"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
            ("cam_origin", device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Cam origin"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
            ("cuts", device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Cuts"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
            ("bbox", device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Bbox of drawing"),
                size: 32,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
            ("zoom", device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Bbox of drawing"),
                size: 32,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
            ("window_size", device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("window size uniform"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
            ("cube_size", device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cube size uniform"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
            ("cube_position", device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cube position uniform"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
            ("block_size", device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("block size uniform"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        ].into_iter().collect();

        // Uniform buffer
        // set the initial cut values
        queue.write_buffer(
            &buffers["cuts"],
            0,
            bytemuck::bytes_of(&[1.0 as f32, 0.0, 0.0, 0.0]),
        );

        // Uniform buffer
        // set the initial cut values
        queue.write_buffer(
            &buffers["cube_size"],
            0,
            bytemuck::bytes_of(&[1.0 as f32, 1.0, 1.0, 0.0]),
        );
        queue.write_buffer(
            &buffers["size"],
            0,
            bytemuck::bytes_of(&[1.0 as f32, 1.0, 1.0, 0.0]),
        );
        queue.write_buffer(
            &buffers["cube_position"],
            0,
            bytemuck::bytes_of(&[0.0 as f32, 0.0, 0.0, 0.0]),
        );
        queue.write_buffer(
            &buffers["bbox"],
            0,
            bytemuck::bytes_of(&[-0.5 as f32, -0.5, -0.5, 0.0, 0.5, 0.5, 0.5, 0.0]),
        );

        queue.write_buffer(
            &buffers["zoom"],
            0,
            bytemuck::bytes_of(&[0.0 as f32, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]),
        );

        queue.write_buffer(
            &buffers["cam_origin"],
            0,
            bytemuck::bytes_of(&[std::f32::consts::PI, 0.0, 0.0, 0.0]),
        );

        let clock = Clock::now();

        // Egui renderer init
        let egui_renderer = gui::EguiRenderer::new(&device, config.format, window);

        // Transfer local data for wasm
        #[cfg(target_arch = "wasm32")]
        let (send_data, recv_data) = async_channel::unbounded::<Vec<u8>>();

        #[cfg(target_arch = "wasm32")]
        {
            // File reading
            let document = web_sys::window().unwrap().document().unwrap();
            let input = document
                .get_element_by_id("file-input")
                .unwrap()
                .dyn_into::<web_sys::HtmlInputElement>()
                .unwrap();

            let input_cloned = input.clone();
            let sdd = send_data.clone();
            let closure = Closure::wrap(Box::new(move |_: web_sys::Event| {
                if let Some(file_list) = input_cloned.files() {
                    if let Some(file) = file_list.get(0) {
                        let reader = web_sys::FileReader::new().unwrap();

                        let reader_cloned = reader.clone();
                        let sd = sdd.clone();
                        let onloadend_cb = Closure::wrap(Box::new(move |_: web_sys::Event| {
                            let result = reader_cloned.result().unwrap();
                            let array = js_sys::Uint8Array::new(&result);
                            let len = array.length() as usize;
                            let sd3 = sd.clone();

                            wasm_bindgen_futures::spawn_local(async move {
                                let data = array.to_vec();
                                sd3.send(data).await.unwrap();
                            });

                            // Here you can use `data` (Vec<u8>) as you like.
                            web_sys::console::log_1(&format!("Read {} bytes from file", len).into());
                        }) as Box<dyn FnMut(_)>);

                        reader.set_onloadend(Some(onloadend_cb.as_ref().unchecked_ref()));
                        reader.read_as_array_buffer(&file).unwrap();
                        onloadend_cb.forget(); // prevent drop
                    }
                }
            }) as Box<dyn FnMut(_)>);

            input.set_onchange(Some(closure.as_ref().unchecked_ref()));
            closure.forget(); // prevent drop
        }

        let volumetric_renderer = VolumetricRenderer::new(&device, &queue, &config, &buffers);
        let selector_renderer = SelectorRenderer::new(&device, &config, &buffers);

        Self {
            surface,
            device,
            queue,
            config,
            size,
            #[cfg(target_arch = "wasm32")]
            recv_data,

            is_surface_configured: false,

            // uniforms
            buffers,

            naxis: (1, 1, 1),

            min_cut_default: 0.0,
            max_cut_default: 1.0,
            min_cut: 0.0,
            max_cut: 1.0,

            f1: 0.0,
            f2: 100.0,
            fmin: 0.0,
            fmax: 100.0,
            fov: 5.0,
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
            show_isosurface: false,
            show_options: false,
            show_unique_slice: false,
            cube: None,

            delta: 0.0,
            theta: std::f64::consts::PI,
            dtheta: 0.0,
            ddelta: 0.0,

            file_dialog: FileDialog::new().add_file_filter_extensions("FITS cube", vec!["fits"]).default_file_filter("FITS cube"),
            picked_file: None,
            clock,
            egui_renderer,
            volumetric_renderer,
            selector_renderer,

            // Moment0 window state
            show_moment0_window: false,
            moment0_texture: None,
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn resize(&mut self, mut new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            new_size.width = (new_size.width as f32 * 0.75_f32) as u32;
            new_size.height = (new_size.height as f32 * 0.75_f32) as u32;

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
        }
        self.queue.write_buffer(
            &self.buffers["window_size"],
            0,
            bytemuck::bytes_of(&[self.size.width as f32, self.size.height as f32, 0.0, 0.0]),
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;
        }
        self.queue.write_buffer(
            &self.buffers["window_size"],
            0,
            bytemuck::bytes_of(&[self.size.width as f32, self.size.height as f32, 0.0, 0.0]),
        );
    }

    fn update(&mut self) {
        let elapsed = self.clock.elapsed_as_secs();

        let rot = Mat4::from_angle_y(cgmath::Rad(elapsed));
        let rot: &[[f32; 4]; 4] = rot.as_ref();

        self.queue
            .write_buffer(&self.buffers["rotmat"], 0, bytemuck::bytes_of(rot));
        self.queue.write_buffer(
            &self.buffers["time"],
            0,
            bytemuck::bytes_of(&[elapsed, 0.0, 0.0, 0.0]),
        );
    }

    fn render(&mut self, window: &Window) -> Result<(), wgpu::SurfaceError> {
        let mut new_view: Option<(f32, f32)> = None;

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

            self.volumetric_renderer.render_frame(&mut encoder, &view, self.show_isosurface);
            self.selector_renderer.render_frame(&mut encoder, &view);

            {
                self.egui_renderer.begin_frame(window);

                let mut isosurface = self.isosurface;
                let mut perspective = self.perspective;
                let mut diffuse_color = self.diffuse_color;
                let mut show_isosurface = self.show_isosurface;
                let mut show_options = self.show_options;
                let mut show_unique_slice = self.show_unique_slice;
                let mut min_cut = self.min_cut;
                let mut max_cut = self.max_cut;
                let min_cut_default = self.min_cut_default;
                let max_cut_default = self.max_cut_default;

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
                let naxis = &self.naxis;
                let mut show_moment0_window = self.show_moment0_window;
                let moment0_texture = &mut self.moment0_texture;

                let mut v_check = false;
                let mut picked_file = self.picked_file.as_ref();

                if let Some(cube) = self.cube.as_ref() {
                    let queue = &self.queue;

                    let mut slice_idx = self.slice_idx;

                    egui::TopBottomPanel::top("top_bar").show(self.egui_renderer.context(), |ui| {
                        ui.horizontal(|ui| {
                            ui.heading("WebGPU 3D FITS viewer");
                            ui.checkbox(&mut show_options, "Show options");
                        });
                    });

                    let data_length = (self.max_cut_default - self.min_cut_default).abs();
                    let datamin = self.min_cut_default - data_length;
                    let datamax = self.max_cut_default + 5.0*data_length;

                    let ctx = self.egui_renderer.context();
                    let buffers = &self.buffers;

                    if show_options {
                        egui::SidePanel::left("fits3 options")
                        .resizable(true)
                        .show(ctx, |ui| {
                            // Volumetric scope
                            ui.add_enabled_ui(!show_isosurface, |ui| {
                                ui.label("Cutout parameters");
                                ui.add_sized(
                                    [ui.available_width(), 0.0],
                                    DoubleSlider::new(&mut min_cut, &mut max_cut, datamin..=datamax)
                                        .width(ui.available_width())
                                        .scroll_factor((datamax - datamin) / 100.0)
                                        .separation_distance((datamax - datamin) / 100.0)
                                );

                                ui.horizontal(|ui| {
                                    ui.add(egui::Slider::new(&mut min_cut, datamin..=datamax).text("min cut"));
                                });
                                ui.horizontal(|ui| {
                                    ui.add(egui::Slider::new(&mut max_cut, datamin..=datamax).text("max cut"));
                                });
                                if ui.button("Reset cuts").clicked() {
                                    min_cut = min_cut_default;
                                    max_cut = max_cut_default;
                                }
                            });
                            
                            ui.separator();

                            // Isosurface scope
                            ui.checkbox(&mut show_isosurface, "Show isosurface");
                            ui.add_enabled_ui(show_isosurface, |ui| {
                                ui.horizontal(|ui| {
                                    ui.add(egui::Slider::new(&mut isosurface, min_cut_default..=max_cut_default).text("isovalue"));
                                    ui.color_edit_button_rgba_unmultiplied(&mut diffuse_color);
                                })
                            });

                            ui.separator();

                            // Viewport scope
                            ui.label("Viewport");
                            ui.checkbox(&mut perspective, "Perspective");

                            if ui.button("RA Dec (Front)").clicked() {
                                new_view = Some((std::f32::consts::PI, 0.0));
                            }

                            if ui.button("-RA Dec (Back)").clicked() {
                                new_view = Some((0.0, 0.0));
                            }

                            if ui.button("-V Dec (Left)").clicked() {
                                new_view = Some((-std::f32::consts::PI/2.0, 0.0));
                            }

                            if ui.button("V Dec (Right)").clicked() {
                                new_view = Some((std::f32::consts::PI/2.0, 0.0));
                            }

                            if ui.button("RA V (Top)").clicked() {
                                new_view = Some((std::f32::consts::PI, std::f32::consts::PI * 0.5 - 1e-3));
                            }

                            if ui.button("RA -V (Bottom)").clicked() {
                                new_view = Some((std::f32::consts::PI, -std::f32::consts::PI * 0.5 + 1e-3));
                            }

                            ui.separator();

                            // File import
                            if ui.button("Pick file").clicked() {
                                // Open the file dialog to pick a file.
                                //self.file_dialog.pick_file();
                                v_check = true;
                            }
                            if picked_file.is_some() {
                                ui.label(format!("Picked file: {:?}", picked_file.unwrap().as_path().file_name().unwrap()));
                            }

                            /*
                            // Update the dialog
                            self.file_dialog.update(ctx);

                            // Check if the user picked a file.
                            if let Some(path) = self.file_dialog.take_picked() {
                                //self.picked_file = Some(path.to_path_buf());
                                println!("SELECTED");
                            }*/


                            ui.separator();

                            ui.checkbox(&mut show_unique_slice, "Slice selector");
                            ui.add_enabled_ui(show_unique_slice, |ui| {
                                ui.add(egui::Slider::new(&mut slice_idx, (fmin as u32)..=(fmax as u32)).text("slice idx"));
                            });

                            ui.separator();

                            ui.add_enabled_ui(!show_unique_slice, |ui| {
                                ui.label("Select a frequency range");
                                ui.add_sized(
                                    [ui.available_width(), 0.0],
                                    DoubleSlider::new(&mut f1, &mut f2, fmin..=fmax)
                                        .width(ui.available_width())
                                );

                                ui.add(egui::Slider::new(&mut fov, fov_min..=fov_max as f32).text("Select FoV"));
                                ui.add(egui::Slider::new(&mut ra, ra_min..=ra_max as f32).text("Select RA"));
                                ui.add(egui::Slider::new(&mut dec, dec_min..=dec_max as f32).text("Select Dec"));

                                ui.horizontal(|ui| {
                                    // f1, f2, fov, ra, dec
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
                                            &buffers["zoom"],
                                            0,
                                            bytemuck::bytes_of(&[
                                                l[0], l[1], l[2], 0.0,
                                                h[0], h[1], h[2], 0.0
                                            ]),
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
                                            &buffers["zoom"],
                                            0,
                                            bytemuck::bytes_of(&[
                                                0.0_f32, 0.0, 0.0, 0.0,
                                                1.0, 1.0, 1.0, 0.0
                                            ]),
                                        );
                                    }
                                });
                            });

                            ui.separator();
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

                            queue.write_buffer(
                                &buffers["isosurface"],
                                0,
                                bytemuck::bytes_of(&[isosurface, 0.0, 0.0, 0.0]),
                            );
                            queue.write_buffer(
                                &buffers["perspective"],
                                0,
                                bytemuck::bytes_of(&[if perspective { 1.0_f32 } else { 0.0_f32 }, 0.0, 0.0, 0.0]),
                            );
                            queue.write_buffer(
                                &buffers["diffuse_color"],
                                0,
                                bytemuck::bytes_of(&diffuse_color),
                            );
                            queue.write_buffer(
                                &buffers["cuts"],
                                0,
                                bytemuck::bytes_of(&[min_cut, max_cut, 0.0, 0.0]),
                            );

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
                                &buffers["bbox"],
                                0,
                                bytemuck::bytes_of(&[
                                    l[0], l[1], l[2], 0.0,
                                    h[0], h[1], h[2], 0.0
                                ]),
                            );
                        });


                        if let Some((theta, delta)) = new_view {
                            self.theta = theta as f64;
                            self.delta = delta as f64;
                            self.dtheta = 0.0;
                            self.ddelta = 0.0;
                            
                            self.queue.write_buffer(
                                &self.buffers["cam_origin"],
                                0,
                                bytemuck::bytes_of(&[theta, delta, 0.0, 0.0]),
                            );
                        }


                        if v_check {
                            self.file_dialog.pick_file();
                        }
                        self.file_dialog.update(ctx);

                        if let Some(path) = self.file_dialog.take_picked() {
                            if path.extension().unwrap() == "fits" {
                                #[cfg(not(target_arch = "wasm32"))]
                                {
                                    self.picked_file = Some(path.to_path_buf());
                                    let file = File::open(path.to_path_buf()).unwrap();
                                    let mmap = unsafe { Mmap::map(&file).unwrap() };

                                    let reader = Cursor::new(mmap);
                                    let _ = self.visualize_cube(reader);
                                }

                                #[cfg(target_arch = "wasm32")]
                                {
                                    // TODO
                                }
                            }
                        }


                        self.isosurface = isosurface;
                        self.perspective = perspective;
                        self.diffuse_color = diffuse_color;
                        self.show_isosurface = show_isosurface;
                        self.show_unique_slice = show_unique_slice;
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
                }

                self.show_options = show_options;

                #[cfg(not(target_arch = "wasm32"))]
                let screen_descriptor = egui_wgpu::ScreenDescriptor {
                    size_in_pixels: [self.config.width, self.config.height],
                    pixels_per_point: window.scale_factor() as f32,
                };
                #[cfg(target_arch = "wasm32")]
                let screen_descriptor = egui_wgpu::ScreenDescriptor {
                    size_in_pixels: [self.config.width, self.config.height],
                    pixels_per_point: (window.scale_factor() as f32) * 0.75_f32,
                };

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
        let cube = read_fits(reader, &self.device, &self.queue)?;

        // reset the cutoff values
        self.queue.write_buffer(
            &self.buffers["cuts"],
            0,
            bytemuck::bytes_of(&[cube.mincut, cube.maxcut, 0.0, 0.0]),
        );
        self.queue.write_buffer(
            &self.buffers["size"],
            0,
            bytemuck::bytes_of(&[cube.dim.0 as f32, cube.dim.1 as f32, cube.dim.2 as f32, 0.0]),
        );
        self.queue.write_buffer(
            &self.buffers["block_size"],
            0,
            bytemuck::bytes_of(&[32.0, 32.0, ((cube.dim.2 as f32) / 64.0).clamp(16.0, 128.0), 0.0]),
        );

        self.volumetric_renderer.set_volume(&self.device, &self.buffers, &cube);

        self.ra = (cube.dim.0 as f32) * 0.5;
        self.dec = (cube.dim.1 as f32) * 0.5;
        self.f1 = 0.0;
        self.f2 = cube.dim.2 as f32;
        self.fov = cube.dim.0 as f32;
        self.naxis = cube.dim;
        self.ra_min = 0.0;
        self.ra_max = cube.dim.0 as f32;
        self.dec_min = 0.0;
        self.dec_max = cube.dim.1 as f32;
        self.fmin = 0.0;
        self.fmax = cube.dim.2 as f32;
        self.fov_min = 0.0;
        self.fov_max = cube.dim.0 as f32;

        if !self.show_unique_slice {
            self.queue.write_buffer(
                &self.buffers["bbox"],
                0,
                bytemuck::bytes_of(&[
                    -0.5_f32, -0.5, -0.5, 0.0,
                    0.5, 0.5, 0.5, 0.0
                ]),
            );
        }

        // TODO: reset the zoom scaling as well to see the whole cube 

        self.min_cut_default = cube.mincut;
        self.max_cut_default = cube.maxcut;
        // by default, set the cuts to the one precalculated
        self.min_cut = cube.mincut;
        self.max_cut = cube.maxcut;

        self.cube = Some(cube);

        Ok(())
    }
}

use std::ops::Range;
#[cfg(target_arch = "wasm32")]
#[derive(Debug, Default)]
struct Params {
    perspective: Option<bool>,
    cuts: Option<Range<f32>>,
    data: Option<Vec<u8>>,
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    #[cfg(target_arch = "wasm32")]
    static ONSELECT: std::cell::RefCell<Option<js_sys::Function>> =
        std::cell::RefCell::new(None);
}

#[cfg(target_arch = "wasm32")]
use lazy_static::lazy_static;
#[cfg(target_arch = "wasm32")]
lazy_static! {
    static ref CHANNEL_PARAMS: (
        async_channel::Sender<Params>,
        async_channel::Receiver<Params>,
    ) = async_channel::unbounded::<Params>();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = "setPerspective")]
pub fn set_perspective(perspective: bool) {
    wasm_bindgen_futures::spawn_local(async move {
        CHANNEL_PARAMS
            .0
            .send(Params {
                perspective: Some(perspective),
                ..Default::default()
            })
            .await
            .unwrap();
    });
}
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = "normalize")]
pub fn normalize(min: f32, max: f32) {
    wasm_bindgen_futures::spawn_local(async move {
        CHANNEL_PARAMS
            .0
            .send(Params {
                cuts: Some(min..max),
                ..Default::default()
            })
            .await
            .unwrap();
    });
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = "onselect")]
pub fn onselect(func: js_sys::Function) {
    ONSELECT.with(|f| {
        *f.borrow_mut() = Some(func);
    });
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = "displayFITS")]
pub fn display(raw_bytes: js_sys::Uint8Array) {
    wasm_bindgen_futures::spawn_local(async move {
        CHANNEL_PARAMS
            .0
            .send(Params {
                data: Some(raw_bytes.to_vec()),
                ..Default::default()
            })
            .await
            .unwrap();
    });
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

        #[cfg(not(target_arch = "wasm32"))]
        let mut state = State::new(
            &window,
            &self.instance,
            surface,
        )
        .await;
        #[cfg(target_arch = "wasm32")]
        let state = State::new(
            &window,
            &self.instance,
            surface,
        )
        .await;

        #[cfg(not(target_arch = "wasm32"))]
        {
            let file = File::open(&CUBES_PATH[0]).unwrap();
            let mmap = unsafe { Mmap::map(&file).unwrap() };

            let reader = Cursor::new(mmap);
            let _ = state.visualize_cube(reader);
        }

        self.window.get_or_insert(window);
        self.state.get_or_insert(state);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = create_window(event_loop);
        pollster::block_on(self.set_window(window));
    }

    #[allow(unused_variables)]
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let state = self.state
            .as_mut()
            .unwrap();
        #[cfg(target_arch = "wasm32")]
        if let Ok(data) = state.recv_data.try_recv() {
            let reader = Cursor::new(data.as_slice());
            match state.visualize_cube(reader) {
                Ok(()) => {}
                Err(error) => web_sys::window()
                    .unwrap()
                    .alert_with_message(error)
                    .unwrap(),
            }
        }

        #[cfg(target_arch = "wasm32")]
        if let Ok(params) = CHANNEL_PARAMS.1.try_recv() {
            let Params {
                perspective,
                cuts,
                data,
                ..
            } = params;

            if let Some(perspective) = perspective {
                state.queue.write_buffer(
                    &state.buffers["perspective"],
                    0,
                    bytemuck::bytes_of(&[
                        if perspective { 1.0_f32 } else { 0.0_f32 },
                        0.0_f32,
                        0.0_f32,
                        0.0_f32,
                    ]),
                );

                state.perspective = perspective;
            }

            if let Some(cuts) = cuts {
                state.queue.write_buffer(
                    &state.buffers["cuts"],
                    0,
                    bytemuck::bytes_of(&[cuts.start, cuts.end, 0.0_f32, 0.0_f32]),
                );

                state.min_cut = cuts.start;
                state.max_cut = cuts.end;
            }

            if let Some(data) = data {
                let reader = Cursor::new(data.as_slice());
                match state.visualize_cube(reader) {
                    Ok(()) => {}
                    Err(error) => web_sys::window()
                        .unwrap()
                        .alert_with_message(error)
                        .unwrap(),
                }
            }
        }

        // let egui render to process the event first
        if let egui_winit::EventResponse { consumed: true, .. } = state
            .egui_renderer
            .handle_input(self.window.as_ref().unwrap(), &event) {
                return;
            }
        
        match event {
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
                state.theta = std::f64::consts::PI;
                state.dtheta = 0.0;
                state.delta = 0.0;
                state.ddelta = 0.0;
                state.queue.write_buffer(
                    &state.buffers["cam_origin"],
                    0,
                    bytemuck::bytes_of(&[state.theta as f32 + state.dtheta as f32, 0.0, 0.0, 0.0]),
                );
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
                state.theta += std::f64::consts::PI/4.0;
                state.queue.write_buffer(
                    &state.buffers["cam_origin"],
                    0,
                    bytemuck::bytes_of(&[state.theta as f32 + state.dtheta as f32, state.delta as f32 + state.ddelta as f32, 0.0, 0.0]),
                );
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
                state.theta -= std::f64::consts::PI/4.0;
                state.queue.write_buffer(
                    &state.buffers["cam_origin"],
                    0,
                    bytemuck::bytes_of(&[state.theta as f32 + state.dtheta as f32, state.delta as f32 + state.ddelta as f32, 0.0, 0.0]),
                );
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
                state.queue.write_buffer(
                    &state.buffers["cam_origin"],
                    0,
                    bytemuck::bytes_of(&[state.theta as f32 + state.dtheta as f32, state.delta as f32 + state.ddelta as f32, 0.0, 0.0]),
                );
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
                state.queue.write_buffer(
                    &state.buffers["cam_origin"],
                    0,
                    bytemuck::bytes_of(&[state.theta as f32 + state.dtheta as f32, state.delta as f32 + state.ddelta as f32, 0.0, 0.0]),
                );
            }













            WindowEvent::Resized(physical_size) => state.resize(physical_size),
            WindowEvent::RedrawRequested => {
                state.update();
                let window = self.window.as_ref().unwrap();
                let _ = state.render(window);

                window.request_redraw();
            }
            // Moving
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                self.panning = true;
                self.start_cursor_pos = self.cursor_pos;
                state.dtheta = 0.0;
                state.ddelta = 0.0;
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
                        &state.buffers["cam_origin"],
                        0,
                        bytemuck::bytes_of(&[state.theta as f32 + state.dtheta as f32, d, 0.0, 0.0]),
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
                        &state.buffers["cuts"],
                        0,
                        bytemuck::bytes_of(&[
                            state.min_cut,
                            state.max_cut,
                            0.0,
                            0.0,
                        ]),
                    );
                }
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

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);

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

struct Cube {
    data: Vec<f32>,
    dim: (u32, u32, u32),
    mincut: f32,
    maxcut: f32,
    texture: Texture,
    downsampled_texture: Texture,

    #[cfg(target_arch = "wasm32")]
    wcs: fitsrs::WCS
}

fn downsample_8x(
    input: &[f32],
    size_x: usize,
    size_y: usize,
    size_z: usize,
    bx: usize,
    by: usize,
    bz: usize,
) -> Vec<f32> {
    let new_x = (size_x + bx - 1) / bx;
    let new_y = (size_y + by - 1) / by;
    let new_z = (size_z + bz - 1) / bz;

    let mut output = vec![0.0; new_x * new_y * new_z];

    for oz in 0..new_z {
        for oy in 0..new_y {
            for ox in 0..new_x {
                let start_x = ox * bx;
                let start_y = oy * by;
                let start_z = oz * bz;

                let end_x = (start_x + bx).min(size_x);
                let end_y = (start_y + by).min(size_y);
                let end_z = (start_z + bz).min(size_z);

                let mut max = f32::NEG_INFINITY;

                for z in start_z..end_z {
                    for y in start_y..end_y {
                        for x in start_x..end_x {
                            let idx = x + size_x * (y + size_y * z);
                            let p = input[idx];
                            if !p.is_nan() {
                                max = p.max(max);
                            }
                        }
                    }
                }

                let out_idx = ox + new_x * (oy + new_y * oz);
                output[out_idx] = max;
            }
        }
    }

    output
}

fn read_fits<R>(reader: Cursor<R>, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Cube, &'static str>
where
    R: AsRef<[u8]> + std::fmt::Debug,
{
    let mut fits = Fits::from_reader(reader);

    if let Some(Ok(hdu)) = fits.next() {
        match hdu {
            HDU::Primary(hdu) => {
                let header = hdu.get_header();

                if let (
                    Some(Value::Integer { value: w, .. }),
                    Some(Value::Integer { value: h, .. }),
                    Some(Value::Integer { value: d, .. }),
                    Some(Value::Integer { value: b, .. })
                ) = (
                    header.get("NAXIS1"),
                    header.get("NAXIS2"),
                    header.get("NAXIS3"),
                    header.get("BITPIX"),
                ) {
                    let image = fits.get_data(&hdu);

                    let d1 = *w as u32;
                    let d2 = *h as u32;
                    let mut d3 = *d as u32;

                    if d3 == 1 {
                        // parse NAXIS4 instead it there is
                        if let Some(Value::Integer { value, .. }) = header.get("NAXIS4") {
                            d3 = *value as u32;
                        }
                    }

                    let raw_bytes = image.raw_bytes();

                    let (data, cuts) = match b {
                        -32 => {
                            let floats: Vec<f32> = raw_bytes
                                .chunks_exact(4)
                                .map(|b| f32::from_be_bytes(b.try_into().unwrap()))
                                .collect();

                            //let minmax = first_and_last_percent_f32(&mut floats, 0.01, 99.5);
                            let cuts = estimate_default_cuts_from_variance(&floats);
                            (floats, cuts)
                        },
                        8 => {
                            todo!();
                            /*let mut bytes: Vec<u8> = data.to_vec();
                            let range = first_and_last_percent(&mut bytes, 0.01, 99.5);
                            (range.start as f32)..(range.end as f32)*/
                        },
                        16 => {
                            todo!();
                            /*let mut shorts: Vec<i16> = data
                                .chunks_exact(2)
                                .map(|b| i16::from_be_bytes(b.try_into().unwrap()))
                                .collect();

                            let range = first_and_last_percent(&mut shorts, 0.01, 99.5);
                            (range.start as f32)..(range.end as f32)*/
                        },
                        32 => {
                            /*let mut int32: Vec<i32> = data
                                .chunks_exact(4)
                                .map(|b| i32::from_be_bytes(b.try_into().unwrap()))
                                .collect();

                            let range = first_and_last_percent(&mut int32, 0.01, 99.5);
                            (range.start as f32)..(range.end as f32)*/
                            todo!();
                        },
                        64 => {
                            /*let mut int64: Vec<i64> = data
                                .chunks_exact(8)
                                .map(|b| i64::from_be_bytes(b.try_into().unwrap()))
                                .collect();

                            let range = first_and_last_percent(&mut int64, 0.01, 99.5);
                            (range.start as f32)..(range.end as f32)*/
                            todo!();
                        },
                        _ => {
                            return Err("F32 only supported");
                        }
                    };

                    let bz = (d3 / 64).clamp(16, 128) as u32;
                    let bx = 32 as u32;
                    let by = 32 as u32;

                    let downsampled_raw_bytes = downsample_8x(&data, d1 as usize, d2 as usize, d3 as usize, bx as usize, by as usize, bz as usize)
                        .iter()
                        .flat_map(|p| {
                            p.to_le_bytes()
                        })
                        .collect::<Vec<u8>>();

                    let original_dim = (d1, d2, d3);
                    let padding = (
                        (bx - (d1 % bx)) % bx,
                        (by - (d2 % by)) % by,
                        (bz - (d3 % bz)) % bz
                    );

                    let texture = Texture::from_raw_bytes::<f32>(
                        &device,
                        &queue,
                        Some(&raw_bytes),
                        original_dim,
                        padding,
                        4,
                        "cube"
                    )?;

                    let downsampled_texture = Texture::from_raw_bytes::<f32>(
                        &device,
                        &queue,
                        Some(&downsampled_raw_bytes),
                        (
                            (d1 + bx - 1) / bx,
                            (d2 + by - 1) / by,
                            (d3 + bz - 1) / bz
                        ),
                        (0, 0, 0),
                        4,
                        "downgraded_cube"
                    )?;

                    let dim = (
                        original_dim.0,
                        original_dim.1,
                        original_dim.2,
                    );

                    #[cfg(target_arch = "wasm32")]
                    let wcs = hdu.wcs().map_err(|_| "wcs not found")?;

                    // Build the downgrade resolued cube for faster raytracing.
                    // This cube will be first sampled to know whether it is interesting
                    // to sample the full resolued one or to skip to the next big voxel (8x8x8)
                    Ok(Cube {
                        data,
                        dim,
                        mincut: cuts.start,
                        maxcut: cuts.end,
                        #[cfg(target_arch = "wasm32")]
                        wcs,
                        texture,
                        downsampled_texture
                    })
                } else {
                    Err("FITS image extension not found")
                }
            }
            _ => Err("FITS image extension not found"),
        }
    } else {
        Err("Is not a FITS file")
    }
}

pub fn first_and_last_percent_f32(
    slice: &mut [f32],
    mut first_percent: f32,
    mut last_percent: f32,
) -> Range<f32> {
    if slice.is_empty() {
        return 0.0..0.0;
    }

    if first_percent > last_percent {
        std::mem::swap(&mut first_percent, &mut last_percent);
    }

    // Move all NaNs to the end
    let valid_len = {
        let mut i = 0;
        for j in 0..slice.len() {
            if !slice[j].is_nan() {
                slice.swap(i, j);
                i += 1;
            }
        }
        i
    };

    if valid_len == 0 {
        return f32::NAN..f32::NAN;
    }

    let valid = &mut slice[..valid_len];

    let i1 = (first_percent.clamp(0.0, 100.0) as usize * valid_len) / 100;
    let i2 = (last_percent.clamp(0.0, 100.0) as usize * valid_len) / 100;

    let min_val = {
        let (_, min_val, _) =
            valid.select_nth_unstable_by(i1, |a, b| {
                a.total_cmp(&b)
            });
        *min_val
    };
    let max_val = {
        let (_, max_val, _) =
            valid.select_nth_unstable_by(i2, |a, b| {
                a.total_cmp(&b)
            });
        *max_val
    };

    min_val..max_val
}


pub fn estimate_default_cuts_from_variance(
    slice: &[f32],
) -> Range<f32> {
    if slice.is_empty() {
        return 0.0..0.0;
    }

    // Move all NaNs to the end
    /*let valid_len = {
        let mut i = 0;
        for j in 0..slice.len() {
            if !slice[j].is_nan() {
                slice.swap(i, j);
                i += 1;
            }
        }
        i
    };*/
    let valid_len = slice.len();

    if valid_len == 0 {
        return f32::NAN..f32::NAN;
    }

    //let valid = &mut slice[..valid_len];

    /*let (_, median, _) = valid.select_nth_unstable_by(valid_len / 2, |a, b| {
        a.total_cmp(&b)
    });
    let median = *median;*/

    let num_samples = 50000.min(valid_len);
    let s = (valid_len / num_samples).max(1);

    let mut sum = 0.0;
    let mut sum2 = 0.0;
    let mut n = 0;

    for i in (0..slice.len()).step_by(s) {
        let v = slice[i];
        if !v.is_nan() {
            sum += v;
            sum2 += v*v;
            n += 1;
        }
    }

    let mean = sum / (n as f32);
    let sigma = ((sum2 / (n as f32)) - mean*mean).sqrt();

    (sigma)..(15.0 * sigma)
}

pub fn first_and_last_percent<T>(
    slice: &mut [T],
    mut first_percent: f32,
    mut last_percent: f32,
) -> Range<T>
where
    T: std::cmp::Ord + cgmath::Zero + Copy
{
    if slice.is_empty() {
        return T::zero()..T::zero();
    }

    if first_percent > last_percent {
        std::mem::swap(&mut first_percent, &mut last_percent);
    }

   
    let n = slice.len();
    let i1 = (first_percent.clamp(0.0, 100.0) as usize * n) / 100;
    let i2 = (last_percent.clamp(0.0, 100.0) as usize * n) / 100;

    let (_, min_val, _) =
        slice.select_nth_unstable(i1);
    let min_val = *min_val;
    let (_, max_val, _) =
        slice.select_nth_unstable(i2);
    let max_val = *max_val;

    min_val..max_val
}
