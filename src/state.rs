use egui_plot::Legend;
use egui_plot::PlotPoints;
use egui_plot::Line;
use egui_plot::Plot;

use crate::gui::EguiRenderer;

use crate::gui::selector_frame::SelectorFrame;
use crate::gui::viewport_frame::ViewportFrame;
use crate::gui::isosurface_frame::IsosurfaceFrame;
use crate::gui::settings_frame::SettingsFrame;

use crate::renderer::texture::TextureRenderer;
use egui_double_slider::DoubleSlider;
#[cfg(not(target_arch = "wasm32"))]
use egui_file_dialog::{FileDialog, DialogMode};
use egui::{Area, Align2};
use std::iter;
use std::mem::offset_of;

use glam::{Vec2, Vec3};

use crate::uniform::{Interaction, VolumetricRenderParams, SurfaceRenderParams, Scene, Volume};
use std::f32::consts::PI;

use winit::window::Window;

use crate::cube::Cube;
use crate::renderer::wireframe::WireframeRenderer;

use crate::renderer::volumetric::VolumetricRenderer;

#[cfg(not(target_arch = "wasm32"))]
use memmap2::Mmap;
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
use std::io::Cursor;

use std::cell::RefCell;
use std::rc::Rc;

use std::collections::HashMap;


use winit::window::Fullscreen;

pub struct State {
    surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pub size: winit::dpi::PhysicalSize<u32>,

    viewport: ViewPort,

    is_surface_configured: bool,
    pub is_fullscreen: bool,
    screen_mode: Option<Fullscreen>,

    volumetric_renderer: VolumetricRenderer,
    wireframe_renderer: WireframeRenderer,
    texture_renderer: TextureRenderer,

    // Offscreen wgpu view texture
    render_texture: wgpu::Texture,
    // A texture storing the final saveable view, rendered on demand when
    // the user saves the image
    final_texture: wgpu::Texture,

    // UI
    selector_frame: SelectorFrame,
    viewport_frame: ViewportFrame,
    isosurface_frame: IsosurfaceFrame,
    settings_frame: SettingsFrame,

    // uniforms
    pub buffers: HashMap<&'static str, wgpu::Buffer>,

    // NAXIS of the current loaded cube
    naxis: (u32, u32, u32),

    pub zoom_factor: f32,

    #[cfg(not(target_arch = "wasm32"))]
    file_dialog: FileDialog,
    picked_file: Rc<RefCell<Option<String>>>,

    // min cut precomputed corresponding to the first 1% of data
    pub min_cut_default: f32,
    // max cut precomputed corresponding to the last 99% of data
    pub max_cut_default: f32,

    // current min cut
    pub min_cut: f32,
    // current max cut
    pub max_cut: f32,

    /// ui options
    pub show_options: bool,
    show_unique_slice: bool,

    // The cube real data
    cube: Option<Cube>,

    // moment 0 state
    show_moment0_window: bool,
    pub show_spectra_window: bool,
    pub spectra_data: Vec<[f64; 2]>,
    moment0_texture: Option<egui::TextureHandle>,

    pub delta: f64,
    pub theta: f64,
    pub egui_renderer: EguiRenderer, //egui: EguiRenderer,

    pub needs_redraw: bool,
    pub needs_redraw_view: bool,

    mode: egui::CursorIcon,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ViewPort {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}
impl State {
    pub(crate) async fn new(
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
        /*let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);*/
        #[cfg(target_arch = "wasm32")]
        let surface_format = wgpu::TextureFormat::Rgba8UnormSrgb;
        #[cfg(not(target_arch = "wasm32"))]
        let surface_format = wgpu::TextureFormat::Bgra8Unorm;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            //view_formats: vec![surface_format.add_srgb_suffix()],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let buffers: HashMap<&'static str, wgpu::Buffer> = vec![
            (
                "scene",
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Scene uniform buffer"),
                    size: std::mem::size_of::<Scene>() as wgpu::BufferAddress,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            ),
            (
                "volume",
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Volume uniform buffer"),
                    size: std::mem::size_of::<Volume>() as wgpu::BufferAddress,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            ),
            (
                "volumetric_render_params",
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Volume render params uniform buffer"),
                    size: std::mem::size_of::<VolumetricRenderParams>() as wgpu::BufferAddress,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            ),
            (
                "surface_render_params",
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Surface render params uniform buffer"),
                    size: std::mem::size_of::<SurfaceRenderParams>() as wgpu::BufferAddress,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            ),
            (
                "interaction",
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Interaction uniform buffer"),
                    size: std::mem::size_of::<Interaction>() as wgpu::BufferAddress,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            ),
        ]
        .into_iter()
        .collect();

        // Uniform buffer
        queue.write_buffer(
            &buffers["scene"],
            0,
            bytemuck::bytes_of(&Scene {
                win_size: [1.0_f32, 0.0, 0.0, 0.0],
                origin: [PI, 0.0, 0.0, 0.0],
                perspective: [0.0, 0.0, 0.0, 0.0],
                zoom_factor: 1.0,
                _pad1: [0.0; 3],
            }),
        );

        queue.write_buffer(
            &buffers["volumetric_render_params"],
            0,
            bytemuck::bytes_of(&VolumetricRenderParams {
                cut: [1.0_f32, 0.0],
                colormap: 0,
                transfer: 0,
                bg_color: [0.0_f32; 3],
                _pad: 0.0,
            }),
        );
        queue.write_buffer(
            &buffers["surface_render_params"],
            0,
            bytemuck::bytes_of(&SurfaceRenderParams {
                diffuse_color: [0.0_f32, 1.0, 0.0, 1.0],
                iso: 0.0,
                _pad: [0.0; 3]
            }),
        );

        queue.write_buffer(
            &buffers["volume"],
            0,
            bytemuck::bytes_of(&Volume {
                cube_size: [1.0_f32, 1.0, 1.0],
                _pad1: 0.0,
                block_size: [1.0_f32, 1.0, 1.0],
                _pad2: 0.0, // padding!
            }),
        );

        queue.write_buffer(
            &buffers["interaction"],
            0,
            bytemuck::bytes_of(&Interaction {
                zoom_min: [0.0_f32, 0.0, 0.0],
                _pad1: 0.0,
                zoom_max: [1.0, 1.0, 1.0],
                _pad2: 0.0, // padding!
                bbox_min: [-0.5_f32, -0.5, -0.5],
                _pad3: 0.0, // padding!
                bbox_max: [0.5_f32, 0.5, 0.5],
                _pad4: 0.0, // padding!
            }),
        );

        // Uniform buffer
        // Egui renderer init
        let egui_renderer = EguiRenderer::new(&device, config.format, window);

        // Transfer local data for wasm
        let volumetric_renderer = VolumetricRenderer::new(&device, &queue, &config, &buffers);
        let wireframe_renderer = WireframeRenderer::new(&device, &config, &buffers);

        let render_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cached_render"),
            size: wgpu::Extent3d {
                width: size.width.max(1),
                height: size.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[config.format],
        });
        let final_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: size.width.max(1),
                height: size.height.max(1),
                depth_or_array_layers: 1,
            },
            format: config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            label: Some("final_texture"),
            view_formats: &[config.format],
        });
        let texture_renderer = TextureRenderer::new(&device, &config, &buffers, &render_texture);

        let is_fullscreen = false;
        let screen_mode = window.fullscreen();

        Self {
            surface,
            device,
            queue,
            config,
            size,

            is_surface_configured: false,
            render_texture,
            final_texture,

            // uniforms
            buffers,

            viewport: ViewPort {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
            },

            naxis: (1, 1, 1),

            min_cut_default: 0.0,
            max_cut_default: 1.0,
            // current min cut
            min_cut: 0.0,
            // current max cut
            max_cut: 1.0,

            is_fullscreen,
            screen_mode,

            mode: egui::CursorIcon::Default,
            
            show_options: false,
            show_unique_slice: false,
            cube: None,

            selector_frame: SelectorFrame::new(),
            viewport_frame: ViewportFrame::new(),
            isosurface_frame: IsosurfaceFrame::new(),
            settings_frame: SettingsFrame::new(),

            delta: 0.0,
            theta: std::f64::consts::PI,
            zoom_factor: 1.0,

            #[cfg(not(target_arch = "wasm32"))]
            file_dialog: FileDialog::new()
                .add_file_filter_extensions("FITS cube", vec!["fits"])
                .default_file_filter("FITS cube"),
            picked_file: Rc::new(RefCell::new(None)),

            egui_renderer,
            volumetric_renderer,
            wireframe_renderer,
            texture_renderer,

            // Moment0 window state
            show_moment0_window: false,
            show_spectra_window: false,
            spectra_data: vec![],
            moment0_texture: None,

            needs_redraw: true,
            needs_redraw_view: true,
        }
    }

    pub fn resize(&mut self, mut new_size: winit::dpi::PhysicalSize<u32>, window: &Window) {
        new_size.width = (new_size.width as f32) as u32;
        new_size.height = (new_size.height as f32) as u32;

        if new_size.width > 0 && new_size.height > 0 {
            #[cfg(target_arch = "wasm32")]
            {
                new_size.width = new_size.width.min(wgpu::Limits::downlevel_webgl2_defaults().max_texture_dimension_2d);
                new_size.height = new_size.height.min(wgpu::Limits::downlevel_webgl2_defaults().max_texture_dimension_2d);
            }

            #[cfg(not(target_arch = "wasm32"))]
            {
                new_size.width = new_size.width;
                new_size.height = new_size.height;
            }

            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;

            self.render_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("cached_render"),
                size: wgpu::Extent3d {
                    width: self.size.width.max(1),
                    height: self.size.height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[self.config.format],
            });
            self.final_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("cached_render"),
                size: wgpu::Extent3d {
                    width: self.size.width.max(1),
                    height: self.size.height.max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[self.config.format],
            });

            self.texture_renderer.set_texture(&self.render_texture, &self.buffers, &self.device);

            let ctx = self.egui_renderer.context();

            let sf = window.scale_factor() as f32;
            let pixels_per_point = ctx.pixels_per_point();

            let screen_descriptor = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [self.config.width, self.config.height],
                pixels_per_point: sf,
            };

            // Check for viewport change
            let rect = ctx.available_rect();

            let vx = (rect.min.x * pixels_per_point).max(0.0);
            let vy = (rect.min.y * pixels_per_point).max(0.0);
            #[cfg(target_arch = "wasm32")]
            let vw = (rect.width() * pixels_per_point)
                .max(0.0)
                .min(wgpu::Limits::downlevel_webgl2_defaults().max_texture_dimension_2d as f32);
            #[cfg(not(target_arch = "wasm32"))]
            let vw = (rect.width() * pixels_per_point).max(0.0);

            #[cfg(target_arch = "wasm32")]
            let vh = (rect.height() * pixels_per_point)
                .max(0.0)
                .min(wgpu::Limits::downlevel_webgl2_defaults().max_texture_dimension_2d as f32);
            #[cfg(not(target_arch = "wasm32"))]
            let vh = (rect.height() * pixels_per_point).max(0.0);

            let new_viewport = ViewPort {
                x: 0.0,
                y: 0.0,
                width: self.config.width as f32,
                height: self.config.height as f32,
            };
            if self.viewport != new_viewport {
                self.viewport = new_viewport;
            }

            #[cfg(target_arch = "wasm32")]
            {
                use wasm_bindgen::JsCast;
                let canvas = web_sys::window()
                    .unwrap()
                    .document()
                    .unwrap()
                    .get_element_by_id("canvas")
                    .unwrap()
                    .dyn_into::<web_sys::HtmlCanvasElement>()
                    .unwrap();

                canvas.set_width(self.config.width as u32);
                canvas.set_height(self.config.height as u32);
            }

            self.queue.write_buffer(
                &self.buffers["scene"],
                0,
                bytemuck::bytes_of(&[self.size.width as f32, self.size.height as f32, 0.0, 0.0]),
            );

            self.needs_redraw_view = true;
        }
    }

    pub fn set_camera_position(&mut self, theta: f64, delta: f64, window: &Window) {
        self.theta = theta;
        self.delta = delta;

        self.viewport_frame.set_camera_position(theta, delta);

        self.queue.write_buffer(
            &self.buffers["scene"],
            offset_of!(Scene, origin) as wgpu::BufferAddress,
            bytemuck::bytes_of(&[self.theta as f32, self.delta as f32]),
        );

        self.needs_redraw_view = true;
        window.request_redraw();
    }

    pub fn render(&mut self, window: &Window) -> Result<(), wgpu::SurfaceError> {
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        if !self.is_surface_configured {
            return Ok(());
        }

        if let Ok(frame) = self.surface.get_current_texture() {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render Encoder"),
                });

            if self.needs_redraw_view {
                let offscreen_view = self.render_texture.create_view(&wgpu::TextureViewDescriptor {
                    //format: Some(self.config.format.add_srgb_suffix()),
                    ..Default::default()
                });

                self.volumetric_renderer.render_frame(
                    &mut encoder,
                    &offscreen_view,
                    self.isosurface_frame.open,
                    &self.viewport,
                );
                self.wireframe_renderer
                    .render_frame(&mut encoder, &offscreen_view, &self.viewport);

                self.needs_redraw_view = false;
            }

            let screen_view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
                //format: Some(self.config.format.add_srgb_suffix()),
                ..Default::default()
            });

            self.texture_renderer.render(&mut encoder, &screen_view, &self.viewport);
            
            let mut save_triggered = None;

            {
                self.egui_renderer.begin_frame(window);

                let mut zoom = false;
                let mut unzoom = false;
                let mut autoscale = false;

                Area::new("zoom_controls".into())
                    .anchor(Align2::RIGHT_CENTER, [-10.0, 0.0]) // right-middle, slight padding
                    .show(self.egui_renderer.context(), |ui| {
                        ui.vertical(|ui| {
                            if ui.button("➕").on_hover_cursor(egui::CursorIcon::ZoomIn).clicked() {
                                // zoom in
                                zoom = true;
                            }

                            if ui.button("➖").on_hover_cursor(egui::CursorIcon::ZoomOut).clicked() {
                                // zoom out
                                unzoom = true;
                            }

                            if ui.button("⛶").on_hover_cursor(egui::CursorIcon::ResizeSouthEast).clicked() {
                                // autoscale / fit
                                autoscale = true;
                            }
                        });
                    });

                if zoom {
                    self.zoom();
                }

                if unzoom {
                    self.unzoom();
                }

                if autoscale {
                    self.autoscale();
                }

                let ctx = self.egui_renderer.context();
                let mut show_options = self.show_options;
                let show_unique_slice = self.show_unique_slice;
                let viewport = &self.viewport;

                let mut theta = self.theta as f32;
                let mut delta = self.delta as f32;

                let mut show_moment0_window = self.show_moment0_window;
                let mut show_spectra_window = self.show_spectra_window;
                let mode = self.mode;

                let cube = self.cube.as_ref();
                let moment0_texture = &mut self.moment0_texture;
                let naxis = &self.naxis;
                let queue = &self.queue;
                let surface = &self.surface;

                let spectra = PlotPoints::new(self.spectra_data.clone());

                let mut needs_redraw_view = self.needs_redraw_view;

                let picked_file = self.picked_file.clone();
                #[cfg(not(target_arch = "wasm32"))]
                let mut file_picked = false;

                #[cfg(not(target_arch = "wasm32"))]
                let file_dialog = &mut self.file_dialog;

                let buffers = &self.buffers;
                let mut needs_redraw_view = self.needs_redraw_view;

                let device = &self.device;
                let render_texture = &self.render_texture;
                let final_texture = &self.final_texture;
                let config = &self.config;
                let texture_renderer = &mut self.texture_renderer;

                let selector_frame = &mut self.selector_frame;
                let viewport_frame = &mut self.viewport_frame;
                let isosurface_frame = &mut self.isosurface_frame;
                let settings_frame = &mut self.settings_frame;
                let min_cut = &mut self.min_cut;
                let max_cut = &mut self.max_cut;
                let min_cut_default = self.min_cut_default;
                let max_cut_default = self.max_cut_default;

                let mut toggle_fullscreen = false;
                let is_fullscreen = self.is_fullscreen;

                egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.ctx().set_cursor_icon(mode);

                        ui.heading("FITS3");

                        // File import
                        if ui.button("Load a file").clicked() {
                            // Open the file dialog to pick a file.
                            #[cfg(target_arch = "wasm32")]
                            crate::open_file::open_file_dialog(picked_file.clone());

                            #[cfg(not(target_arch = "wasm32"))]
                            file_dialog.pick_file();
                        }

                        settings_frame.render(ui, queue, buffers, &mut needs_redraw_view, min_cut, max_cut, min_cut_default, max_cut_default);

                        if let Some(cube) = cube.as_ref() {
                            selector_frame.render(ui, queue, buffers, cube, &mut needs_redraw_view);
                        }

                        isosurface_frame.render(ui, queue, buffers, &mut needs_redraw_view);

                        viewport_frame.render(ui, queue, buffers, &mut needs_redraw_view);

                        if let Some(cube) = cube {
                            ui.horizontal(|ui| {
                                ui.toggle_value(&mut show_moment0_window, "Moment-0");

                                if show_moment0_window && moment0_texture.is_none() {
                                    let image = crate::moment::compute_moment0(cube);

                                    let tex = ctx.load_texture(
                                        "moment0",
                                        egui::ColorImage::from_rgba_unmultiplied(
                                            [naxis.0 as usize, naxis.1 as usize],
                                            &image,
                                        ),
                                        egui::TextureOptions::NEAREST,
                                    );

                                    *moment0_texture = Some(tex);
                                }

                                egui::Window::new("Moment 0")
                                    .open(&mut show_moment0_window)
                                    .show(ctx, |ui| {
                                        if let Some(tex) = &moment0_texture {
                                            //let size = tex.size_vec2();

                                            ui.image((tex.id(), egui::Vec2::new(500.0, 500.0)));
                                        }
                                    });

                                ui.toggle_value(&mut show_spectra_window, "Extract spectra");

                                if show_spectra_window {
                                    let line = Line::new("Spectra under the mouse", spectra)
                                        .name("Spectra under the mouse");

                                    egui::Window::new("spectra")
                                        .resizable(true)
                                        .movable(true) // still movable
                                        .open(&mut show_spectra_window)
                                        .show(ctx, |ui| {
                                            Plot::new("Spectra")
                                                .legend(Legend::default())
                                                .allow_drag(false)
                                                .x_axis_label("Frequency")
                                                .y_axis_label("Intensity")
                                                .view_aspect(2.0)
                                                .show(ui, |plot_ui| plot_ui.line(line));
                                        });
                                }
                            });
                        }

                        if ui.button("💾 Save").clicked() {
                            let saved_image_view = final_texture.create_view(&wgpu::TextureViewDescriptor::default());
                            texture_renderer.render(&mut encoder, &saved_image_view, viewport);

                            #[cfg(target_arch = "wasm32")] {
                                save_triggered = Some("preview.png".to_string());
                            }  

                            #[cfg(not(target_arch = "wasm32"))]
                            file_dialog.save_file();  // Opens in save mode
                        }

                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                if ui
                                    .add(egui::Button::new("⛶").selected(is_fullscreen))
                                    .on_hover_text("Enable fullscreen")
                                    .clicked()
                                {
                                    toggle_fullscreen = true;
                                }
                            },
                        );

                        #[cfg(target_arch = "wasm32")]
                        {
                            let is_browser_fullscreen = is_browser_fullscreen();
                            if is_fullscreen != is_browser_fullscreen {
                                toggle_fullscreen = true;
                            }
                        }
                    });
                });

                #[cfg(not(target_arch = "wasm32"))]
                {
                    file_dialog.update(ctx);

                    if let Some(mut path) = file_dialog.take_picked() {
                        match file_dialog.mode() {
                            DialogMode::PickFile => {
                                file_picked = true;
                                if path.extension().unwrap() == "fits" {
                                    let file_name = path
                                        .to_path_buf()
                                        .into_os_string()
                                        .to_str()
                                        .unwrap()
                                        .to_string();
                                    self.picked_file = Rc::new(RefCell::new(Some(file_name)));
                                }
                            },
                            DialogMode::SaveFile => {
                                if path.extension().is_none() {
                                    path.add_extension("png");
                                }

                                let file_name = path
                                    .to_path_buf()
                                    .into_os_string()
                                    .to_str()
                                    .unwrap()
                                    .to_string();

                                save_triggered = Some(file_name);
                            },
                            _ => ()
                        }
                    }
                }

                egui::Area::new("status".into())
                    .anchor(egui::Align2::RIGHT_BOTTOM, [-8.0, -8.0])
                    .interactable(false)
                    .show(ctx, |ui| {
                        ui.set_min_width(0.0); // 🔥 important

                        egui::Frame::default()
                            .fill(egui::Color32::from_black_alpha(120))
                            .corner_radius(4.0)
                            .inner_margin(egui::Margin::same(5))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    if let Some(file) = picked_file.borrow().as_ref() {
                                        ui.label(format!("Picked file: {}", file));
                                    }
                                });
                            });
                    });

                self.show_moment0_window = show_moment0_window;
                self.show_spectra_window = show_spectra_window;

                self.min_cut = *min_cut;
                self.max_cut = *max_cut;

                self.needs_redraw_view = needs_redraw_view;
                self.show_options = show_options;

                if self.needs_redraw_view || ctx.has_requested_repaint() {
                    window.request_redraw();
                }

                let sf = window.scale_factor() as f32;
                let pixels_per_point = ctx.pixels_per_point();

                let screen_descriptor = egui_wgpu::ScreenDescriptor {
                    size_in_pixels: [self.config.width, self.config.height],
                    pixels_per_point: sf,
                };

                // Check for viewport change
                let rect = ctx.available_rect();

                let vx = (rect.min.x * pixels_per_point).max(0.0);
                let vy = (rect.min.y * pixels_per_point).max(0.0);
                #[cfg(target_arch = "wasm32")]
                let vw = (rect.width() * pixels_per_point)
                    .max(0.0)
                    .min(wgpu::Limits::downlevel_webgl2_defaults().max_texture_dimension_2d as f32);
                #[cfg(not(target_arch = "wasm32"))]
                let vw = (rect.width() * pixels_per_point).max(0.0);

                #[cfg(target_arch = "wasm32")]
                let vh = (rect.height() * pixels_per_point)
                    .max(0.0)
                    .min(wgpu::Limits::downlevel_webgl2_defaults().max_texture_dimension_2d as f32);
                #[cfg(not(target_arch = "wasm32"))]
                let vh = (rect.height() * pixels_per_point).max(0.0);

                let new_viewport = ViewPort {
                    x: vx,
                    y: vy,
                    width: vw,
                    height: vh,
                };
                if self.viewport != new_viewport {
                    self.viewport = new_viewport;

                    self.queue.write_buffer(
                        &self.buffers["scene"],
                        std::mem::offset_of!(Scene, win_size) as wgpu::BufferAddress,
                        bytemuck::bytes_of(&[self.viewport.width, self.viewport.height]),
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

                if toggle_fullscreen {
                    self.toggle_fullscreen(window);
                }

                self.egui_renderer.end_frame_and_draw(
                    &self.device,
                    &self.queue,
                    &mut encoder,
                    window,
                    &screen_view,
                    screen_descriptor,
                );
            }

            self.queue.submit(iter::once(encoder.finish()));
            frame.present();

            if let Some(file_name) = save_triggered.take() {
                self.export2PNG(&file_name);
            }
        }

        Ok(())
    }

    pub fn export2PNG(&self, filename: &str) {
        let vw = self.viewport.width as u32;
        let vh = self.viewport.height as u32;
        let vx = self.viewport.x as u32;
        let vy = self.viewport.y as u32;

        #[cfg(target_arch = "wasm32")]
        {
            let device2 = self.device.clone();
            let queue2 = self.queue.clone();
            let render_texture2 = self.final_texture.clone();
            //let render_texture2 = render_texture.clone();
            let config2 = self.config.clone();
            let filename = filename.to_string();

            wasm_bindgen_futures::spawn_local(async move {
                crate::save_file::save_texture_to_png(
                    device2,
                    queue2,
                    render_texture2,
                    config2,
                    vw,
                    vh,
                    vx,
                    vy,
                    &filename
                ).await;
            });
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            crate::save_file::save_texture_to_png(
                &self.device,
                &self.queue,
                &self.render_texture,
                &self.config,
                vw,
                vh,
                vx,
                vy,
                &filename
            );
        }
    }

    pub fn set_perspective(&mut self, perspective: bool) {
        self.queue.write_buffer(
            &self.buffers["scene"],
            offset_of!(Scene, perspective) as wgpu::BufferAddress,
            bytemuck::bytes_of(&[if perspective { 1.0_f32 } else { 0.0_f32 }]),
        );

        self.viewport_frame.set_perspective(perspective);
    }


    pub fn toggle_fullscreen(&mut self, window: &Window) {
        use winit::window::Fullscreen;
        self.is_fullscreen = !self.is_fullscreen;
                            
        if self.is_fullscreen {    
            window.set_fullscreen(Some(Fullscreen::Borderless(None)));
        } else {
            window.set_fullscreen(self.screen_mode.clone());
        }

        window.request_redraw();
    }

    pub fn set_zoom_factor(&mut self, zoom_factor: f32) {
        self.zoom_factor = zoom_factor;
        self.queue.write_buffer(
            &self.buffers["scene"],
            offset_of!(Scene, zoom_factor) as wgpu::BufferAddress,
            bytemuck::bytes_of(&[zoom_factor]),
        );
    }

    pub(crate) fn zoom(&mut self) {
        self.zoom_factor = (self.zoom_factor / 1.3).clamp(0.01, 1.0);
        self.set_zoom_factor(self.zoom_factor);
    }

    pub(crate) fn unzoom(&mut self) {
        self.zoom_factor = (self.zoom_factor * 1.3).clamp(0.01, 1.0);
        self.set_zoom_factor(self.zoom_factor);
    }

    pub(crate) fn autoscale(&mut self) {
        self.zoom_factor = 1.0;
        self.set_zoom_factor(self.zoom_factor);
    }

    pub(crate) fn set_cursor(&mut self, mode: egui::CursorIcon) {
        self.mode = mode;
    }

    pub(crate) fn extract_spectra_at_screen_position(&self, x: u32, y: u32) -> Option<Vec<f32>> {
        if let Some(cube) = self.cube.as_ref() {
            // Transform screen space coordinates into NDCs
            let mut ndc = Vec2::new(
                (x as f32 - self.viewport.x) / self.viewport.width,
                (y as f32) / self.viewport.height,
            ) * 2.0
                - 1.0;

            ndc.y = ndc.y * self.viewport.height / self.viewport.width;
            ndc *= self.zoom_factor;

            // Define the direction and the origin of the ray
            let cam_origin = crate::math::lonlat2xyz(self.theta, self.delta).as_vec3();

            // vector from camera origin to the look
            const CAM_NEAR: f32 = 1.0;

            let cam_dir = -cam_origin;
            let o_cam = cam_origin + cam_dir * CAM_NEAR;

            // find a vector belonging to the plane of screen oriented with y
            let ox = Vec3::new(cam_dir.z, 0.0, -cam_dir.x).normalize();
            let oy = -ox.cross(cam_dir);

            let p_cam = o_cam + ox * ndc.x + oy * ndc.y;

            // Define the zooming level
            let l = Vec3::new(
                (self.selector_frame.ra - self.selector_frame.fov * 0.5) / (self.naxis.0 as f32),
                (self.selector_frame.dec - self.selector_frame.fov * 0.5) / (self.naxis.1 as f32),
                self.selector_frame.f1 / (self.naxis.2 as f32),
            );
            let h = Vec3::new(
                (self.selector_frame.ra + self.selector_frame.fov * 0.5) / (self.naxis.0 as f32),
                (self.selector_frame.dec + self.selector_frame.fov * 0.5) / (self.naxis.1 as f32),
                self.selector_frame.f2 / (self.naxis.2 as f32),
            );

            cube.extract_spectra_from_origin_and_dir(&p_cam, &cam_dir, Some((l, h)))
        } else {
            None
        }
    }

    pub fn visualize_cube<R: AsRef<[u8]> + std::fmt::Debug>(
        &mut self,
        reader: Cursor<R>,
    ) -> Result<(), &'static str> {
        let cube = Cube::from_fits(reader, &self.device, &self.queue)?;

        // reset the cutoff values
        self.queue.write_buffer(
            &self.buffers["volumetric_render_params"],
            0,
            bytemuck::bytes_of(&[cube.mincut, cube.maxcut]),
        );
        self.queue.write_buffer(
            &self.buffers["volume"],
            0,
            bytemuck::bytes_of(&Volume {
                cube_size: [cube.size.0 as f32, cube.size.1 as f32, cube.size.2 as f32],
                _pad1: 0.0,
                block_size: [
                    cube.b_size.0 as f32,
                    cube.b_size.1 as f32,
                    cube.b_size.2 as f32,
                ],
                _pad2: 0.0,
            }),
        );

        self.volumetric_renderer
            .set_volume(&self.device, &self.buffers, &cube);

        self.naxis = cube.size;

        self.selector_frame.reset(&cube);

        if !self.show_unique_slice {
            self.queue.write_buffer(
                &self.buffers["interaction"],
                0,
                bytemuck::bytes_of(&Interaction {
                    zoom_min: [0.0_f32, 0.0, 0.0],
                    _pad1: 0.0,
                    zoom_max: [1.0, 1.0, 1.0],
                    _pad2: 0.0, // padding!
                    bbox_min: [-0.5_f32, -0.5, -0.5],
                    _pad3: 0.0, // padding!
                    bbox_max: [0.5_f32, 0.5, 0.5],
                    _pad4: 0.0, // padding!
                }),
            );
        }

        self.isosurface_frame.set_default_cuts(cube.mincut, cube.maxcut);
        self.min_cut = cube.mincut;
        self.max_cut = cube.maxcut;
        self.min_cut_default = cube.mincut;
        self.max_cut_default = cube.maxcut;

        //self.settings_frame.set_default_cuts(cube.mincut, cube.maxcut);
        //self.settings_frame.set_cuts(cube.mincut, cube.maxcut);

        self.cube = Some(cube);

        self.needs_redraw_view = true;

        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
fn is_browser_fullscreen() -> bool {
    web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.fullscreen_element())
        .is_some()
}
