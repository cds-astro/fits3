extern crate byte_slice_cast;

use log::warn;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use wgpu::Device;
use winit::event_loop::ControlFlow;
#[cfg(target_arch = "wasm32")]
extern crate console_error_panic_hook;

use std::iter;
use std::path::Path;

use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalPosition,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop, EventLoopProxy},
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

const CUBES_PATH: &[&'static str] = &[
    "./cubes/cutout-CDS_P_LGLBSHI16.fits",
    "./cubes/cutout-CDS_C_GALFAHI.fits",
    "./cubes/NGC3198_cube.fits",
    "./cubes/NGC7331_cube.fits",
    "./cubes/CO_21.fits",
    "./cubes/DHIGLS_DF_Tb.fits",
    "./cubes/DHIGLS_MG_Tb.fits",
    "./cubes/DHIGLS_PO_Tb.fits", //"./cubes/cosmo512-be.fits",
];

const MINMAX: &[Range<f32>] = &[
    0.0..1.0,
    0.0..1.0,
    -2.451346722E-03..1.179221552E-02,
    -2.451346722E-03..1.179221552E-02,
    -2.451346722E-03..1.179221552E-02,
    0.0..1.0,
    0.0..1.0,
    0.0..1.0,
];

use std::collections::HashMap;
struct State {
    surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,

    #[cfg(target_arch = "wasm32")]
    send_data: async_channel::Sender<Vec<u8>>,
    #[cfg(target_arch = "wasm32")]
    recv_data: async_channel::Receiver<Vec<u8>>,
    
    is_surface_configured: bool,

    volumetric_renderer: VolumetricRenderer,
    selector_renderer: SelectorRenderer,

    // uniforms
    buffers: HashMap<&'static str, wgpu::Buffer>,

    clock: Clock,

    mincut: f32,
    maxcut: f32,

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
            ("perspective", device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("perspective"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
            ("minmax", device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("minmax"),
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
            ("window_size", device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("window size uniform"),
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

        queue.write_buffer(
            &buffers["minmax"],
            0,
            bytemuck::bytes_of(&[0.0_f32, 1.0, 0.0, 0.0]),
        );

        let clock = Clock::now();

        // Egui renderer init
        let mut egui_renderer = gui::EguiRenderer::new(&device, config.format, None, 1, window);

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
                                let mut data = array.to_vec();
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
        let selector_renderer = SelectorRenderer::new(&device, &queue, &config, &buffers);

        Self {
            surface,
            device,
            queue,
            config,
            size,
            #[cfg(target_arch = "wasm32")]
            send_data,
            #[cfg(target_arch = "wasm32")]
            recv_data,

            is_surface_configured: false,

            // uniforms
            buffers,

            mincut: 0.0,
            maxcut: 1.0,

            clock,
            egui_renderer,
            volumetric_renderer,
            selector_renderer
        }
    }

    fn resize(&mut self, mut new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            #[cfg(target_arch = "wasm32")]
            {
                new_size.width = (new_size.width as f32 * 0.75_f32) as u32;
                new_size.height = (new_size.height as f32 * 0.75_f32) as u32;

                new_size.width = new_size
                    .width
                    .min(wgpu::Limits::downlevel_webgl2_defaults().max_texture_dimension_2d);
                new_size.height = new_size
                    .height
                    .min(wgpu::Limits::downlevel_webgl2_defaults().max_texture_dimension_2d);
            }

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

    #[allow(unused_variables)]
    fn input(&mut self, event: &WindowEvent) -> bool {
        false
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

            self.volumetric_renderer.render_frame(&mut encoder, &view);
            self.selector_renderer.render_frame(&mut encoder, &view);

            {
                self.egui_renderer.begin_frame(window);

                egui::Window::new("fits3 cube viewer")
                    .resizable(true)
                    .vscroll(true)
                    .default_open(false)
                    .show(self.egui_renderer.context(), |ui| {
                        ui.label("This will host the UI of the viewer!");

                        /*if ui.button("Button!").clicked() {
                            println!("boom!")
                        }

                        ui.separator();
                        ui.horizontal(|ui| {
                            ui.label(format!(
                                "Pixels per point: {}",
                                self.egui_renderer.context().pixels_per_point()
                            ));
                            if ui.button("-").clicked() {
                                //state.scale_factor = (state.scale_factor - 0.1).max(0.3);
                            }
                            if ui.button("+").clicked() {
                                //state.scale_factor = (state.scale_factor + 0.1).min(3.0);
                            }
                        });*/
                    });

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
        // override the minmax value
        min: Option<f32>,
        max: Option<f32>,
    ) -> Result<(), &'static str> {
        let (new_cube, datamin, datamax, mincut, maxcut) = read_fits(reader, &self.device, &self.queue)?;

        // set the new datamin/datamax if there is some
        let datamin = min.or(datamin).unwrap_or(0.0);
        let datamax = max.or(datamax).unwrap_or(1.0);

        self.queue.write_buffer(
            &self.buffers["minmax"],
            0,
            bytemuck::bytes_of(&[datamin, datamax, 0.0_f32, 0.0_f32]),
        );

        // reset the cutoff values
        self.queue.write_buffer(
            &self.buffers["cuts"],
            0,
            bytemuck::bytes_of(&[mincut, maxcut, 0.0, 0.0]),
        );

        self.mincut = mincut;
        self.maxcut = maxcut;

        self.volumetric_renderer.set_volume(&self.device, &self.buffers, new_cube);

        Ok(())
    }
}

use std::ops::Range;
#[derive(Debug, Default)]
struct Params {
    perspective: Option<bool>,
    cuts: Option<Range<f32>>,
    data: Option<Vec<u8>>,
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

static mut PARAMS: Params = Params {
    perspective: None,
    cuts: None,
    data: None,
};

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
    delta: f64,
    theta: f64,
    dtheta: f64,
    ddelta: f64,
    dscale: f32,
    doffset: f32,
    scale: f32,
    offset: f32,

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
            delta: 0.0,
            theta: 0.0,
            dtheta: 0.0,
            ddelta: 0.0,
            dscale: 0.0,
            doffset: 0.0,

            scale: 1.0,
            offset: 0.0,
            i: 0,
        }
    }

    async fn set_window(&mut self, window: Window) {
        let window = Arc::new(window);
        let surface = self
            .instance
            .create_surface(window.clone())
            .expect("Failed to created the wgpu surface.");

        let mut state = State::new(
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
            let _ = state.visualize_cube(reader, None, None);
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

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let state = self.state
            .as_mut()
            .unwrap();
        #[cfg(target_arch = "wasm32")]
        if let Ok(data) = state.recv_data.try_recv() {
            let reader = Cursor::new(data.as_slice());
            match state.visualize_cube(reader, None, None) {
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
            }

            if let Some(cuts) = cuts {
                state.queue.write_buffer(
                    &state.buffers["cuts"],
                    0,
                    bytemuck::bytes_of(&[cuts.start, cuts.end, 0.0_f32, 0.0_f32]),
                );
            }

            if let Some(data) = data {
                let reader = Cursor::new(data.as_slice());
                match state.visualize_cube(reader, None, None) {
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

                let minmax = &MINMAX[self.i];
                let _ = state
                    .visualize_cube(reader, Some(minmax.start), Some(minmax.end));
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
                self.dtheta = 0.0;
                self.ddelta = 0.0;
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => {
                self.panning = false;
                self.theta += self.dtheta;
                self.delta += self.ddelta;

                self.delta = self.delta.clamp(
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
                self.dscale = 0.0;
                self.doffset = 0.0;
                self.offset = self.state.as_ref().unwrap().maxcut;
                self.scale = self.state.as_ref().unwrap().mincut;
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Right,
                ..
            } => {
                self.cuts = false;
                self.scale = self.scale * (1.0 + self.dscale);
                self.offset = self.offset * (1.0 + self.doffset);
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_pos = position;

                if self.panning {
                    let dx = (self.cursor_pos.x - self.start_cursor_pos.x)
                        / ((state.size.width as f64) * 0.5);
                    let dy = (self.cursor_pos.y - self.start_cursor_pos.y)
                        / ((state.size.height as f64) * 0.5);

                    self.dtheta = 2.0 * dx;
                    self.ddelta = dy;

                    let d = (self.delta as f32 + self.ddelta as f32).clamp(
                        -std::f32::consts::PI * 0.5 + 1e-3,
                        std::f32::consts::PI * 0.5 - 1e-3,
                    );

                    state.queue.write_buffer(
                        &state.buffers["cam_origin"],
                        0,
                        bytemuck::bytes_of(&[self.theta as f32 + self.dtheta as f32, d, 0.0, 0.0]),
                    );
                } else if self.cuts {
                    let dx =
                        (self.cursor_pos.x - self.start_cursor_pos.x) / ((state.size.width as f64) * 0.5);
                    let dy =
                        (self.cursor_pos.y - self.start_cursor_pos.y) / ((state.size.height as f64) * 0.5);

                    // between 0 and 1
                    self.dscale = dy as f32;
                    self.doffset = dx as f32;

                    state.queue.write_buffer(
                        &state.buffers["cuts"],
                        0,
                        bytemuck::bytes_of(&[
                            self.scale * (1.0 + self.dscale),
                            self.offset * (1.0 + self.doffset),
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

    // Winit prevents sizing with CSS, so we have to set
    // the size manually when on web.
    #[cfg(target_arch = "wasm32")]
    {
        use winit::dpi::LogicalSize;
        //let _ = window.request_inner_size(LogicalSize::new(768, 512));
    }

    event_loop.create_window(win_attrs).unwrap()
}


struct Cube<'a> {
    data: &'a [u8],
    dim: (u32, u32, u32),
    datamin: Option<f32>,
    datamax: Option<f32>,
    mincut: f32,
    maxcut: f32,
}

fn parse_fits_data_cube<'a, R>(fits: &'a mut Fits<Cursor<R>>) -> Result<Cube<'a>, &'static str>
where
    R: AsRef<[u8]> + std::fmt::Debug + 'a,
{
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

                    let mincut = 0.0;
                    let maxcut = 1.0;

                    let mut data = image.raw_bytes();

                    let cuts = match b {
                        -32 => {
                            use std::convert::TryInto;
                            let mut floats: Vec<f32> = data
                                .chunks_exact(4)
                                .map(|b| f32::from_be_bytes(b.try_into().unwrap()))
                                .collect();

                            first_and_last_percent_f32(&mut floats, 5, 95)
                        }
                        _ => todo!()
                    };


                    let datamin = if let Some(Value::Float { value, .. }) = header.get("DATAMIN") {
                        Some(*value as f32)
                    } else {
                        None
                    };
                    let datamax = if let Some(Value::Float { value, .. }) = header.get("DATAMAX") {
                        Some(*value as f32)
                    } else {
                        None
                    };

                    Ok(Cube {
                        data,
                        dim: (d1, d2, d3),
                        datamin,
                        datamax,
                        mincut: cuts.start,
                        maxcut: cuts.end
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

use std::fmt::Debug;
fn read_fits<R: AsRef<[u8]> + Debug>(
    reader: Cursor<R>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<(Texture, Option<f32>, Option<f32>, f32, f32), &'static str> {
    let mut fits = Fits::from_reader(reader);
    let Cube {
        data: raw_bytes,
        dim,
        datamin,
        datamax,
        mincut,
        maxcut
    } = parse_fits_data_cube(&mut fits)?;

    Ok((
        Texture::from_raw_bytes::<f32>(&device, &queue, Some(raw_bytes), dim, 4, "cube")?,
        datamin,
        datamax,
        mincut,
        maxcut
    ))
}

trait EndianConvert {
    fn to_be(self) -> Self;
    fn to_le(self) -> Self;
}

macro_rules! impl_endian {
    ($($t:ty),*) => {
        $(
            impl EndianConvert for $t {
                fn to_be(self) -> Self { <$t>::to_be(self) }
                fn to_le(self) -> Self { <$t>::to_le(self) }
            }
        )*
    };
}

impl_endian!(u16, u32, u64, i16, i32, i64, f32, f64);

use std::cmp::Ordering;
pub fn first_and_last_percent_f32(
    slice: &mut [f32],
    mut first_percent: i32,
    mut last_percent: i32,
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

    let n = valid.len();
    let i1 = (first_percent.clamp(0, 100) as usize * valid_len) / 100;
    let i2 = (last_percent.clamp(0, 100) as usize * valid_len) / 100;

    let min_val = {
        let (_, min_val, _) =
            valid.select_nth_unstable_by(i1, |a, b| {
                //let a = a.to_be();
                //let b = b.to_be();

                a.total_cmp(&b)
            });
        *min_val
    };
    let max_val = {
        let (_, max_val, _) =
            valid.select_nth_unstable_by(i2, |a, b| {
                //let a = a.to_be();
                //let b = b.to_be();

                a.total_cmp(&b)
            });
        *max_val
    };

    min_val..max_val
}
