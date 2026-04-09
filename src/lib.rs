extern crate byte_slice_cast;
#[cfg(target_arch = "wasm32")]
extern crate console_error_panic_hook;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use winit::event_loop::ControlFlow;

use crate::state::State;

use crate::states::FiniteStateMachine;

use crate::states::Mode;

use uniform::Scene;

use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Fullscreen, Window, WindowId},
};

use crate::cube::Cube;
use texture::Texture;
use time::Clock;
use user_event::UserEvent;
use vertex::{Vertex, VertexNDC};

#[cfg(target_arch = "wasm32")]
use js_api::*;

#[cfg(not(target_arch = "wasm32"))]
use memmap2::Mmap;
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
#[cfg(not(target_arch = "wasm32"))]
use std::io::Cursor;

use crate::short_keys::ShortKeyCommands;

use crate::input::Input;
use std::sync::Arc;

mod cube;
mod gui;
mod renderer;
mod save_file;
mod input;
#[cfg(target_arch = "wasm32")]
mod js_api;
mod math;
mod moment;
mod open_file;
mod short_keys;
mod state;
mod states;
mod texture;
mod time;
mod uniform;
mod user_event;
mod vertex;

#[cfg(not(target_arch = "wasm32"))]
const CUBES_PATH: &[&str] = &[
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

pub struct App {
    instance: wgpu::Instance,
    state: Option<State>,
    window: Option<Arc<Window>>,

    input: Input,

    mode: Mode,

    pub needs_redraw: bool,

    shortcuts: ShortKeyCommands,

    #[cfg(not(target_arch = "wasm32"))]
    i: usize,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
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

        let input = Input::new();
        let mode = Mode::new(&input);

        Self {
            instance,
            state: None,
            window: None,
            input,

            mode,
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

        let state = State::new(&window, &self.instance, surface).await;

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
                window.request_redraw(); // ✅ trigger render
            }
        }
    }

    #[allow(unused_variables)]
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let state = self.state.as_mut().unwrap();

        let window = self.window.as_ref().unwrap();

        #[cfg(target_arch = "wasm32")]
        js_api::handle_events(state, window);

        // let egui render to process the event first
        let mut consumed_by_ui = false;
        let response = state
            .egui_renderer
            .handle_input(self.window.as_ref().unwrap(), &event);

        if let egui_winit::EventResponse { consumed: true, .. } = response {
            // an action has been done on the ui so we must redraw things.
            window.request_redraw();
            consumed_by_ui = true;
        }

        let pointer_over_egui = state.egui_renderer.context().is_pointer_over_area();

        self.input.update_event(&event);

        /*if pointer_over_egui {
            state.set_cursor(egui::CursorIcon::Default);
        }*/

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
            } => {
                if state.is_fullscreen {
                    state.toggle_fullscreen(window);
                } else {
                    event_loop.exit();
                }
            }
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

                let file = File::open(CUBES_PATH[self.i]).unwrap();
                let mmap = unsafe { Mmap::map(&file).unwrap() };

                let reader = Cursor::new(mmap);

                let _ = state.visualize_cube(reader);

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
                window.set_fullscreen(Some(Fullscreen::Borderless(None)));

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
                state.set_camera_position(std::f64::consts::PI, 0.0, window);
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
                let theta = state.theta - std::f64::consts::PI / 4.0;
                state.set_camera_position(theta, state.delta, window);
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
                let theta = state.theta + std::f64::consts::PI / 4.0;
                state.set_camera_position(theta, state.delta, window);
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
                let delta = (state.delta + std::f64::consts::PI / 4.0).clamp(
                    -std::f64::consts::PI * 0.5 + 1e-3,
                    std::f64::consts::PI * 0.5 - 1e-3,
                );
                state.set_camera_position(state.theta, delta, window);
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
                let delta = (state.delta - std::f64::consts::PI / 4.0).clamp(
                    -std::f64::consts::PI * 0.5 + 1e-3,
                    std::f64::consts::PI * 0.5 - 1e-3,
                );
                state.set_camera_position(state.theta, delta, window);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key:
                            PhysicalKey::Code(KeyCode::KeyF) | PhysicalKey::Code(KeyCode::Numpad2),
                        ..
                    },
                ..
            } => {
                state.set_camera_position(std::f64::consts::PI, 0.0, window);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key:
                            PhysicalKey::Code(KeyCode::KeyB) | PhysicalKey::Code(KeyCode::Numpad8),
                        ..
                    },
                ..
            } => {
                state.set_camera_position(0.0, 0.0, window);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key:
                            PhysicalKey::Code(KeyCode::KeyL) | PhysicalKey::Code(KeyCode::Numpad4),
                        ..
                    },
                ..
            } => {
                state.set_camera_position(-std::f64::consts::PI * 0.5, 0.0, window);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key:
                            PhysicalKey::Code(KeyCode::KeyR) | PhysicalKey::Code(KeyCode::Numpad6),
                        ..
                    },
                ..
            } => {
                state.set_camera_position(std::f64::consts::PI * 0.5, 0.0, window);
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key:
                            PhysicalKey::Code(KeyCode::KeyT) | PhysicalKey::Code(KeyCode::Numpad0),
                        ..
                    },
                ..
            } => {
                state.set_camera_position(
                    std::f64::consts::PI,
                    std::f64::consts::PI * 0.5 - 1e-3,
                    window,
                );
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
                state.set_camera_position(
                    std::f64::consts::PI,
                    -std::f64::consts::PI * 0.5 + 1e-3,
                    window,
                );
            }
            WindowEvent::KeyboardInput { event, .. } => {
                // Ctrl+O
                self.shortcuts.process_key_event(
                    PhysicalKey::Code(KeyCode::KeyO),
                    &event,
                    true,
                    || {
                        state.show_options = !state.show_options;

                        window.request_redraw();
                    },
                );
            }
            WindowEvent::Resized(physical_size) => {
                state.resize(physical_size, window);
            }
            WindowEvent::RedrawRequested => {
                let window = self.window.as_ref().unwrap();
                let _ = state.render(window);

                if state.needs_redraw {
                    state.needs_redraw = false;

                    window.request_redraw();
                }
            }
            WindowEvent::CursorEntered { .. } => {
                window.request_redraw(); // ✅ force a frame
            }
            WindowEvent::CursorMoved { .. } => {
                window.request_redraw();
            }
            WindowEvent::MouseWheel { .. } => {
                window.request_redraw();
            }
            _ => {}
        }

        self.mode = self.mode.clone().verify_transitions(
            state,
            &self.input,
            pointer_over_egui,
            consumed_by_ui,
            window,
        );

        // end of frame
        self.input.end_frame();
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
    let _proxy = event_loop.create_proxy();

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
