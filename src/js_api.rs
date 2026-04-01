
use crate::user_event::UserEvent;
use wasm_bindgen::prelude::wasm_bindgen;
use winit::window::Window;
use crate::uniform::Scene;
use crate::State;
use std::mem::offset_of;
use std::ops::Range;

#[derive(Debug, Default)]
pub(crate) struct Params {
    pub perspective: Option<bool>,
    pub cuts: Option<Range<f32>>,
    pub data: Option<Vec<u8>>,
}

thread_local! {
    pub(crate) static ONSELECT: std::cell::RefCell<Option<js_sys::Function>> =
    std::cell::RefCell::new(None);
}

use lazy_static::lazy_static;
lazy_static! {
    pub(crate) static ref CHANNEL_PARAMS: (
        async_channel::Sender<Params>,
        async_channel::Receiver<Params>,
    ) = async_channel::unbounded::<Params>();
}

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

    // Wake up the winit window event
    crate::user_event::send_event(UserEvent::DisplayData);
}
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

    // Wake up the winit window event
    crate::user_event::send_event(UserEvent::DisplayData);
}

#[wasm_bindgen(js_name = "onselect")]
pub fn onselect(func: js_sys::Function) {
    ONSELECT.with(|f| {
        *f.borrow_mut() = Some(func);
    });
}

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

    // Wake up the winit window event
    crate::user_event::send_event(UserEvent::DisplayData);
}

pub(crate) fn handle_events(state: &mut State, window: &Window) {
    if let Ok(params) = CHANNEL_PARAMS.1.try_recv() {
        let Params {
            perspective,
            cuts,
            data,
            ..
        } = params;
        
        if let Some(perspective) = perspective {
            state.queue.write_buffer(
            &state.buffers["scene"],
            offset_of!(Scene, perspective) as wgpu::BufferAddress,
            bytemuck::bytes_of(&[
                if perspective { 1.0_f32 } else { 0.0_f32 },
                ]),
            );
            
            state.perspective = perspective;
            window.request_redraw();
        }
            
        if let Some(cuts) = cuts {
            state.queue.write_buffer(
                &state.buffers["render_params"],
                0,
                bytemuck::bytes_of(&[cuts.start, cuts.end]),
            );
            
            state.min_cut = cuts.start;
            state.max_cut = cuts.end;
            
            window.request_redraw();
        }
            
        if let Some(data) = data {
            use std::io::Cursor;
            
            let reader = Cursor::new(data.as_slice());
            match state.visualize_cube(reader) {
                Ok(()) => {}
                Err(error) => web_sys::window()
                .unwrap()
                .alert_with_message(error)
                .unwrap(),
            }
            
            window.request_redraw();
        }
    }
}