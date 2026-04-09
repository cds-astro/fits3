#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{closure::Closure, JsCast};
#[cfg(target_arch = "wasm32")]
use web_sys::{Event, HtmlInputElement};

#[cfg(target_arch = "wasm32")]
use crate::{Params, CHANNEL_PARAMS};

#[cfg(target_arch = "wasm32")]
use crate::UserEvent;

#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
#[cfg(target_arch = "wasm32")]
use std::rc::Rc;

#[cfg(target_arch = "wasm32")]
pub(crate) fn open_file_dialog(picked_file: Rc<RefCell<Option<String>>>) {
    let document = web_sys::window().unwrap().document().unwrap();

    let input: HtmlInputElement = document
        .create_element("input")
        .unwrap()
        .dyn_into()
        .unwrap();

    input.set_type("file");

    // Optional: accept filter
    input.set_accept(".fits");

    // Handle file selection
    let onchange = Closure::wrap(Box::new(move |event: Event| {
        let input: HtmlInputElement = event.target().unwrap().dyn_into().unwrap();

        if let Some(files) = input.files() {
            if let Some(file) = files.get(0) {
                let file_name = file.name();
                let reader = web_sys::FileReader::new().unwrap();

                let reader_cloned = reader.clone();
                *picked_file.borrow_mut() = Some(file_name);
                let onloadend_cb = Closure::wrap(Box::new(move |_: web_sys::Event| {
                    let result = reader_cloned.result().unwrap();
                    let array = js_sys::Uint8Array::new(&result);

                    wasm_bindgen_futures::spawn_local(async move {
                        let data = array.to_vec();

                        // Wake up the winit window event
                        crate::user_event::send_event(UserEvent::DisplayData);

                        CHANNEL_PARAMS
                            .0
                            .send(Params {
                                data: Some(data),
                                ..Default::default()
                            })
                            .await
                            .unwrap();
                    });
                }) as Box<dyn FnMut(_)>);

                reader.set_onloadend(Some(onloadend_cb.as_ref().unchecked_ref()));
                reader.read_as_array_buffer(&file).unwrap();
                onloadend_cb.forget(); // prevent drop
            }
        }
    }) as Box<dyn FnMut(_)>);

    input.set_onchange(Some(onchange.as_ref().unchecked_ref()));
    onchange.forget(); // prevent it from being dropped

    // Trigger dialog
    input.click();
}
