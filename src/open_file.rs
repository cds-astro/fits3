#[cfg(target_arch = "wasm32")]
use web_sys::{HtmlInputElement, Event};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{closure::Closure, JsCast};

#[cfg(target_arch = "wasm32")]
pub(crate) fn open_file_dialog(send_data: async_channel::Sender<Vec<u8>>) {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();

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
                web_sys::console::log_1(&file.name().into());
                let reader = web_sys::FileReader::new().unwrap();

                let reader_cloned = reader.clone();
                let sd = send_data.clone();
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

    input.set_onchange(Some(onchange.as_ref().unchecked_ref()));
    onchange.forget(); // prevent it from being dropped

    // Trigger dialog
    input.click();
}