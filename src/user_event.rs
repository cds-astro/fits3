use std::cell::RefCell;
use winit::event_loop::{EventLoop, EventLoopProxy};

thread_local! {
    static PROXY: RefCell<Option<EventLoopProxy<UserEvent>>> =
        const { RefCell::new(None) };
}

pub enum UserEvent {
    DisplayData,
}

pub fn create_proxy(event_loop: &EventLoop<UserEvent>) {
    let proxy = event_loop.create_proxy();

    PROXY.with(|p| {
        *p.borrow_mut() = Some(proxy.clone());
    });
}

#[cfg(target_arch = "wasm32")]
pub fn send_event(event: UserEvent) {
    PROXY.with(|p| {
        if let Some(proxy) = &*p.borrow() {
            proxy.send_event(event).ok();
        }
    });
}
