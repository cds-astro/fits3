use winit::event_loop::{EventLoop, EventLoopProxy};
use std::cell::RefCell;

thread_local! {
    static PROXY: RefCell<Option<EventLoopProxy<UserEvent>>> =
        RefCell::new(None);
}

pub(crate) enum UserEvent {
    DisplayData,
}

pub(crate) fn create_proxy(event_loop: &EventLoop<UserEvent>) {
    let proxy = event_loop.create_proxy();

    PROXY.with(|p| {
        *p.borrow_mut() = Some(proxy.clone());
    });
}

pub(crate) fn send_event(event: UserEvent) {
    PROXY.with(|p| {
        if let Some(proxy) = &*p.borrow() {
            proxy.send_event(event).ok();
        }
    });
}




