mod debug;
mod vulkan_state;

use crate::vulkan_state::VulkanState;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::time::Duration;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::pump_events::{EventLoopExtPumpEvents, PumpStatus};
use winit::window::{Window, WindowAttributes, WindowId};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 800;

#[derive(Default)]
struct App {
    window: Option<Window>,
    state: Option<VulkanState>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attribs = WindowAttributes::default()
            .with_title("Vulkan")
            .with_inner_size(PhysicalSize::new(WIDTH, HEIGHT));

        let window = event_loop.create_window(window_attribs).unwrap();
        let raw_display_handle = window.display_handle().unwrap().as_raw();
        let raw_window_handle = window.window_handle().unwrap().as_raw();

        self.window = Some(window);
        self.state = Some(VulkanState::new(
            WIDTH,
            HEIGHT,
            raw_display_handle,
            raw_window_handle,
        ));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let window = match self.window.as_ref() {
            Some(window) => window,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                window.pre_present_notify();
                // TODO! render and present
            }
            _ => (),
        }
    }
}

fn main() {
    env_logger::init();

    let mut event_loop = EventLoop::new().unwrap();
    let mut app = App::default();

    // let mut last_time = Instant::now();

    loop {
        // let now = Instant::now();
        // let delta_time = now.duration_since(last_time);
        // last_time = now;

        let status = event_loop.pump_app_events(Some(Duration::ZERO), &mut app);

        if let PumpStatus::Exit(_) = status {
            break;
        }

        if let Some(window) = &app.window {
            window.request_redraw();
        }

        // let fps = 1.0 / delta_time.as_secs_f64();
        // println!("{} FPS", fps);
        break;
    }
}
