use crate::renderer::Renderer;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::time::Duration;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::pump_events::{EventLoopExtPumpEvents, PumpStatus};
use winit::window::{Window, WindowAttributes, WindowId};

pub struct State {
    title: String,
    width: u32,
    height: u32,
    pub status: PumpStatus,
    window: Option<Window>,
    renderer: Option<Renderer>,
}

impl State {
    pub fn new(title: &str, width: u32, height: u32) -> Self {
        Self {
            title: title.to_string(),
            width,
            height,
            status: PumpStatus::Continue,
            window: None,
            renderer: None,
        }
    }

    pub fn update(&mut self, event_loop: &mut EventLoop<()>) {
        self.status = event_loop.pump_app_events(Some(Duration::ZERO), self);
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    pub fn is_running(&self) -> bool {
        match self.status {
            PumpStatus::Exit(_) => false,
            _ => true,
        }
    }
}

impl ApplicationHandler for State {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attribs = WindowAttributes::default()
            .with_title(self.title.as_str())
            .with_inner_size(PhysicalSize::new(self.width, self.height));

        let window = event_loop.create_window(window_attribs).unwrap();
        let renderer = Renderer::new(
            self.width,
            self.height,
            window.display_handle().unwrap().as_raw(),
            window.window_handle().unwrap().as_raw(),
        );

        self.window = Some(window);
        self.renderer = Some(renderer);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.draw_frame();
                }
            }
            WindowEvent::Resized(resolution) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.framebuffer_resized = true;
                    renderer.width = resolution.width;
                    renderer.height = resolution.height;
                }
            }
            _ => (),
        }
    }
}
