use crate::input::Input;
use crate::renderer::Renderer;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::time::Duration;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::pump_events::{EventLoopExtPumpEvents, PumpStatus};
use winit::window::{Window, WindowAttributes, WindowId};

pub struct State {
    title: String,
    width: u32,
    height: u32,
    pub status: PumpStatus,
    window: Option<Window>,
    pub renderer: Option<Renderer>,
    pub input: Input,
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
            input: Input::default(),
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
            WindowEvent::KeyboardInput { event, .. } => {
                self.input.keyboard_event_queue.push_back(event)
            }
            WindowEvent::MouseInput { state, button, .. } => {
                self.input.mouse_button_event_queue.push_back((state, button))
            }
            WindowEvent::CursorMoved { position, .. } => self.input.cursor_pos = position,
            _ => (),
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        match event {
            DeviceEvent::MouseMotion { delta } => self.input.mouse_motion_event_queue.push_back(delta),
            _ => (),
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        self.input.reset_once_keys();
        if let Some(event) = self.input.keyboard_event_queue.pop_front() {
            self.input.send_keyboard_event(event);
        }

        self.input.reset_once_buttons();
        if let Some((state, button)) = self.input.mouse_button_event_queue.pop_front() {
            self.input.send_mouse_button_event(state, button);
        }

        self.input.reset_mouse_motion();
        if let Some(motion) = self.input.mouse_motion_event_queue.pop_front() {
            self.input.send_mouse_motion_event(motion);
        }
    }
}
