use crate::context::RenderContext;
use crate::input::Input;
use crate::renderer::Renderer;
use crate::scene::Scene;
use std::sync::Arc;
use std::time::Duration;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, DeviceId, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::pump_events::{EventLoopExtPumpEvents, PumpStatus};
use winit::window::{Window, WindowAttributes, WindowId};

pub struct State {
    title: String,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub status: PumpStatus,
    pub window: Option<Window>,
    pub renderer: Option<Renderer>,
    pub scene: Scene,
    pub ctx: Option<Arc<RenderContext>>,
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
            ctx: None,
            renderer: None,
            scene: Scene::default(),
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
        if self.window.is_some() {
            return;
        }

        let window_attribs = WindowAttributes::default()
            .with_title(self.title.as_str())
            .with_inner_size(PhysicalSize::new(self.width, self.height));

        let window = event_loop
            .create_window(window_attribs)
            .expect("Failed to create window");

        self.window = Some(window);
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
                    renderer.draw_frame(&self.scene);
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
            WindowEvent::MouseInput { state, button, .. } => self
                .input
                .mouse_button_event_queue
                .push_back((state, button)),
            WindowEvent::CursorMoved { position, .. } => self.input.cursor_pos = position,
            _ => (),
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        match event {
            DeviceEvent::MouseMotion { delta } => self.input.send_mouse_motion_event(delta),
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
    }
}
