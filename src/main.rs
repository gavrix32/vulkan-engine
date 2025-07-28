mod camera;
mod fps_counter;
mod input;
mod renderer;
mod state;
mod vulkan;

use crate::camera::Camera;
use crate::fps_counter::FpsCounter;
use crate::state::State;
use env_logger::Env;
use glam::Vec3;
use winit::event::MouseButton;
use winit::event_loop::EventLoop;
use winit::keyboard::KeyCode;

fn main() {
    let env = Env::default().default_filter_or(if cfg!(debug_assertions) {
        "debug"
    } else {
        "info"
    });
    env_logger::Builder::from_env(env).init();

    let mut event_loop = EventLoop::new().unwrap();
    let mut state = State::new("Vulkan", 1280, 720);
    let mut fps_counter = FpsCounter::default().log_fps(Some(1000));

    let mut camera = Camera::default();
    camera.pos = Vec3::new(0.0, 0.0, 2.0);

    while state.is_running() {
        fps_counter.begin();

        if state.input.released_once_keys.contains(&KeyCode::Escape) {
            break;
        }

        let mut speed = fps_counter.delta.as_secs_f32() * 5.0;
        if state.input.pressed_keys.contains(&KeyCode::ControlLeft) {
            speed *= 2.0;
        }

        for key in &state.input.pressed_keys {
            match key {
                KeyCode::KeyW => camera.move_local_z(speed),
                KeyCode::KeyA => camera.move_local_x(-speed),
                KeyCode::KeyS => camera.move_local_z(-speed),
                KeyCode::KeyD => camera.move_local_x(speed),
                KeyCode::Space => camera.move_local_y(speed),
                KeyCode::ShiftLeft => camera.move_local_y(-speed),
                _ => {}
            }
        }

        let sensitivity = 0.01;

        if state.input.pressed_buttons.contains(&MouseButton::Left) {
            camera.set_euler_rot(
                camera.euler_rot().0 - state.input.mouse_motion.0 as f32 * sensitivity,
                camera.euler_rot().1 - state.input.mouse_motion.1 as f32 * sensitivity,
                0.0,
            );
        }

        if let Some(renderer) = &mut state.renderer {
            renderer.camera_view = camera.view();
        }

        state.update(&mut event_loop);

        fps_counter.end();
    }
}
