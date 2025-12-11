mod asset;
mod camera;
mod context;
mod fps_counter;
mod frame;
mod input;
mod model;
mod parser;
mod renderer;
mod scene;
mod state;
mod vertex;
mod vulkan;

use crate::asset::AssetManager;
use crate::camera::Camera;
use crate::context::RenderContext;
use crate::fps_counter::FpsCounter;
use crate::renderer::Renderer;
use crate::scene::Scene;
use crate::state::State;
use env_logger::Env;
use glam::Vec3;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::sync::Arc;
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

    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("Usage: vulkan-engine [OPTIONS]");
        println!("    -h --help          display this help and exit");
        println!("    -v --validation    use vulkan validation layers");
        std::process::exit(0);
    }

    let validation = args.iter().any(|a| a == "-v" || a == "--validation");

    let mut event_loop = EventLoop::new().expect("Failed to create event loop");
    let mut state = State::new("Vulkan", 1280, 720);

    state.update(&mut event_loop);

    let window = state.window.as_ref().expect("Failed to get window");

    let ctx = Arc::new(RenderContext::new(
        window
            .display_handle()
            .expect("Failed to get display handle")
            .as_raw(),
        window
            .window_handle()
            .expect("Failed to get window handle")
            .as_raw(),
        validation,
    ));
    state.ctx = Some(ctx.clone());

    let renderer = Renderer::new(state.width, state.height, ctx.clone());
    state.renderer = Some(renderer);

    let mut fps_counter = FpsCounter::default().log_fps(Some(1000));

    let asset = AssetManager::new(
        state.ctx.clone().expect("Failed to find RenderContext"),
        state
            .ctx
            .as_ref()
            .expect("Failed to find GPU allocator in RenderContext")
            .allocator
            .clone(),
    );

    let mut meshes = Vec::new();
    meshes.push(asset.load_gltf(include_bytes!("../resources/models/sponza.glb")));

    let mut camera = Camera::default();
    camera.pos = Vec3::new(0.0, 0.0, 2.0);

    let scene = Scene {
        models: meshes,
        camera,
    };
    state.scene = scene;

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

        state.scene.camera = camera;

        state.input.reset_mouse_motion();

        state.update(&mut event_loop);

        fps_counter.end();
    }
}
