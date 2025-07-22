mod fps_counter;
mod renderer;
mod state;
mod vulkan;

use crate::fps_counter::FpsCounter;
use crate::state::State;
use env_logger::Env;
use winit::event_loop::EventLoop;

fn main() {
    let env = Env::default().default_filter_or(if cfg!(debug_assertions) {
        "debug"
    } else {
        "info"
    });
    env_logger::Builder::from_env(env).init();

    let mut event_loop = EventLoop::new().unwrap();
    let mut state = State::new("Vulkan", 800, 600);
    let mut fps_counter = FpsCounter::default().log_fps(Some(1000));

    while state.is_running() {
        fps_counter.begin();
        state.update(&mut event_loop);
        fps_counter.end();
    }
}
