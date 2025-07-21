use log::info;
use std::ops::Add;
use std::time::{Duration, Instant};

pub struct FpsCounter {
    last_time: Instant,
    pub delta: Duration,
    elapsed: Duration,
    log_interval: Option<u128>,
}

impl Default for FpsCounter {
    fn default() -> Self {
        Self {
            last_time: Instant::now(),
            delta: Duration::default(),
            elapsed: Duration::default(),
            log_interval: None,
        }
    }
}

impl FpsCounter {
    pub fn log_fps(mut self, interval_millis: Option<u128>) -> Self {
        self.log_interval = interval_millis;
        self
    }

    pub fn begin(&mut self) {
        let now = Instant::now();
        self.delta = now.duration_since(self.last_time);
        self.last_time = now;
        self.elapsed = self.elapsed.add(self.delta);
    }

    pub fn end(&mut self) {
        if let Some(log_interval) = self.log_interval {
            if self.elapsed.as_millis() >= log_interval {
                let delta_time = self.delta.as_secs_f64();
                let fps = (1.0 / delta_time) as u32;
                self.elapsed = Duration::ZERO;

                info!("{} fps ({:.2} ms)", fps, delta_time * 1000.0);
            }
        }
    }
}
