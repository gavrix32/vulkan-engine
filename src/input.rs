use std::collections::{HashSet, VecDeque};
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, KeyEvent, MouseButton};
use winit::keyboard::{KeyCode, PhysicalKey};

#[derive(Default)]
pub struct Input {
    pub keyboard_event_queue: VecDeque<KeyEvent>,
    pub pressed_keys: HashSet<KeyCode>,
    pub pressed_once_keys: HashSet<KeyCode>,
    pub released_once_keys: HashSet<KeyCode>,

    pub mouse_button_event_queue: VecDeque<(ElementState, MouseButton)>,
    pub pressed_buttons: HashSet<MouseButton>,
    pub pressed_once_buttons: HashSet<MouseButton>,
    pub released_once_buttons: HashSet<MouseButton>,

    pub mouse_motion_event_queue: VecDeque<(f64, f64)>,
    pub mouse_motion: (f64, f64),
    pub cursor_pos: PhysicalPosition<f64>,
}

impl Input {
    pub fn reset_once_keys(&mut self) {
        self.pressed_once_keys.clear();
        self.released_once_keys.clear();
    }

    pub fn send_keyboard_event(&mut self, event: KeyEvent) {
        let (state, physical_key, repeat) = (event.state, event.physical_key, event.repeat);

        if let PhysicalKey::Code(key_code) = physical_key {
            match state {
                ElementState::Pressed => {
                    self.pressed_keys.insert(key_code);
                    if !repeat {
                        self.pressed_once_keys.insert(key_code);
                    }
                }
                ElementState::Released => {
                    self.pressed_keys.remove(&key_code);
                    if !repeat {
                        self.released_once_keys.insert(key_code);
                    }
                }
            }
        }
    }

    pub fn reset_once_buttons(&mut self) {
        self.pressed_once_buttons.clear();
        self.released_once_buttons.clear();
    }

    pub fn send_mouse_button_event(&mut self, state: ElementState, button: MouseButton) {
        match state {
            ElementState::Pressed => {
                self.pressed_buttons.insert(button);
                self.pressed_once_buttons.insert(button);
            }
            ElementState::Released => {
                self.pressed_buttons.remove(&button);
                self.released_once_buttons.insert(button);
            }
        }
    }

    pub fn reset_mouse_motion(&mut self) {
        self.mouse_motion = (0.0, 0.0);
    }

    pub fn send_mouse_motion_event(&mut self, motion: (f64, f64)) {
        self.mouse_motion = motion;
    }
}
