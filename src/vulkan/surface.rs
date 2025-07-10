use crate::vulkan::instance::Instance;
use ash::{khr, vk};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

pub struct Surface {
    pub surface_instance: khr::surface::Instance,
    pub surface_khr: vk::SurfaceKHR,
}

impl Surface {
    pub fn new(
        instance: &Instance,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> Self {
        let surface_instance = khr::surface::Instance::new(&instance.entry, &instance.ash_instance);
        let surface_khr = unsafe {
            ash_window::create_surface(
                &instance.entry,
                &instance.ash_instance,
                display_handle,
                window_handle,
                None,
            )
        }
        .expect("Failed to create window surface");

        Self {
            surface_instance,
            surface_khr,
        }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            self.surface_instance
                .destroy_surface(self.surface_khr, None);
        }
    }
}
