use crate::unsafe_vk_try;
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
        let surface_instance = khr::surface::Instance::new(&instance.entry, &instance.handle);
        let surface_khr = unsafe_vk_try!(ash_window::create_surface(
            &instance.entry,
            &instance.handle,
            display_handle,
            window_handle,
            None,
        ));

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
