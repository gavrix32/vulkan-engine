use crate::vulkan::adapter::Adapter;
use crate::vulkan::device::Device;
use crate::vulkan::instance::Instance;
use crate::vulkan::surface::Surface;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::sync::Arc;

pub struct RenderContext {
    pub device: Arc<Device>,
    pub adapter: Adapter,
    pub surface: Surface,
    pub instance: Instance,
}

impl RenderContext {
    pub fn new(
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        validation: bool,
    ) -> Self {
        let instance = Instance::new(display_handle, validation);
        let surface = Surface::new(&instance, display_handle, window_handle);
        let adapter = Adapter::new(&instance, &surface);
        let device = Arc::new(Device::new(&instance, &adapter));

        RenderContext {
            device,
            adapter,
            surface,
            instance,
        }
    }
}
