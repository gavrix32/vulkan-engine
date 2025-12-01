use crate::vulkan::adapter::Adapter;
use crate::vulkan::device::Device;
use crate::vulkan::encoder::Encoder;
use crate::vulkan::instance::Instance;
use crate::vulkan::surface::Surface;
use crate::vulkan::swapchain::Swapchain;
use crate::vulkan::sync;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::sync::Arc;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct RenderContext {
    // TODO: Rename or move to FrameContext
    pub in_flight_fences: Vec<sync::Fence>,
    pub image_available_semaphores: Vec<sync::Semaphore>,
    pub render_finished_semaphores: Vec<sync::Semaphore>,

    pub encoder: Encoder,
    pub swapchain: Swapchain,
    pub device: Arc<Device>,
    pub adapter: Adapter,
    pub surface: Surface,
    pub instance: Instance,
}

impl RenderContext {
    pub fn new(
        width: u32,
        height: u32,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        validation: bool,
    ) -> Self {
        let instance = Instance::new(display_handle, validation);
        let surface = Surface::new(&instance, display_handle, window_handle);
        let adapter = Adapter::new(&instance, &surface);
        let device = Arc::new(Device::new(&instance, &adapter));
        let msaa_samples = adapter.max_usable_sample_count(&instance);

        let swapchain = Swapchain::new(
            &instance,
            &adapter,
            device.clone(),
            &surface,
            width,
            height,
            msaa_samples,
        );

        let swapchain_image_count = swapchain.images.len();

        let encoder = Encoder::new(device.clone(), &adapter, MAX_FRAMES_IN_FLIGHT);

        let mut image_available_semaphores: Vec<sync::Semaphore> =
            Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut render_finished_semaphores: Vec<sync::Semaphore> =
            Vec::with_capacity(swapchain_image_count);
        let mut in_flight_fences: Vec<sync::Fence> = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            image_available_semaphores.push(sync::Semaphore::new(device.clone()));
            in_flight_fences.push(sync::Fence::new(device.clone(), true));
        }
        for _ in 0..swapchain_image_count {
            render_finished_semaphores.push(sync::Semaphore::new(device.clone()));
        }

        RenderContext {
            in_flight_fences,
            image_available_semaphores,
            render_finished_semaphores,
            encoder,
            swapchain,
            device,
            adapter,
            surface,
            instance,
        }
    }
}
