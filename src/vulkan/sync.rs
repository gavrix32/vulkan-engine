use crate::unsafe_vk_try;
use crate::vulkan::device::Device;
use ash::vk;
use std::sync::Arc;

pub struct Semaphore {
    device: Arc<Device>,
    pub vk_semaphore: vk::Semaphore,
}

impl Semaphore {
    pub fn new(device: Arc<Device>) -> Self {
        let semaphore_create_info = vk::SemaphoreCreateInfo::default();

        let vk_semaphore = unsafe_vk_try!(
            device
                .ash_device
                .create_semaphore(&semaphore_create_info, None)
        );

        Self {
            device,
            vk_semaphore,
        }
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe {
            self.device
                .ash_device
                .destroy_semaphore(self.vk_semaphore, None);
        }
    }
}

pub struct Fence {
    device: Arc<Device>,
    pub vk_fence: vk::Fence,
}

impl Fence {
    pub fn new(device: Arc<Device>, signaled: bool) -> Self {
        let signaled_flag = if signaled {
            vk::FenceCreateFlags::SIGNALED
        } else {
            vk::FenceCreateFlags::default()
        };

        let fence_create_info = vk::FenceCreateInfo::default().flags(signaled_flag);

        let vk_fence = unsafe_vk_try!(device.ash_device.create_fence(&fence_create_info, None));

        Self { device, vk_fence }
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe {
            self.device.ash_device.destroy_fence(self.vk_fence, None);
        }
    }
}
