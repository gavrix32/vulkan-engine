use crate::unsafe_vk_try;
use crate::vulkan::device::Device;
use ash::vk;
use std::sync::Arc;

pub struct Semaphore {
    device: Arc<Device>,
    pub handle: vk::Semaphore,
}

impl Semaphore {
    pub fn new(device: Arc<Device>) -> Self {
        let semaphore_create_info = vk::SemaphoreCreateInfo::default();

        let handle = unsafe_vk_try!(device.handle.create_semaphore(&semaphore_create_info, None));

        Self { device, handle }
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_semaphore(self.handle, None);
        }
    }
}

pub struct Fence {
    device: Arc<Device>,
    pub handle: vk::Fence,
}

impl Fence {
    pub fn new(device: Arc<Device>, signaled: bool) -> Self {
        let signaled_flag = if signaled {
            vk::FenceCreateFlags::SIGNALED
        } else {
            vk::FenceCreateFlags::default()
        };

        let fence_create_info = vk::FenceCreateInfo::default().flags(signaled_flag);

        let handle = unsafe_vk_try!(device.handle.create_fence(&fence_create_info, None));

        Self { device, handle }
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_fence(self.handle, None);
        }
    }
}
