use crate::unsafe_vk_try;
use crate::vulkan::adapter::Adapter;
use crate::vulkan::device::Device;
use ash::vk;
use std::sync::Arc;

pub struct CmdPool {
    device: Arc<Device>,
    pub handle: vk::CommandPool,
}

impl CmdPool {
    pub fn new(device: Arc<Device>, adapter: &Adapter) -> Self {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(
                adapter
                    .queue_family_indices
                    .graphics_family
                    .expect("Graphics queue family not found"),
            );

        let handle = unsafe_vk_try!(device.handle.create_command_pool(&pool_info, None));

        Self { device, handle }
    }
}

impl Drop for CmdPool {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_command_pool(self.handle, None);
        }
    }
}
