use crate::unsafe_vk_try;
use crate::vulkan::adapter::{Adapter, DEVICE_EXTENSIONS};
use crate::vulkan::instance::Instance;
use crate::vulkan::sync::{Fence, Semaphore};
use ash::vk;
use std::collections::HashSet;
use std::ffi::c_char;

pub struct Device {
    pub handle: ash::Device,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
}

impl Device {
    pub fn new(instance: &Instance, adapter: &Adapter) -> Self {
        let queue_priority = 1.0f32;
        let queue_priorities = [queue_priority];

        let mut unique_queue_families = HashSet::new();
        unique_queue_families.insert(
            adapter
                .queue_family_indices
                .graphics_family
                .expect("Graphics queue family not found"),
        );
        unique_queue_families.insert(
            adapter
                .queue_family_indices
                .present_family
                .expect("Present queue family not found"),
        );

        let mut queue_create_infos = Vec::new();

        for queue_family in unique_queue_families {
            let queue_create_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family)
                .queue_priorities(&queue_priorities);
            queue_create_infos.push(queue_create_info);
        }

        let adapter_extension_name_pointers: Vec<*const c_char> =
            DEVICE_EXTENSIONS.iter().map(|s| s.as_ptr()).collect();

        let mut vulkan12_features = vk::PhysicalDeviceVulkan12Features::default()
            .descriptor_indexing(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_variable_descriptor_count(true)
            .runtime_descriptor_array(true);

        let mut vulkan13_features = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);

        vulkan13_features.p_next = &mut vulkan12_features as *mut _ as *mut std::ffi::c_void;

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&*adapter_extension_name_pointers)
            .push_next(&mut vulkan13_features);

        let handle = unsafe_vk_try!(instance.handle.create_device(
            adapter.handle,
            &device_create_info,
            None
        ));

        let graphics_queue = unsafe {
            handle.get_device_queue(
                adapter
                    .queue_family_indices
                    .graphics_family
                    .expect("Graphics queue family not found"),
                0,
            )
        };
        let present_queue = unsafe {
            handle.get_device_queue(
                adapter
                    .queue_family_indices
                    .present_family
                    .expect("Present queue family not found"),
                0,
            )
        };

        Self {
            handle,
            graphics_queue,
            present_queue,
        }
    }

    pub fn wait_idle(&self) {
        unsafe_vk_try!(self.handle.device_wait_idle());
    }

    pub fn wait_for_fence(&self, fence: &Fence) {
        unsafe_vk_try!(
            self.handle
                .wait_for_fences(&[fence.handle], true, u64::MAX,)
        );
    }

    pub fn reset_fence(&self, fence: &Fence) {
        unsafe_vk_try!(self.handle.reset_fences(&[fence.handle]));
    }

    pub fn reset_command_buffer(&self, cmd: vk::CommandBuffer) {
        unsafe_vk_try!(
            self.handle
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
        );
    }

    pub fn submit_graphics(
        &self,
        cmd_buffer: vk::CommandBuffer,
        wait_semaphore: &Semaphore,
        signal_semaphore: &Semaphore,
        fence: &Fence,
    ) {
        let wait_semaphores = [wait_semaphore.handle];
        let signal_semaphores = [signal_semaphore.handle];
        let cmd_buffers = [cmd_buffer];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
            .command_buffers(&cmd_buffers)
            .signal_semaphores(&signal_semaphores);
        let submit_infos = [submit_info];

        unsafe_vk_try!(
            self.handle
                .queue_submit(self.graphics_queue, &submit_infos, fence.handle,)
        );
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.handle.destroy_device(None);
        }
    }
}
