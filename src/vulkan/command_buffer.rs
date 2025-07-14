use crate::vulkan::device::Device;
use ash::vk;
use std::sync::Arc;

pub struct CommandBuffer;

impl CommandBuffer {
    pub fn begin_single_time_commands(
        device: Arc<Device>,
        command_pool: vk::CommandPool,
    ) -> Vec<vk::CommandBuffer> {
        let mut command_buffers: Vec<vk::CommandBuffer> = Vec::with_capacity(1);

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(command_buffers.capacity() as u32);

        command_buffers = unsafe {
            device
                .ash_device
                .allocate_command_buffers(&command_buffer_allocate_info)
        }
        .expect("Failed to allocate command buffers");

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device
                .ash_device
                .begin_command_buffer(command_buffers[0], &command_buffer_begin_info)
        }
        .expect("Failed to begin recording command buffer");

        command_buffers
    }

    pub fn end_single_time_commands(
        device: Arc<Device>,
        command_pool: vk::CommandPool,
        command_buffers: Vec<vk::CommandBuffer>,
        graphics_queue: vk::Queue,
    ) {
        unsafe { device.ash_device.end_command_buffer(command_buffers[0]) }
            .expect("Failed to end recording command buffer");

        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);

        unsafe {
            device
                .ash_device
                .queue_submit(graphics_queue, &[submit_info], vk::Fence::null())
                .unwrap();
            device.ash_device.queue_wait_idle(graphics_queue).unwrap();
            device
                .ash_device
                .free_command_buffers(command_pool, &command_buffers);
        }
    }
}
