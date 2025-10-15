use crate::unsafe_vk_try;
use crate::vulkan::adapter::Adapter;
use crate::vulkan::device::Device;
use ash::vk;
use std::sync::Arc;

pub struct CommandEncoder {
    device: Arc<Device>,
    pub command_pool: vk::CommandPool,
    pub command_buffers: Vec<vk::CommandBuffer>,
    command_buffer_index: usize,
}

impl CommandEncoder {
    pub fn new(device: Arc<Device>, adapter: &Adapter, max_command_buffers: usize) -> Self {
        let command_pool_create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(adapter.queue_family_indices.graphics_family.unwrap());

        let command_pool = unsafe_vk_try!(
            device
                .ash_device
                .create_command_pool(&command_pool_create_info, None)
        );

        let command_buffers: Vec<vk::CommandBuffer> = Vec::with_capacity(max_command_buffers);
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(command_buffers.capacity() as u32);

        let command_buffers = unsafe_vk_try!(
            device
                .ash_device
                .allocate_command_buffers(&command_buffer_allocate_info)
        );

        Self {
            device,
            command_pool,
            command_buffers,
            command_buffer_index: 0,
        }
    }

    pub fn begin(&mut self, frame_in_flight: usize) {
        self.command_buffer_index = frame_in_flight;

        let cmd = self.command_buffers[self.command_buffer_index];

        unsafe_vk_try!(
            self.device
                .ash_device
                .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
        );
    }

    pub fn end(&self) {
        unsafe_vk_try!(
            self.device
                .ash_device
                .end_command_buffer(self.command_buffers[self.command_buffer_index])
        );
    }

    pub fn cmd_begin_render_pass(
        &self,
        render_pass_begin: &vk::RenderPassBeginInfo,
        contents: vk::SubpassContents,
    ) {
        let cmd = self.command_buffers[self.command_buffer_index];

        unsafe {
            self.device
                .ash_device
                .cmd_begin_render_pass(cmd, render_pass_begin, contents);
        }
    }

    pub fn cmd_end_render_pass(&self) {
        let cmd = self.command_buffers[self.command_buffer_index];

        unsafe {
            self.device.ash_device.cmd_end_render_pass(cmd);
        }
    }

    pub fn cmd_set_viewport(&self, first_viewport: u32, viewports: &[vk::Viewport]) {
        let cmd = self.command_buffers[self.command_buffer_index];

        unsafe {
            self.device
                .ash_device
                .cmd_set_viewport(cmd, first_viewport, viewports);
        }
    }

    pub fn cmd_set_scissor(&self, first_scissor: u32, scissors: &[vk::Rect2D]) {
        let cmd = self.command_buffers[self.command_buffer_index];

        unsafe {
            self.device
                .ash_device
                .cmd_set_scissor(cmd, first_scissor, scissors);
        }
    }

    pub fn cmd_bind_pipeline(
        &self,
        pipeline_bind_point: vk::PipelineBindPoint,
        pipeline: vk::Pipeline,
    ) {
        let cmd = self.command_buffers[self.command_buffer_index];

        unsafe {
            self.device
                .ash_device
                .cmd_bind_pipeline(cmd, pipeline_bind_point, pipeline);
        }
    }

    pub fn cmd_bind_vertex_buffers(
        &self,
        first_binding: u32,
        buffers: &[vk::Buffer],
        offsets: &[vk::DeviceSize],
    ) {
        let cmd = self.command_buffers[self.command_buffer_index];

        unsafe {
            self.device
                .ash_device
                .cmd_bind_vertex_buffers(cmd, first_binding, buffers, offsets);
        }
    }

    pub fn cmd_bind_index_buffer(
        &self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        index_type: vk::IndexType,
    ) {
        let cmd = self.command_buffers[self.command_buffer_index];

        unsafe {
            self.device
                .ash_device
                .cmd_bind_index_buffer(cmd, buffer, offset, index_type);
        }
    }

    pub fn cmd_bind_descriptor_sets(
        &self,
        pipeline_bind_point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        first_set: u32,
        descriptor_sets: &[vk::DescriptorSet],
        dynamic_offsets: &[u32],
    ) {
        let cmd = self.command_buffers[self.command_buffer_index];

        unsafe {
            self.device.ash_device.cmd_bind_descriptor_sets(
                cmd,
                pipeline_bind_point,
                layout,
                first_set,
                descriptor_sets,
                dynamic_offsets,
            );
        }
    }

    pub fn cmd_draw_indexed(
        &self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        let cmd = self.command_buffers[self.command_buffer_index];

        unsafe {
            self.device.ash_device.cmd_draw_indexed(
                cmd,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
        }
    }

    pub fn cmd_pipeline_barrier(
        &self,
        src_stage_mask: vk::PipelineStageFlags,
        dst_stage_mask: vk::PipelineStageFlags,
        dependency_flags: vk::DependencyFlags,
        memory_barriers: &[vk::MemoryBarrier],
        buffer_memory_barriers: &[vk::BufferMemoryBarrier],
        image_memory_barriers: &[vk::ImageMemoryBarrier],
    ) {
        let cmd = self.command_buffers[self.command_buffer_index];

        unsafe {
            self.device.ash_device.cmd_pipeline_barrier(
                cmd,
                src_stage_mask,
                dst_stage_mask,
                dependency_flags,
                memory_barriers,
                buffer_memory_barriers,
                image_memory_barriers,
            );
        }
    }

    pub fn cmd_copy_buffer(
        &self,
        src_buffer: vk::Buffer,
        dst_buffer: vk::Buffer,
        regions: &[vk::BufferCopy],
    ) {
        let cmd = self.command_buffers[self.command_buffer_index];

        unsafe {
            self.device
                .ash_device
                .cmd_copy_buffer(cmd, src_buffer, dst_buffer, regions);
        }
    }

    pub fn cmd_copy_buffer_to_image(
        &self,
        src_buffer: vk::Buffer,
        dst_image: vk::Image,
        dst_image_layout: vk::ImageLayout,
        regions: &[vk::BufferImageCopy],
    ) {
        let cmd = self.command_buffers[self.command_buffer_index];

        unsafe {
            self.device.ash_device.cmd_copy_buffer_to_image(
                cmd,
                src_buffer,
                dst_image,
                dst_image_layout,
                regions,
            );
        }
    }

    pub fn cmd_blit_image(
        &self,
        src_image: vk::Image,
        src_image_layout: vk::ImageLayout,
        dst_image: vk::Image,
        dst_image_layout: vk::ImageLayout,
        regions: &[vk::ImageBlit],
        filter: vk::Filter,
    ) {
        let cmd = self.command_buffers[self.command_buffer_index];

        unsafe {
            self.device.ash_device.cmd_blit_image(
                cmd,
                src_image,
                src_image_layout,
                dst_image,
                dst_image_layout,
                regions,
                filter,
            );
        }
    }

    pub fn begin_single_time(&self) -> Self {
        let mut command_buffers: Vec<vk::CommandBuffer> = Vec::with_capacity(1);
        let command_buffer_index = 0;

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(command_buffers.capacity() as u32);

        command_buffers = unsafe_vk_try!(
            self.device
                .ash_device
                .allocate_command_buffers(&command_buffer_allocate_info)
        );

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe_vk_try!(self.device.ash_device.begin_command_buffer(
            command_buffers[command_buffer_index],
            &command_buffer_begin_info
        ));

        Self {
            device: self.device.clone(),
            command_pool: self.command_pool,
            command_buffers,
            command_buffer_index,
        }
    }

    pub fn end_single_time(&self, queue: vk::Queue) {
        unsafe_vk_try!(
            self.device
                .ash_device
                .end_command_buffer(self.command_buffers[self.command_buffer_index])
        );

        let submit_info = vk::SubmitInfo::default().command_buffers(&self.command_buffers);

        unsafe_vk_try!(self.device.ash_device.queue_submit(
            queue,
            &[submit_info],
            vk::Fence::null()
        ));

        unsafe_vk_try!(self.device.ash_device.queue_wait_idle(queue));
    }
}
