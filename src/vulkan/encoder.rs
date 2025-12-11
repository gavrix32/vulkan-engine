use crate::unsafe_vk_try;
use crate::vulkan::device::Device;
use crate::vulkan::pool::CmdPool;
use ash::vk;
use std::sync::Arc;

pub struct Encoder {
    device: Arc<Device>,
    pub cmd: vk::CommandBuffer,
}

impl Encoder {
    pub fn new(device: Arc<Device>, cmd_pool: &CmdPool) -> Self {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool.handle)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd = unsafe_vk_try!(device.handle.allocate_command_buffers(&alloc_info))[0];

        Self { device, cmd }
    }

    pub fn begin(&mut self) {
        unsafe_vk_try!(
            self.device
                .handle
                .begin_command_buffer(self.cmd, &vk::CommandBufferBeginInfo::default())
        );
    }

    pub fn end(&self) {
        unsafe_vk_try!(self.device.handle.end_command_buffer(self.cmd));
    }

    pub fn cmd_set_viewport(&self, first_viewport: u32, viewports: &[vk::Viewport]) {
        let cmd = self.cmd;

        unsafe {
            self.device
                .handle
                .cmd_set_viewport(cmd, first_viewport, viewports);
        }
    }

    pub fn cmd_set_scissor(&self, first_scissor: u32, scissors: &[vk::Rect2D]) {
        let cmd = self.cmd;

        unsafe {
            self.device
                .handle
                .cmd_set_scissor(cmd, first_scissor, scissors);
        }
    }

    pub fn cmd_bind_pipeline(
        &self,
        pipeline_bind_point: vk::PipelineBindPoint,
        pipeline: vk::Pipeline,
    ) {
        let cmd = self.cmd;

        unsafe {
            self.device
                .handle
                .cmd_bind_pipeline(cmd, pipeline_bind_point, pipeline);
        }
    }

    pub fn cmd_bind_vertex_buffers(
        &self,
        first_binding: u32,
        buffers: &[vk::Buffer],
        offsets: &[vk::DeviceSize],
    ) {
        let cmd = self.cmd;

        unsafe {
            self.device
                .handle
                .cmd_bind_vertex_buffers(cmd, first_binding, buffers, offsets);
        }
    }

    pub fn cmd_bind_index_buffer(
        &self,
        buffer: vk::Buffer,
        offset: vk::DeviceSize,
        index_type: vk::IndexType,
    ) {
        let cmd = self.cmd;

        unsafe {
            self.device
                .handle
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
        let cmd = self.cmd;

        unsafe {
            self.device.handle.cmd_bind_descriptor_sets(
                cmd,
                pipeline_bind_point,
                layout,
                first_set,
                descriptor_sets,
                dynamic_offsets,
            );
        }
    }

    pub fn cmd_push_constants(
        &self,
        layout: vk::PipelineLayout,
        stage_flags: vk::ShaderStageFlags,
        offset: u32,
        constants: &[u8],
    ) {
        let cmd = self.cmd;

        unsafe {
            self.device
                .handle
                .cmd_push_constants(cmd, layout, stage_flags, offset, constants);
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
        let cmd = self.cmd;

        unsafe {
            self.device.handle.cmd_draw_indexed(
                cmd,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
        }
    }

    pub fn cmd_draw(
        &self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        let cmd = self.cmd;

        unsafe {
            self.device.handle.cmd_draw(
                cmd,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );
        }
    }

    pub fn cmd_dispatch(&self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        let cmd = self.cmd;

        unsafe {
            self.device
                .handle
                .cmd_dispatch(cmd, group_count_x, group_count_y, group_count_z);
        }
    }

    pub fn cmd_pipeline_barrier2(&self, dependency_info: &vk::DependencyInfo) {
        let cmd = self.cmd;

        unsafe {
            self.device
                .handle
                .cmd_pipeline_barrier2(cmd, dependency_info);
        }
    }

    pub fn cmd_copy_buffer(
        &self,
        src_buffer: vk::Buffer,
        dst_buffer: vk::Buffer,
        regions: &[vk::BufferCopy],
    ) {
        let cmd = self.cmd;

        unsafe {
            self.device
                .handle
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
        let cmd = self.cmd;

        unsafe {
            self.device.handle.cmd_copy_buffer_to_image(
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
        let cmd = self.cmd;

        unsafe {
            self.device.handle.cmd_blit_image(
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

    pub fn cmd_begin_rendering(&self, rendering_info: &vk::RenderingInfo) {
        let cmd = self.cmd;

        unsafe {
            self.device.handle.cmd_begin_rendering(cmd, rendering_info);
        }
    }

    pub fn cmd_end_rendering(&self) {
        let cmd = self.cmd;

        unsafe {
            self.device.handle.cmd_end_rendering(cmd);
        }
    }

    pub fn begin_single_time(device: Arc<Device>, pool: &CmdPool) -> Self {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool.handle)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd = unsafe_vk_try!(device.handle.allocate_command_buffers(&alloc_info))[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe_vk_try!(device.handle.begin_command_buffer(cmd, &begin_info));

        Self {
            device: device.clone(),
            cmd,
        }
    }

    pub fn end_single_time(&self, queue: vk::Queue) {
        unsafe_vk_try!(self.device.handle.end_command_buffer(self.cmd));

        let cmds = [self.cmd];

        let submit_info = vk::SubmitInfo::default().command_buffers(&cmds);

        unsafe_vk_try!(
            self.device
                .handle
                .queue_submit(queue, &[submit_info], vk::Fence::null())
        );

        unsafe_vk_try!(self.device.handle.queue_wait_idle(queue));
    }
}
