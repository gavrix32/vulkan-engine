use crate::context::RenderContext;
use crate::model::Model;
use crate::scene::Scene;
use crate::vulkan::buffer::Buffer;
use crate::vulkan::descriptor::{DescriptorPool, DescriptorSetLayout, DescriptorWriter};
use crate::vulkan::encoder::Encoder;
use crate::vulkan::image::Image;
use crate::vulkan::pipeline::Pipeline;
use crate::vulkan::pool::CmdPool;
use crate::vulkan::swapchain::Swapchain;
use crate::vulkan::sync;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use gpu_allocator::MemoryLocation;
use std::sync::Arc;
use std::time::Instant;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct UniformBufferData {
    // std140 base alignment 16 bytes
    model: [f32; 16],    // 0
    view: [f32; 16],     // 64
    proj: [f32; 16],     // 128
    light_pos: [f32; 4], // 192
    cam_pos: [f32; 4],   // 208
                         // 224
}

pub struct FrameData {
    pub in_flight_fence: sync::Fence,
    pub image_available_semaphore: sync::Semaphore,

    pub _cmd_pool: CmdPool,
    pub encoder: Encoder,

    pub _descriptor_pool: DescriptorPool,

    pub uniform_buffer: Buffer,
    pub uniform_descriptor_layout: DescriptorSetLayout,
    pub uniform_descriptor_set: vk::DescriptorSet,
}

impl FrameData {
    pub fn new(ctx: Arc<RenderContext>) -> Self {
        let cmd_pool = CmdPool::new(ctx.device.clone(), &ctx.adapter);
        let encoder = Encoder::new(ctx.device.clone(), &cmd_pool);
        let uniform_buffer = Buffer::new(
            ctx.device.clone(),
            ctx.allocator.clone(),
            size_of::<UniformBufferData>() as vk::DeviceSize,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
        );

        let uniform_descriptor_layout = DescriptorSetLayout::builder(ctx.device.clone())
            .binding(
                0,
                vk::DescriptorType::UNIFORM_BUFFER,
                1,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                vk::DescriptorBindingFlags::empty(),
            )
            .build();

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(3)];
        let descriptor_pool = DescriptorPool::new(ctx.device.clone(), 3, &pool_sizes);
        let uniform_descriptor_set = descriptor_pool.allocate(&uniform_descriptor_layout, 0);

        DescriptorWriter::default()
            .buffer(0, vk::DescriptorType::UNIFORM_BUFFER, &uniform_buffer)
            .update(ctx.device.clone(), uniform_descriptor_set);

        Self {
            in_flight_fence: sync::Fence::new(ctx.device.clone(), true),
            image_available_semaphore: sync::Semaphore::new(ctx.device.clone()),

            _cmd_pool: cmd_pool,
            encoder,

            _descriptor_pool: descriptor_pool,

            uniform_buffer,
            uniform_descriptor_layout,
            uniform_descriptor_set,
        }
    }

    pub fn update_uniform_buffer(
        &mut self,
        width: u32,
        height: u32,
        scene: &Scene,
        mesh: &Model,
        primitive_index: usize,
        light_pos_transform: Mat4,
        is_light: bool,
    ) {
        let primitive = &mesh.primitives[primitive_index];
        let primitive_model = primitive.model_matrix;

        let model = if is_light {
            light_pos_transform * primitive_model * Mat4::from_scale(Vec3::new(0.1, 0.1, 0.1))
        } else {
            primitive_model
            // * Mat4::from_rotation_y(self.timer.elapsed().as_secs_f32() * 00.0_f32.to_radians())
        };

        let mut proj = Mat4::perspective_rh(
            70.0_f32.to_radians(),
            width as f32 / height as f32,
            0.01,
            1000.0,
        );
        proj.y_axis *= -1.0;

        let pos = scene.camera.pos;

        let uniform_buffer_data = UniformBufferData {
            model: model.to_cols_array(),
            view: scene.camera.view().to_cols_array(),
            proj: proj.to_cols_array(),
            light_pos: (light_pos_transform * Vec4::new(0.0, 0.0, 0.0, 1.0)).to_array(),
            cam_pos: [pos.x, pos.y, pos.z, 0.0],
        };

        self.uniform_buffer.update(&[uniform_buffer_data]);
    }

    pub fn record_commands(
        &mut self,
        swapchain: &Swapchain,
        timer: &Instant,
        pbr_pipeline: &Pipeline,
        ibl_descriptor_set: vk::DescriptorSet,
        skybox_pipeline: &Pipeline,
        sky_descriptor_set: vk::DescriptorSet,
        image_index: u32,
        scene: &Scene,
    ) {
        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(swapchain.extent.width as f32)
            .height(swapchain.extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);
        let viewports = [viewport];

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain.extent,
        };
        let scissors = [scissor];

        let light_pos_transform = Mat4::from_translation(Vec3::new(
            timer.elapsed().as_secs_f32().sin() * 5.0,
            3.0,
            -0.3,
        ));

        self.encoder.begin();

        Image::transition_layout(
            &self.encoder,
            swapchain.images[image_index as usize],
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            0,
            1,
            1,
        );

        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(swapchain.color_image_view.handle)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .resolve_mode(vk::ResolveModeFlags::AVERAGE)
            .resolve_image_view(swapchain.image_views[image_index as usize])
            .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(clear_color);
        let color_attachments = [color_attachment];

        let clear_depth = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };

        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(swapchain.depth_image_view.handle)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(clear_depth);

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: swapchain.extent.width,
                    height: swapchain.extent.height,
                },
            })
            .layer_count(1)
            .color_attachments(&color_attachments)
            .depth_attachment(&depth_attachment);

        self.encoder.cmd_begin_rendering(&rendering_info);

        self.encoder.cmd_set_viewport(0, &viewports);
        self.encoder.cmd_set_scissor(0, &scissors);

        // PBR
        self.encoder
            .cmd_bind_pipeline(vk::PipelineBindPoint::GRAPHICS, pbr_pipeline.handle);

        for (primitive_index, primitive) in scene.models[0].primitives.iter().enumerate() {
            self.encoder
                .cmd_bind_vertex_buffers(0, &[scene.models[0].vertex_buffer.handle], &[0]);
            self.encoder.cmd_bind_index_buffer(
                scene.models[0].index_buffer.handle,
                vk::DeviceSize::default(),
                vk::IndexType::UINT32,
            );

            self.update_uniform_buffer(
                swapchain.extent.width,
                swapchain.extent.height,
                &scene,
                &scene.models[0],
                primitive_index,
                light_pos_transform,
                false,
            );

            self.encoder.cmd_bind_descriptor_sets(
                vk::PipelineBindPoint::GRAPHICS,
                pbr_pipeline.layout,
                0,
                &[
                    self.uniform_descriptor_set,
                    ibl_descriptor_set,
                    scene.models[0].res_descriptor_set,
                ],
                &[],
            );

            self.encoder
                .cmd_draw_indexed(primitive.index_count, 1, primitive.first_index, 0, 0);
        }

        // Light
        // self.encoder.cmd_bind_pipeline(
        //     vk::PipelineBindPoint::GRAPHICS,
        //     self.light_pipeline.vk_pipeline,
        // );
        //
        // for (primitive_index, primitive) in self.scene.meshes[1].primitives.iter().enumerate() {
        //     self.encoder.cmd_bind_vertex_buffers(
        //         0,
        //         &[self.scene.meshes[1].vertex_buffer.vk_buffer],
        //         &[0],
        //     );
        //     self.encoder.cmd_bind_index_buffer(
        //         self.scene.meshes[1].index_buffer.vk_buffer,
        //         vk::DeviceSize::default(),
        //         vk::IndexType::UINT32,
        //     );
        //
        //     self.update_uniform_buffer(
        //         &self.scene.meshes[1],
        //         primitive_index,
        //         light_pos_transform,
        //         true,
        //     );
        //
        //     self.encoder.cmd_bind_descriptor_sets(
        //         vk::PipelineBindPoint::GRAPHICS,
        //         self.light_pipeline.layout,
        //         0,
        //         &[self.light_descriptor_sets[self.frame_in_flight]],
        //         &[],
        //     );
        //
        //     self.encoder
        //         .cmd_draw_indexed(primitive.index_count, 1, primitive.first_index, 0, 0);
        // }

        // Skybox
        self.encoder
            .cmd_bind_pipeline(vk::PipelineBindPoint::GRAPHICS, skybox_pipeline.handle);

        self.encoder.cmd_bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            skybox_pipeline.layout,
            0,
            &[self.uniform_descriptor_set, sky_descriptor_set],
            &[],
        );

        self.encoder.cmd_draw(36, 1, 0, 0);

        self.encoder.cmd_end_rendering();

        Image::transition_layout(
            &self.encoder,
            swapchain.images[image_index as usize],
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            0,
            1,
            1,
        );

        self.encoder.end();
    }
}

pub struct FrameManager {
    pub frames: Vec<FrameData>,
    current_index: usize,
}

impl FrameManager {
    pub fn new(ctx: Arc<RenderContext>, frame_count: usize) -> Self {
        let mut frames = Vec::new();

        for _ in 0..frame_count {
            frames.push(FrameData::new(ctx.clone()));
        }

        Self {
            frames,
            current_index: 0,
        }
    }

    pub fn current(&mut self) -> &mut FrameData {
        &mut self.frames[self.current_index]
    }

    pub fn update(&mut self) {
        self.current_index = (self.current_index + 1) % self.frames.len();
    }
}
