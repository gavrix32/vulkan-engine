use crate::asset::AssetManager;
use crate::camera::Camera;
use crate::context::RenderContext;
use crate::mesh::Mesh;
use crate::scene::Scene;
use crate::unsafe_vk_try;
use crate::vertex::Vertex;
use crate::vulkan::adapter::Adapter;
use crate::vulkan::buffer::Buffer;
use crate::vulkan::descriptor::{DescriptorPool, DescriptorSetLayout, DescriptorWriter};
use crate::vulkan::device::Device;
use crate::vulkan::encoder::Encoder;
use crate::vulkan::image::{Image, ImageBuilder, ImageView, Sampler};
use crate::vulkan::instance::Instance;
use crate::vulkan::pipeline::{Pipeline, PipelineBuilder};
use crate::vulkan::swapchain::Swapchain;
use crate::vulkan::sync;
use ash::util::Align;
use ash::vk;
use glam::{Mat4, Vec3, Vec4};
use log::warn;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::sync::Arc;
use std::time::Instant;

// TODO: FrameContext?
const MAX_FRAMES_IN_FLIGHT: usize = 2;

#[repr(C)]
#[derive(Clone, Copy)]
struct UniformBufferData {
    // std140 base alignment 16 bytes
    model: Mat4,     // 0
    view: Mat4,      // 64
    proj: Mat4,      // 128
    light_pos: Vec4, // 192
    cam_pos: Vec4,   // 208
                     // 224
}

pub struct Renderer {
    skybox_pipeline: Pipeline,
    light_pipeline: Pipeline,
    pbr_pipeline: Pipeline,

    _descriptor_pools: Vec<DescriptorPool>,

    _pbr_descriptor_layout: DescriptorSetLayout,
    _light_descriptor_layout: DescriptorSetLayout,
    _skybox_descriptor_layout: DescriptorSetLayout,

    pbr_descriptor_sets: Vec<vk::DescriptorSet>,
    light_descriptor_sets: Vec<vk::DescriptorSet>,
    skybox_descriptor_sets: Vec<vk::DescriptorSet>,

    uniform_buffers: Vec<Vec<Buffer>>,

    pub scene: Scene,

    _cubemap_image_view: ImageView,
    _cubemap_image: Arc<Image>,

    _sampler: Sampler,

    _env_image_view: ImageView,
    _env_image: Arc<Image>,

    frame_in_flight: usize,

    pub framebuffer_resized: bool,
    pub width: u32,
    pub height: u32,

    timer: Instant,

    in_flight_fences: Vec<sync::Fence>,
    image_available_semaphores: Vec<sync::Semaphore>,
    render_finished_semaphores: Vec<sync::Semaphore>,

    encoder: Encoder,
    swapchain: Swapchain,

    ctx: Arc<RenderContext>,
}

impl Renderer {
    pub fn new(
        width: u32,
        height: u32,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        validation: bool,
    ) -> Self {
        let ctx = Arc::new(RenderContext::new(
            display_handle,
            window_handle,
            validation,
        ));

        let msaa_samples = ctx.adapter.max_usable_sample_count(&ctx.instance);

        let swapchain = Swapchain::new(
            &ctx.instance,
            &ctx.adapter,
            ctx.device.clone(),
            &ctx.surface,
            width,
            height,
            msaa_samples,
        );

        let swapchain_image_count = swapchain.images.len();

        let encoder = Encoder::new(ctx.device.clone(), &ctx.adapter, MAX_FRAMES_IN_FLIGHT);

        let mut image_available_semaphores: Vec<sync::Semaphore> =
            Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut render_finished_semaphores: Vec<sync::Semaphore> =
            Vec::with_capacity(swapchain_image_count);
        let mut in_flight_fences: Vec<sync::Fence> = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            image_available_semaphores.push(sync::Semaphore::new(ctx.device.clone()));
            in_flight_fences.push(sync::Fence::new(ctx.device.clone(), true));
        }
        for _ in 0..swapchain_image_count {
            render_finished_semaphores.push(sync::Semaphore::new(ctx.device.clone()));
        }

        let asset = AssetManager::new(ctx.clone());

        let mut meshes = Vec::new();
        meshes.push(asset.load_gltf(include_bytes!("../resources/models/sponza.glb")));
        meshes.push(asset.load_gltf(include_bytes!("../resources/models/Box.glb")));

        let scene = Scene {
            meshes,
            camera: Camera::default(),
        };

        let uniform_buffers =
            Self::create_uniform_buffers(&ctx.instance, &ctx.adapter, ctx.device.clone());

        let sampler = Sampler::new(
            ctx.device.clone(),
            vk::Filter::LINEAR,
            vk::Filter::LINEAR,
            vk::LOD_CLAMP_NONE,
        );

        let (env_image, env_image_view) = asset.load_texture_rgba32f(include_bytes!(
            "../resources/textures/the_sky_is_on_fire_4k.hdr"
        ));

        let pbr_descriptor_layout = DescriptorSetLayout::builder(ctx.device.clone())
            .binding(
                0,
                vk::DescriptorType::UNIFORM_BUFFER,
                1,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                vk::DescriptorBindingFlags::empty(),
            )
            .binding(
                1,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                1,
                vk::ShaderStageFlags::FRAGMENT,
                vk::DescriptorBindingFlags::empty(),
            )
            .binding(
                2,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                scene.meshes[0].images.len() as u32,
                vk::ShaderStageFlags::FRAGMENT,
                vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                    | vk::DescriptorBindingFlags::PARTIALLY_BOUND
                    | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT,
            )
            .build();

        let light_descriptor_layout = DescriptorSetLayout::builder(ctx.device.clone())
            .binding(
                0,
                vk::DescriptorType::UNIFORM_BUFFER,
                1,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                vk::DescriptorBindingFlags::empty(),
            )
            .build();

        let skybox_descriptor_layout = DescriptorSetLayout::builder(ctx.device.clone())
            .binding(
                0,
                vk::DescriptorType::UNIFORM_BUFFER,
                1,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                vk::DescriptorBindingFlags::empty(),
            )
            .binding(
                1,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                1,
                vk::ShaderStageFlags::FRAGMENT,
                vk::DescriptorBindingFlags::empty(),
            )
            .build();

        let color_format = Swapchain::get_format(&ctx.adapter, &ctx.surface);

        let binding_descriptions = [Vertex::get_binding_description()];
        let attribute_descriptions = Vertex::get_attribute_descriptions();

        // Equirectangular to cubemap

        let cubemap_size = 1024;
        let faces_count = 6;

        let cubemap_image = Arc::new(
            ImageBuilder::default()
                .size(cubemap_size, cubemap_size)
                .layers(faces_count)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
                .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
                .build(&ctx.instance, &ctx.adapter, ctx.device.clone()),
        );

        let cubemap_image_array_view = ImageView::new(
            ctx.device.clone(),
            cubemap_image.clone(),
            vk::ImageViewType::TYPE_2D_ARRAY,
            vk::ImageAspectFlags::COLOR,
        );

        let cubemap_image_view = ImageView::new(
            ctx.device.clone(),
            cubemap_image.clone(),
            vk::ImageViewType::CUBE,
            vk::ImageAspectFlags::COLOR,
        );

        let equirect_to_cubemap_descriptor_layout =
            DescriptorSetLayout::builder(ctx.device.clone())
                .binding(
                    0,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    1,
                    vk::ShaderStageFlags::COMPUTE,
                    vk::DescriptorBindingFlags::empty(),
                )
                .binding(
                    1,
                    vk::DescriptorType::STORAGE_IMAGE,
                    1,
                    vk::ShaderStageFlags::COMPUTE,
                    vk::DescriptorBindingFlags::empty(),
                )
                .build();

        let equirect_to_cubemap_pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1),
        ];
        let equirect_to_cubemap_pool =
            DescriptorPool::new(ctx.device.clone(), 1, &equirect_to_cubemap_pool_sizes);
        let equirect_to_cubemap_descriptor_set =
            equirect_to_cubemap_pool.allocate(&equirect_to_cubemap_descriptor_layout, 0);

        DescriptorWriter::default()
            .image(
                0,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                &env_image_view,
                &sampler,
            )
            .image(
                1,
                vk::DescriptorType::STORAGE_IMAGE,
                vk::ImageLayout::GENERAL,
                &cubemap_image_array_view,
                &sampler,
            )
            .update(ctx.device.clone(), equirect_to_cubemap_descriptor_set);

        let cubemap_image_encoder = Encoder::begin_single_time(ctx.device.clone(), &ctx.adapter);
        Image::transition_layout(
            &cubemap_image_encoder,
            cubemap_image.handle,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            1,
            faces_count,
        );

        let equirect_to_cubemap_pipeline = PipelineBuilder::default()
            .shader_stage(
                Vec::from(include_bytes!("shaders/spirv/equirect_to_cubemap.spv")),
                vk::ShaderStageFlags::COMPUTE,
                c"main",
            )
            .descriptor_set_layouts(&[equirect_to_cubemap_descriptor_layout.vk_layout])
            .build_compute_pipeline(ctx.device.clone());

        cubemap_image_encoder.cmd_bind_pipeline(
            vk::PipelineBindPoint::COMPUTE,
            equirect_to_cubemap_pipeline.vk_pipeline,
        );

        let group_count = (cubemap_size + 16 - 1) / 16;

        cubemap_image_encoder.cmd_bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            equirect_to_cubemap_pipeline.layout,
            0,
            &[equirect_to_cubemap_descriptor_set],
            &[],
        );

        cubemap_image_encoder.cmd_dispatch(group_count, group_count, faces_count);

        Image::transition_layout(
            &cubemap_image_encoder,
            cubemap_image.handle,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            1,
            faces_count,
        );
        cubemap_image_encoder.end_single_time(ctx.device.graphics_queue);

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(3),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(2),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(scene.meshes[0].images.len() as u32),
        ];

        let mut descriptor_pools = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            descriptor_pools.push(DescriptorPool::new(ctx.device.clone(), 3, &pool_sizes));
        }

        let mut pbr_descriptor_sets = Vec::new();
        for frame in 0..MAX_FRAMES_IN_FLIGHT {
            let pool = &descriptor_pools[frame];
            let set = pool.allocate(&pbr_descriptor_layout, scene.meshes[0].images.len() as u32);

            DescriptorWriter::default()
                .buffer(
                    0,
                    vk::DescriptorType::UNIFORM_BUFFER,
                    &uniform_buffers[0][frame],
                )
                .image(
                    1,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    &cubemap_image_view,
                    &sampler,
                )
                .images(
                    2,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    &scene.meshes[0].image_views,
                    &sampler,
                )
                .update(ctx.device.clone(), set);

            pbr_descriptor_sets.push(set);
        }

        let mut light_descriptor_sets = Vec::new();
        for frame in 0..MAX_FRAMES_IN_FLIGHT {
            let pool = &descriptor_pools[frame];
            let set = pool.allocate(&light_descriptor_layout, 0);

            DescriptorWriter::default()
                .buffer(
                    0,
                    vk::DescriptorType::UNIFORM_BUFFER,
                    &uniform_buffers[1][frame],
                )
                .update(ctx.device.clone(), set);

            light_descriptor_sets.push(set);
        }

        let mut skybox_descriptor_sets = Vec::new();
        for frame in 0..MAX_FRAMES_IN_FLIGHT {
            let pool = &descriptor_pools[frame];
            let set = pool.allocate(&skybox_descriptor_layout, 0);

            DescriptorWriter::default()
                .buffer(
                    0,
                    vk::DescriptorType::UNIFORM_BUFFER,
                    &uniform_buffers[0][frame],
                )
                .image(
                    1,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    &cubemap_image_view,
                    &sampler,
                )
                .update(ctx.device.clone(), set);

            skybox_descriptor_sets.push(set);
        }

        let pipeline_builder = PipelineBuilder::default()
            .vertex_input(&binding_descriptions, &attribute_descriptions)
            .input_assembly(vk::PrimitiveTopology::TRIANGLE_LIST, false)
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR])
            .viewport_state(1, 1)
            .rasterization_state(
                false,
                false,
                vk::PolygonMode::FILL,
                1.0,
                vk::CullModeFlags::BACK,
                vk::FrontFace::COUNTER_CLOCKWISE,
                false,
            )
            .multisample_state(swapchain.msaa_samples)
            .color_blend_state(vk::ColorComponentFlags::RGBA)
            .formats(color_format, vk::Format::D32_SFLOAT_S8_UINT);

        let pbr_pipeline = pipeline_builder
            .clone()
            .shader_stage(
                Vec::from(include_bytes!("shaders/spirv/pbr.spv")),
                vk::ShaderStageFlags::VERTEX,
                c"vertex_main",
            )
            .shader_stage(
                Vec::from(include_bytes!("shaders/spirv/pbr.spv")),
                vk::ShaderStageFlags::FRAGMENT,
                c"fragment_main",
            )
            .depth_stencil_state(true, true, vk::CompareOp::LESS)
            .descriptor_set_layouts(&[pbr_descriptor_layout.vk_layout])
            .build_graphics_pipeline(ctx.device.clone());

        let light_pipeline = pipeline_builder
            .clone()
            .shader_stage(
                Vec::from(include_bytes!("shaders/spirv/light.spv")),
                vk::ShaderStageFlags::VERTEX,
                c"vertex_main",
            )
            .shader_stage(
                Vec::from(include_bytes!("shaders/spirv/light.spv")),
                vk::ShaderStageFlags::FRAGMENT,
                c"fragment_main",
            )
            .depth_stencil_state(true, true, vk::CompareOp::LESS)
            .descriptor_set_layouts(&[light_descriptor_layout.vk_layout])
            .build_graphics_pipeline(ctx.device.clone());

        let skybox_pipeline = PipelineBuilder::default()
            .vertex_input(&binding_descriptions, &attribute_descriptions)
            .input_assembly(vk::PrimitiveTopology::TRIANGLE_LIST, false)
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR])
            .viewport_state(1, 1)
            .rasterization_state(
                false,
                false,
                vk::PolygonMode::FILL,
                1.0,
                vk::CullModeFlags::NONE,
                vk::FrontFace::COUNTER_CLOCKWISE,
                false,
            )
            .multisample_state(swapchain.msaa_samples)
            .color_blend_state(vk::ColorComponentFlags::RGBA)
            .formats(color_format, vk::Format::D32_SFLOAT_S8_UINT)
            .shader_stage(
                Vec::from(include_bytes!("shaders/spirv/skybox.spv")),
                vk::ShaderStageFlags::VERTEX,
                c"vertex_main",
            )
            .shader_stage(
                Vec::from(include_bytes!("shaders/spirv/skybox.spv")),
                vk::ShaderStageFlags::FRAGMENT,
                c"fragment_main",
            )
            .depth_stencil_state(true, true, vk::CompareOp::LESS_OR_EQUAL)
            .descriptor_set_layouts(&[skybox_descriptor_layout.vk_layout])
            .build_graphics_pipeline(ctx.device.clone());

        Self {
            _descriptor_pools: descriptor_pools,

            _pbr_descriptor_layout: pbr_descriptor_layout,
            _light_descriptor_layout: light_descriptor_layout,
            _skybox_descriptor_layout: skybox_descriptor_layout,

            pbr_descriptor_sets,
            light_descriptor_sets,
            skybox_descriptor_sets,

            pbr_pipeline,
            light_pipeline,
            skybox_pipeline,

            _sampler: sampler,

            _env_image_view: env_image_view,
            _env_image: env_image,

            scene,

            _cubemap_image_view: cubemap_image_view,
            _cubemap_image: cubemap_image,

            uniform_buffers,

            frame_in_flight: 0,

            framebuffer_resized: false,
            width,
            height,

            timer: Instant::now(),

            in_flight_fences,
            image_available_semaphores,
            render_finished_semaphores,

            encoder,
            swapchain,

            ctx,
        }
    }

    fn create_uniform_buffers(
        instance: &Instance,
        adapter: &Adapter,
        device: Arc<Device>,
    ) -> Vec<Vec<Buffer>> {
        let buffer_size = size_of::<UniformBufferData>() as vk::DeviceSize;

        let mut uniform_buffers: Vec<Vec<Buffer>> = Vec::new();

        for _ in 0..2 {
            let mut uniform_buffer: Vec<Buffer> = Vec::new();
            for _ in 0..MAX_FRAMES_IN_FLIGHT {
                let mut buffer = Buffer::new(
                    instance,
                    adapter,
                    device.clone(),
                    buffer_size,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                );
                buffer.map_memory();
                uniform_buffer.push(buffer);
            }
            uniform_buffers.push(uniform_buffer);
        }

        uniform_buffers
    }

    fn record_command_buffer(&mut self, swapchain_image_index: u32) {
        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(self.swapchain.extent.width as f32)
            .height(self.swapchain.extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);
        let viewports = [viewport];

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.swapchain.extent,
        };
        let scissors = [scissor];

        let light_pos_transform = Mat4::from_translation(Vec3::new(
            self.timer.elapsed().as_secs_f32().sin() * 5.0,
            3.0,
            -0.3,
        ));

        self.encoder.begin(self.frame_in_flight);

        Image::transition_layout(
            &self.encoder,
            self.swapchain.images[swapchain_image_index as usize],
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            1,
            1,
        );

        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.swapchain.color_image_view.handle)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .resolve_mode(vk::ResolveModeFlags::AVERAGE)
            .resolve_image_view(self.swapchain.image_views[swapchain_image_index as usize])
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
            .image_view(self.swapchain.depth_image_view.handle)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(clear_depth);

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: self.swapchain.extent.width,
                    height: self.swapchain.extent.height,
                },
            })
            .layer_count(1)
            .color_attachments(&color_attachments)
            .depth_attachment(&depth_attachment);

        self.encoder.cmd_begin_rendering(&rendering_info);

        self.encoder.cmd_set_viewport(0, &viewports);
        self.encoder.cmd_set_scissor(0, &scissors);

        // PBR
        self.encoder.cmd_bind_pipeline(
            vk::PipelineBindPoint::GRAPHICS,
            self.pbr_pipeline.vk_pipeline,
        );

        for (primitive_index, primitive) in self.scene.meshes[0].primitives.iter().enumerate() {
            self.encoder.cmd_bind_vertex_buffers(
                0,
                &[self.scene.meshes[0].vertex_buffer.vk_buffer],
                &[0],
            );
            self.encoder.cmd_bind_index_buffer(
                self.scene.meshes[0].index_buffer.vk_buffer,
                vk::DeviceSize::default(),
                vk::IndexType::UINT32,
            );

            self.update_uniform_buffer(
                &self.scene.meshes[0],
                primitive_index,
                light_pos_transform,
                false,
            );

            self.encoder.cmd_bind_descriptor_sets(
                vk::PipelineBindPoint::GRAPHICS,
                self.pbr_pipeline.layout,
                0,
                &[self.pbr_descriptor_sets[self.frame_in_flight]],
                &[],
            );

            self.encoder
                .cmd_draw_indexed(primitive.index_count, 1, primitive.first_index, 0, 0);
        }

        // Light
        self.encoder.cmd_bind_pipeline(
            vk::PipelineBindPoint::GRAPHICS,
            self.light_pipeline.vk_pipeline,
        );

        for (primitive_index, primitive) in self.scene.meshes[1].primitives.iter().enumerate() {
            self.encoder.cmd_bind_vertex_buffers(
                0,
                &[self.scene.meshes[1].vertex_buffer.vk_buffer],
                &[0],
            );
            self.encoder.cmd_bind_index_buffer(
                self.scene.meshes[1].index_buffer.vk_buffer,
                vk::DeviceSize::default(),
                vk::IndexType::UINT32,
            );

            self.update_uniform_buffer(
                &self.scene.meshes[1],
                primitive_index,
                light_pos_transform,
                true,
            );

            self.encoder.cmd_bind_descriptor_sets(
                vk::PipelineBindPoint::GRAPHICS,
                self.light_pipeline.layout,
                0,
                &[self.light_descriptor_sets[self.frame_in_flight]],
                &[],
            );

            self.encoder
                .cmd_draw_indexed(primitive.index_count, 1, primitive.first_index, 0, 0);
        }

        // Skybox
        self.encoder.cmd_bind_pipeline(
            vk::PipelineBindPoint::GRAPHICS,
            self.skybox_pipeline.vk_pipeline,
        );

        self.encoder.cmd_bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            self.skybox_pipeline.layout,
            0,
            &[self.skybox_descriptor_sets[self.frame_in_flight]],
            &[],
        );

        self.encoder.cmd_draw(36, 1, 0, 0);

        self.encoder.cmd_end_rendering();

        Image::transition_layout(
            &self.encoder,
            self.swapchain.images[swapchain_image_index as usize],
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            1,
            1,
        );

        self.encoder.end();
    }

    fn update_uniform_buffer(
        &self,
        mesh: &Mesh,
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
            self.width as f32 / self.height as f32,
            0.01,
            1000.0,
        );
        proj.y_axis *= -1.0;

        let pos = self.scene.camera.pos;

        let uniform_buffer_data = UniformBufferData {
            model,
            view: self.scene.camera.view(),
            proj,
            light_pos: light_pos_transform * Vec4::new(0.0, 0.0, 0.0, 1.0),
            cam_pos: Vec4::new(pos.x, pos.y, pos.z, 0.0),
        };

        let uniform_buffer = if is_light {
            &self.uniform_buffers[1][self.frame_in_flight]
        } else {
            &self.uniform_buffers[0][self.frame_in_flight]
        };

        let mut uniform_align = unsafe {
            Align::new(
                uniform_buffer
                    .p_data
                    .expect("No data pointer in uniform buffer"),
                align_of::<f32>() as vk::DeviceSize,
                size_of::<UniformBufferData>() as vk::DeviceSize,
            )
        };
        uniform_align.copy_from_slice(&[uniform_buffer_data]);
    }

    pub fn draw_frame(&mut self) {
        unsafe_vk_try!(self.ctx.device.ash_device.wait_for_fences(
            &[self.in_flight_fences[self.frame_in_flight].vk_fence],
            true,
            u64::MAX,
        ));

        let image_index: u32;

        match self
            .swapchain
            .acquire_next_image(&self.image_available_semaphores[self.frame_in_flight])
        {
            None => {
                warn!("Recreating swapchain...");
                self.recreate_swapchain();
                return;
            }
            Some(index) => image_index = index,
        }

        unsafe_vk_try!(
            self.ctx
                .device
                .ash_device
                .reset_fences(&[self.in_flight_fences[self.frame_in_flight].vk_fence])
        );

        unsafe_vk_try!(self.ctx.device.ash_device.reset_command_buffer(
            self.encoder.command_buffers[self.frame_in_flight],
            vk::CommandBufferResetFlags::empty(),
        ));

        self.record_command_buffer(image_index);

        let signal_semaphore = &self.render_finished_semaphores[image_index as usize];

        self.ctx.device.submit_graphics(
            self.encoder.command_buffers[self.frame_in_flight],
            &self.image_available_semaphores[self.frame_in_flight],
            signal_semaphore,
            &self.in_flight_fences[self.frame_in_flight],
        );

        if !self.swapchain.present(image_index, signal_semaphore) {
            self.recreate_swapchain();
        }

        if self.framebuffer_resized {
            self.framebuffer_resized = false;
            self.recreate_swapchain();
        }

        self.frame_in_flight = (self.frame_in_flight + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    fn recreate_swapchain(&mut self) {
        self.swapchain.recreate(
            &self.ctx.instance,
            &self.ctx.adapter,
            &self.ctx.surface,
            self.width,
            self.height,
            self.swapchain.msaa_samples,
        );
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        self.ctx.device.wait_idle();
    }
}
