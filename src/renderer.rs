use crate::asset::AssetManager;
use crate::context::RenderContext;
use crate::frame::FrameManager;
use crate::scene::Scene;
use crate::vertex::Vertex;
use crate::vulkan::descriptor::{DescriptorPool, DescriptorSetLayout, DescriptorWriter};
use crate::vulkan::encoder::Encoder;
use crate::vulkan::image::{Image, ImageBuilder, ImageView, Sampler};
use crate::vulkan::pipeline::{Pipeline, PipelineBuilder};
use crate::vulkan::pool::CmdPool;
use crate::vulkan::swapchain::Swapchain;
use crate::vulkan::sync;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;
use log::warn;
use std::sync::Arc;
use std::time::Instant;

const MAX_FRAMES_IN_FLIGHT: usize = 2;
const MAX_MIP_LEVELS: u32 = 5;

// TODO: Uniform descriptor set
// TODO: loading scene in main.rs and separate descriptor sets (IBL and Meshes)
// TODO: Normal map UNORM -> SNORM + remove * 2.0 - 1.0 in shader
// TODO: Albedo SRGB?

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PushConstantData {
    roughness: f32,
}

pub struct Renderer {
    skybox_pipeline: Pipeline,
    // light_pipeline: Pipeline,
    pbr_pipeline: Pipeline,

    _descriptor_pool: DescriptorPool,

    ibl_descriptor_set: vk::DescriptorSet,
    // light_descriptor_sets: Vec<vk::DescriptorSet>,
    sky_descriptor_set: vk::DescriptorSet,

    _sampler: Sampler,

    _cubemap_image_view: ImageView,
    _cubemap_image: Arc<Image>,

    _env_image_view: ImageView,
    _env_image: Arc<Image>,

    _irr_image_view: ImageView,
    _irr_image: Arc<Image>,

    _prefilter_image_view: ImageView,
    _prefilter_image: Arc<Image>,

    _lut_image_view: ImageView,
    _lut_image: Arc<Image>,

    pub framebuffer_resized: bool,
    pub width: u32,
    pub height: u32,
    timer: Instant,

    swapchain: Swapchain,
    render_finished_semaphores: Vec<sync::Semaphore>,
    frame_mgr: FrameManager,

    ctx: Arc<RenderContext>,
}

impl Renderer {
    pub fn new(width: u32, height: u32, ctx: Arc<RenderContext>) -> Self {
        let msaa_samples = ctx.adapter.max_usable_sample_count(&ctx.instance);

        let swapchain = Swapchain::new(
            &ctx.instance,
            &ctx.adapter,
            ctx.device.clone(),
            &ctx.surface,
            ctx.allocator.clone(),
            width,
            height,
            msaa_samples,
        );

        let frame_mgr = FrameManager::new(ctx.clone(), MAX_FRAMES_IN_FLIGHT);

        let mut render_finished_semaphores = Vec::new();

        for _ in 0..swapchain.images.len() {
            render_finished_semaphores.push(sync::Semaphore::new(ctx.device.clone()));
        }

        let asset = AssetManager::new(ctx.clone(), ctx.allocator.clone());

        let sampler = Sampler::new(
            ctx.device.clone(),
            vk::Filter::LINEAR,
            vk::Filter::LINEAR,
            vk::LOD_CLAMP_NONE,
        );

        let (env_image, env_image_view) = asset.load_texture_rgba32f(include_bytes!(
            "../resources/textures/the_sky_is_on_fire_4k.hdr"
        ));

        // let light_descriptor_layout = DescriptorSetLayout::builder(ctx.device.clone())
        //     .binding(
        //         0,
        //         vk::DescriptorType::UNIFORM_BUFFER,
        //         1,
        //         vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        //         vk::DescriptorBindingFlags::empty(),
        //     )
        //     .build();

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
                .build(
                    &ctx.instance,
                    &ctx.adapter,
                    ctx.device.clone(),
                    ctx.allocator.clone(),
                    MemoryLocation::Unknown,
                ),
        );

        let cubemap_image_array_view = ImageView::new(
            ctx.device.clone(),
            cubemap_image.clone(),
            vk::ImageViewType::TYPE_2D_ARRAY,
            vk::ImageAspectFlags::COLOR,
            0,
            1,
        );

        let cubemap_image_view = ImageView::new(
            ctx.device.clone(),
            cubemap_image.clone(),
            vk::ImageViewType::CUBE,
            vk::ImageAspectFlags::COLOR,
            0,
            1,
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

        let single_time_pool = CmdPool::new(ctx.device.clone(), &ctx.adapter);
        let cubemap_image_encoder =
            Encoder::begin_single_time(ctx.device.clone(), &single_time_pool);

        Image::transition_layout(
            &cubemap_image_encoder,
            cubemap_image.handle,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            0,
            1,
            faces_count,
        );

        let equirect_to_cubemap_pipeline = PipelineBuilder::default()
            .shader_stage(
                Vec::from(include_bytes!("shaders/spirv/equirect_to_cubemap.spv")),
                vk::ShaderStageFlags::COMPUTE,
                c"main",
            )
            .descriptor_set_layouts(&[equirect_to_cubemap_descriptor_layout.handle])
            .build_compute_pipeline(ctx.device.clone());

        cubemap_image_encoder.cmd_bind_pipeline(
            vk::PipelineBindPoint::COMPUTE,
            equirect_to_cubemap_pipeline.handle,
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
            0,
            1,
            faces_count,
        );
        cubemap_image_encoder.end_single_time(ctx.device.graphics_queue);

        // Irradiance

        let irr_size = 32;

        let irr_image = Arc::new(
            ImageBuilder::default()
                .size(irr_size, irr_size)
                .layers(faces_count)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
                .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
                .build(
                    &ctx.instance,
                    &ctx.adapter,
                    ctx.device.clone(),
                    ctx.allocator.clone(),
                    MemoryLocation::Unknown,
                ),
        );

        let irr_image_array_view = ImageView::new(
            ctx.device.clone(),
            irr_image.clone(),
            vk::ImageViewType::TYPE_2D_ARRAY,
            vk::ImageAspectFlags::COLOR,
            0,
            1,
        );

        let irr_image_view = ImageView::new(
            ctx.device.clone(),
            irr_image.clone(),
            vk::ImageViewType::CUBE,
            vk::ImageAspectFlags::COLOR,
            0,
            1,
        );

        let irr_descriptor_layout = DescriptorSetLayout::builder(ctx.device.clone())
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

        let irr_pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1),
        ];
        let irr_pool = DescriptorPool::new(ctx.device.clone(), 1, &irr_pool_sizes);
        let irr_descriptor_set = irr_pool.allocate(&irr_descriptor_layout, 0);

        DescriptorWriter::default()
            .image(
                0,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                &cubemap_image_view,
                &sampler,
            )
            .image(
                1,
                vk::DescriptorType::STORAGE_IMAGE,
                vk::ImageLayout::GENERAL,
                &irr_image_array_view,
                &sampler,
            )
            .update(ctx.device.clone(), irr_descriptor_set);

        let irr_image_encoder = Encoder::begin_single_time(ctx.device.clone(), &single_time_pool);

        Image::transition_layout(
            &irr_image_encoder,
            irr_image.handle,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            0,
            1,
            faces_count,
        );

        let irr_pipeline = PipelineBuilder::default()
            .shader_stage(
                Vec::from(include_bytes!("shaders/spirv/irradiance_convolution.spv")),
                vk::ShaderStageFlags::COMPUTE,
                c"main",
            )
            .descriptor_set_layouts(&[irr_descriptor_layout.handle])
            .build_compute_pipeline(ctx.device.clone());

        irr_image_encoder.cmd_bind_pipeline(vk::PipelineBindPoint::COMPUTE, irr_pipeline.handle);

        let group_count = (irr_size + 16 - 1) / 16;

        irr_image_encoder.cmd_bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            irr_pipeline.layout,
            0,
            &[irr_descriptor_set],
            &[],
        );

        irr_image_encoder.cmd_dispatch(group_count, group_count, faces_count);

        Image::transition_layout(
            &irr_image_encoder,
            irr_image.handle,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            0,
            1,
            faces_count,
        );
        irr_image_encoder.end_single_time(ctx.device.graphics_queue);

        // Prefilter

        let prefilter_size = 128;

        let prefilter_image = Arc::new(
            ImageBuilder::default()
                .size(prefilter_size, prefilter_size)
                .layers(faces_count)
                .mip_levels(MAX_MIP_LEVELS)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
                .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
                .build(
                    &ctx.instance,
                    &ctx.adapter,
                    ctx.device.clone(),
                    ctx.allocator.clone(),
                    MemoryLocation::Unknown,
                ),
        );

        let prefilter_image_view = ImageView::new(
            ctx.device.clone(),
            prefilter_image.clone(),
            vk::ImageViewType::CUBE,
            vk::ImageAspectFlags::COLOR,
            0,
            MAX_MIP_LEVELS,
        );

        let prefilter_descriptor_layout = DescriptorSetLayout::builder(ctx.device.clone())
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

        let prefilter_pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(MAX_MIP_LEVELS),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(MAX_MIP_LEVELS),
        ];
        let prefilter_pool =
            DescriptorPool::new(ctx.device.clone(), MAX_MIP_LEVELS, &prefilter_pool_sizes);
        let mut prefilter_descriptor_sets = Vec::with_capacity(MAX_MIP_LEVELS as usize);

        let prefilter_encoder = Encoder::begin_single_time(ctx.device.clone(), &single_time_pool);

        Image::transition_layout(
            &prefilter_encoder,
            prefilter_image.handle,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            0,
            MAX_MIP_LEVELS,
            faces_count,
        );

        let push_constant_range = vk::PushConstantRange::default()
            .offset(0)
            .size(size_of::<PushConstantData>() as u32)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);

        let prefilter_pipeline = PipelineBuilder::default()
            .shader_stage(
                Vec::from(include_bytes!("shaders/spirv/prefilter.spv")),
                vk::ShaderStageFlags::COMPUTE,
                c"main",
            )
            .descriptor_set_layouts(&[prefilter_descriptor_layout.handle])
            .push_constant_ranges(&[push_constant_range])
            .build_compute_pipeline(ctx.device.clone());

        let mut prefilter_push_constants = PushConstantData { roughness: 0.0 };

        let mut prefilter_image_array_views = Vec::new();

        for level in 0..MAX_MIP_LEVELS {
            prefilter_image_array_views.push(ImageView::new(
                ctx.device.clone(),
                prefilter_image.clone(),
                vk::ImageViewType::TYPE_2D_ARRAY,
                vk::ImageAspectFlags::COLOR,
                level,
                1,
            ));

            prefilter_descriptor_sets
                .push(prefilter_pool.allocate(&prefilter_descriptor_layout, 0));

            DescriptorWriter::default()
                .image(
                    0,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    &cubemap_image_view,
                    &sampler,
                )
                .image(
                    1,
                    vk::DescriptorType::STORAGE_IMAGE,
                    vk::ImageLayout::GENERAL,
                    &prefilter_image_array_views[level as usize],
                    &sampler,
                )
                .update(
                    ctx.device.clone(),
                    prefilter_descriptor_sets[level as usize],
                );

            prefilter_encoder
                .cmd_bind_pipeline(vk::PipelineBindPoint::COMPUTE, prefilter_pipeline.handle);

            prefilter_encoder.cmd_bind_descriptor_sets(
                vk::PipelineBindPoint::COMPUTE,
                prefilter_pipeline.layout,
                0,
                &[prefilter_descriptor_sets[level as usize]],
                &[],
            );

            prefilter_push_constants.roughness = level as f32 / (MAX_MIP_LEVELS - 1) as f32;

            prefilter_encoder.cmd_push_constants(
                prefilter_pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&[prefilter_push_constants]),
            );

            let group_count = ((prefilter_size >> level) + 16 - 1) / 16;

            prefilter_encoder.cmd_dispatch(group_count, group_count, faces_count);
        }

        Image::transition_layout(
            &prefilter_encoder,
            prefilter_image.handle,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            0,
            MAX_MIP_LEVELS,
            faces_count,
        );
        prefilter_encoder.end_single_time(ctx.device.graphics_queue);

        // BRDF LUT

        let lut_size = 512;

        let lut_image = Arc::new(
            ImageBuilder::default()
                .size(lut_size, lut_size)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
                .build(
                    &ctx.instance,
                    &ctx.adapter,
                    ctx.device.clone(),
                    ctx.allocator.clone(),
                    MemoryLocation::Unknown,
                ),
        );

        let lut_image_view = ImageView::new(
            ctx.device.clone(),
            lut_image.clone(),
            vk::ImageViewType::TYPE_2D,
            vk::ImageAspectFlags::COLOR,
            0,
            1,
        );

        let lut_descriptor_layout = DescriptorSetLayout::builder(ctx.device.clone())
            .binding(
                0,
                vk::DescriptorType::STORAGE_IMAGE,
                1,
                vk::ShaderStageFlags::COMPUTE,
                vk::DescriptorBindingFlags::empty(),
            )
            .build();

        let lut_pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)];
        let lut_pool = DescriptorPool::new(ctx.device.clone(), 1, &lut_pool_sizes);
        let lut_descriptor_set = lut_pool.allocate(&lut_descriptor_layout, 0);

        DescriptorWriter::default()
            .image(
                0,
                vk::DescriptorType::STORAGE_IMAGE,
                vk::ImageLayout::GENERAL,
                &lut_image_view,
                &sampler,
            )
            .update(ctx.device.clone(), lut_descriptor_set);

        let lut_image_encoder = Encoder::begin_single_time(ctx.device.clone(), &single_time_pool);
        Image::transition_layout(
            &lut_image_encoder,
            lut_image.handle,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
            0,
            1,
            1,
        );

        let lut_pipeline = PipelineBuilder::default()
            .shader_stage(
                Vec::from(include_bytes!("shaders/spirv/brdf_lut.spv")),
                vk::ShaderStageFlags::COMPUTE,
                c"main",
            )
            .descriptor_set_layouts(&[lut_descriptor_layout.handle])
            .build_compute_pipeline(ctx.device.clone());

        lut_image_encoder.cmd_bind_pipeline(vk::PipelineBindPoint::COMPUTE, lut_pipeline.handle);

        let group_count = (lut_size + 16 - 1) / 16;

        lut_image_encoder.cmd_bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            lut_pipeline.layout,
            0,
            &[lut_descriptor_set],
            &[],
        );

        lut_image_encoder.cmd_dispatch(group_count, group_count, 1);

        Image::transition_layout(
            &lut_image_encoder,
            lut_image.handle,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            0,
            1,
            1,
        );
        lut_image_encoder.end_single_time(ctx.device.graphics_queue);

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(128)];

        let descriptor_pool = DescriptorPool::new(ctx.device.clone(), 3, &pool_sizes);

        let ibl_descriptor_set = descriptor_pool.allocate(&ctx.ibl_descriptor_layout, 0);

        DescriptorWriter::default()
            .image(
                0,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                &cubemap_image_view,
                &sampler,
            )
            .image(
                1,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                &irr_image_view,
                &sampler,
            )
            .image(
                2,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                &prefilter_image_view,
                &sampler,
            )
            .image(
                3,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                &lut_image_view,
                &sampler,
            )
            .update(ctx.device.clone(), ibl_descriptor_set);

        let sky_descriptor_set = descriptor_pool.allocate(&ctx.sky_descriptor_layout, 0);

        DescriptorWriter::default()
            .image(
                0,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                &cubemap_image_view,
                &sampler,
            )
            .update(ctx.device.clone(), sky_descriptor_set);

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
            .descriptor_set_layouts(&[
                frame_mgr.frames[0].uniform_descriptor_layout.handle,
                ctx.ibl_descriptor_layout.handle,
                ctx.res_descriptor_layout.handle,
            ])
            .build_graphics_pipeline(ctx.device.clone());

        // let light_pipeline = pipeline_builder
        //     .clone()
        //     .shader_stage(
        //         Vec::from(include_bytes!("shaders/spirv/light.spv")),
        //         vk::ShaderStageFlags::VERTEX,
        //         c"vertex_main",
        //     )
        //     .shader_stage(
        //         Vec::from(include_bytes!("shaders/spirv/light.spv")),
        //         vk::ShaderStageFlags::FRAGMENT,
        //         c"fragment_main",
        //     )
        //     .depth_stencil_state(true, true, vk::CompareOp::LESS)
        //     .descriptor_set_layouts(&[light_descriptor_layout.vk_layout])
        //     .build_graphics_pipeline(ctx.device.clone());

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
            .descriptor_set_layouts(&[
                frame_mgr.frames[0].uniform_descriptor_layout.handle,
                ctx.sky_descriptor_layout.handle,
            ])
            .build_graphics_pipeline(ctx.device.clone());

        Self {
            _descriptor_pool: descriptor_pool,

            ibl_descriptor_set,
            // light_descriptor_set,
            sky_descriptor_set,

            pbr_pipeline,
            // light_pipeline,
            skybox_pipeline,

            _sampler: sampler,

            _env_image_view: env_image_view,
            _env_image: env_image,

            _irr_image_view: irr_image_view,
            _irr_image: irr_image,

            _prefilter_image_view: prefilter_image_view,
            _prefilter_image: prefilter_image,

            _lut_image_view: lut_image_view,
            _lut_image: lut_image,

            _cubemap_image_view: cubemap_image_view,
            _cubemap_image: cubemap_image,

            framebuffer_resized: false,
            width,
            height,

            timer: Instant::now(),

            swapchain,

            render_finished_semaphores,

            frame_mgr,

            ctx,
        }
    }

    pub fn draw_frame(&mut self, scene: &Scene) {
        let frame = self.frame_mgr.current();

        self.ctx.device.wait_for_fence(&frame.in_flight_fence);

        let image_index: u32;

        match self
            .swapchain
            .acquire_next_image(&frame.image_available_semaphore)
        {
            None => {
                warn!("Recreating swapchain...");
                self.recreate_swapchain();
                return;
            }
            Some(index) => image_index = index,
        }

        self.ctx.device.reset_fence(&frame.in_flight_fence);

        self.ctx.device.reset_command_buffer(frame.encoder.cmd);

        frame.record_commands(
            &self.swapchain,
            &self.timer,
            &self.pbr_pipeline,
            self.ibl_descriptor_set,
            &self.skybox_pipeline,
            self.sky_descriptor_set,
            image_index,
            scene,
        );

        let signal_semaphore = &self.render_finished_semaphores[image_index as usize];

        self.ctx.device.submit_graphics(
            frame.encoder.cmd,
            &frame.image_available_semaphore,
            signal_semaphore,
            &frame.in_flight_fence,
        );

        if !self.swapchain.present(image_index, signal_semaphore) {
            self.recreate_swapchain();
        }

        if self.framebuffer_resized {
            self.framebuffer_resized = false;
            self.recreate_swapchain();
        }

        self.frame_mgr.update();
    }

    fn recreate_swapchain(&mut self) {
        self.swapchain.recreate(
            &self.ctx.instance,
            &self.ctx.adapter,
            self.ctx.allocator.clone(),
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
