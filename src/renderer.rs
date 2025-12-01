use crate::camera::Camera;
use crate::context::RenderContext;
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
use crate::vulkan::util;
use crate::{loader, unsafe_vk_try};
use ash::util::Align;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use image::ImageReader;
use log::{info, warn};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::io::Cursor;
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
    env_index: u32,  // 224
    _pad0: [u32; 3], //
                     // 240
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PushConstantData {
    // std140 base alignment 16 bytes
    resolution: [f32; 2], // 0
    _pad0: [f32; 2],      //
    inv_view_proj: [[f32; 4]; 4], // 16 bytes
                          // 80 bytes
}

pub struct Renderer {
    blit_pipeline: Pipeline,
    light_pipeline: Pipeline,
    pbr_pipeline: Pipeline,

    _descriptor_pools: Vec<DescriptorPool>,

    _pbr_descriptor_layout: DescriptorSetLayout,
    _light_descriptor_layout: DescriptorSetLayout,
    _blit_descriptor_layout: DescriptorSetLayout,

    pbr_descriptor_sets: Vec<vk::DescriptorSet>,
    light_descriptor_sets: Vec<vk::DescriptorSet>,
    blit_descriptor_sets: Vec<vk::DescriptorSet>,

    push_constant_data: PushConstantData,

    uniform_buffers: Vec<Vec<Buffer>>,

    light_index_buffer: Buffer,
    light_vertex_buffer: Buffer,

    index_buffer: Buffer,
    vertex_buffer: Buffer,

    _sampler: Sampler,
    _image_views: Vec<ImageView>,

    _images: Vec<Arc<Image>>,

    frame_in_flight: usize,

    pub framebuffer_resized: bool,
    pub width: u32,
    pub height: u32,

    pub camera: Camera,

    primitives: Vec<loader::PrimitiveInfo>,
    light_primitives: Vec<loader::PrimitiveInfo>,

    timer: Instant,

    env_index: u32,

    render_context: RenderContext,
}

impl Renderer {
    pub fn new(
        width: u32,
        height: u32,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        validation: bool,
    ) -> Self {
        let render_context =
            RenderContext::new(width, height, display_handle, window_handle, validation);

        info!("Importing model");
        let (document, buffers_data, images_data) =
            gltf::import_slice(include_bytes!("../resources/models/sponza.glb"))
                .expect("Failed to load model");

        let mut vertices: Vec<Vertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        let mut primitives: Vec<loader::PrimitiveInfo> = Vec::new();

        info!("Parsing model");
        loader::parse_model(
            document,
            &buffers_data,
            &mut vertices,
            &mut indices,
            &mut primitives,
        );
        info!("Vertices: {}, Indices: {}", vertices.len(), indices.len());

        info!("Generating tangents");
        let mut mesh_view = loader::MeshView {
            vertices: &mut vertices,
            indices: &indices,
        };
        mikktspace::generate_tangents(&mut mesh_view);

        let mut images = Vec::<Arc<Image>>::new();
        let mut image_views = Vec::<ImageView>::new();

        let sampler = Sampler::new(
            render_context.device.clone(),
            vk::Filter::LINEAR,
            vk::Filter::LINEAR,
            vk::LOD_CLAMP_NONE,
        );

        let mut size_mb = 0;

        info!("Loading textures");
        for image_data in &images_data {
            size_mb += image_data.pixels.len() / 1024 / 1024;

            let (image_bytes, image_width, image_height) = {
                let image_width = image_data.width;
                let image_height = image_data.height;

                match image_data.format {
                    gltf::image::Format::R8G8B8A8 => {
                        (&image_data.pixels, image_width, image_height)
                    }
                    gltf::image::Format::R8G8B8 => (
                        &util::rgb_to_rgba(&image_data.pixels),
                        image_width,
                        image_height,
                    ),
                    gltf::image::Format::R8G8 => (
                        &util::rg_to_rgba(&image_data.pixels),
                        image_width,
                        image_height,
                    ),
                    gltf::image::Format::R8 => (
                        &util::r_to_rgba(&image_data.pixels),
                        image_width,
                        image_height,
                    ),
                    _ => panic!("Unsupported texture format: {:?}", image_data.format),
                }
            };

            let image = Arc::new(
                ImageBuilder::default()
                    .size(image_width, image_height)
                    .mipmapping(true)
                    .format(vk::Format::R8G8B8A8_SRGB)
                    .usage(
                        vk::ImageUsageFlags::TRANSFER_SRC
                            | vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::SAMPLED,
                    )
                    .bytes(image_bytes)
                    .build(
                        &render_context.instance,
                        &render_context.adapter,
                        render_context.device.clone(),
                    ),
            );

            images.push(image.clone());
            image_views.push(ImageView::new(
                render_context.device.clone(),
                image.clone(),
                vk::ImageViewType::TYPE_2D,
                vk::ImageAspectFlags::COLOR,
            ));
        }

        let env_index = images.len() as u32;

        let env_raw_bytes = include_bytes!("../resources/textures/the_sky_is_on_fire_4k.hdr");
        let env_image_reader = ImageReader::new(Cursor::new(env_raw_bytes))
            .with_guessed_format()
            .expect("Failed to guess format")
            .decode()
            .expect("Failed to decode image")
            .to_rgba32f();
        let env_bytes = bytemuck::cast_slice(env_image_reader.as_raw());
        let env_image = Arc::new(
            ImageBuilder::default()
                .size(env_image_reader.width(), env_image_reader.height())
                .mipmapping(true)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .usage(
                    vk::ImageUsageFlags::TRANSFER_SRC
                        | vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::SAMPLED,
                )
                .bytes(env_bytes)
                .build(
                    &render_context.instance,
                    &render_context.adapter,
                    render_context.device.clone(),
                ),
        );

        images.push(env_image.clone());
        image_views.push(ImageView::new(
            render_context.device.clone(),
            env_image.clone(),
            vk::ImageViewType::TYPE_2D,
            vk::ImageAspectFlags::COLOR,
        ));

        let placeholder_raw_bytes = include_bytes!("../resources/textures/placeholder.png");
        let placeholder_image_reader = ImageReader::new(Cursor::new(placeholder_raw_bytes))
            .with_guessed_format()
            .expect("Failed to guess format")
            .decode()
            .expect("Failed to decode image")
            .to_rgba8();
        let placeholder_bytes = bytemuck::cast_slice(placeholder_image_reader.as_raw());
        let placeholder_image = Arc::new(
            ImageBuilder::default()
                .size(
                    placeholder_image_reader.width(),
                    placeholder_image_reader.height(),
                )
                .mipmapping(true)
                .format(vk::Format::R8G8B8A8_SRGB)
                .usage(
                    vk::ImageUsageFlags::TRANSFER_SRC
                        | vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::SAMPLED,
                )
                .bytes(placeholder_bytes)
                .build(
                    &render_context.instance,
                    &render_context.adapter,
                    render_context.device.clone(),
                ),
        );

        images.push(placeholder_image.clone());
        image_views.push(ImageView::new(
            render_context.device.clone(),
            placeholder_image.clone(),
            vk::ImageViewType::TYPE_2D,
            vk::ImageAspectFlags::COLOR,
        ));

        info!("Textures: {}, Size: {} MB", images.len(), size_mb);

        let vertex_buffer = Self::create_buffer(
            &render_context.instance,
            &render_context.adapter,
            render_context.device.clone(),
            render_context.device.graphics_queue,
            &vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        );

        let index_buffer = Self::create_buffer(
            &render_context.instance,
            &render_context.adapter,
            render_context.device.clone(),
            render_context.device.graphics_queue,
            &indices,
            vk::BufferUsageFlags::INDEX_BUFFER,
        );

        info!("Importing light model");
        let (light_document, light_buffers_data, _) =
            gltf::import_slice(include_bytes!("../resources/models/Box.glb"))
                .expect("Failed to load model");

        let mut light_vertices: Vec<Vertex> = Vec::new();
        let mut light_indices: Vec<u32> = Vec::new();

        let mut light_primitives: Vec<loader::PrimitiveInfo> = Vec::new();

        info!("Parsing light model");
        loader::parse_model(
            light_document,
            &light_buffers_data,
            &mut light_vertices,
            &mut light_indices,
            &mut light_primitives,
        );
        info!(
            "Vertices: {}, Indices: {}",
            light_vertices.len(),
            light_indices.len()
        );

        let light_vertex_buffer = Self::create_buffer(
            &render_context.instance,
            &render_context.adapter,
            render_context.device.clone(),
            render_context.device.graphics_queue,
            &light_vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        );

        let light_index_buffer = Self::create_buffer(
            &render_context.instance,
            &render_context.adapter,
            render_context.device.clone(),
            render_context.device.graphics_queue,
            &light_indices,
            vk::BufferUsageFlags::INDEX_BUFFER,
        );

        let uniform_buffers = Self::create_uniform_buffers(
            &render_context.instance,
            &render_context.adapter,
            render_context.device.clone(),
        );

        let pbr_descriptor_layout = DescriptorSetLayout::builder(render_context.device.clone())
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
                images.len() as u32,
                vk::ShaderStageFlags::FRAGMENT,
                vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                    | vk::DescriptorBindingFlags::PARTIALLY_BOUND
                    | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT,
            )
            .build();

        let light_descriptor_layout = DescriptorSetLayout::builder(render_context.device.clone())
            .binding(
                0,
                vk::DescriptorType::UNIFORM_BUFFER,
                1,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                vk::DescriptorBindingFlags::empty(),
            )
            .build();

        let blit_descriptor_layout = DescriptorSetLayout::builder(render_context.device.clone())
            .binding(
                0,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                1,
                vk::ShaderStageFlags::FRAGMENT,
                vk::DescriptorBindingFlags::empty(),
            )
            .build();

        let push_constant_data = PushConstantData {
            resolution: [width as f32, height as f32],
            _pad0: [0.0; 2],
            inv_view_proj: [[0.0; 4]; 4],
        };

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(2),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(images.len() as u32 + 1),
        ];

        let mut descriptor_pools = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            descriptor_pools.push(DescriptorPool::new(
                render_context.device.clone(),
                3,
                &pool_sizes,
            ));
        }

        let mut pbr_descriptor_sets = Vec::new();
        for frame in 0..MAX_FRAMES_IN_FLIGHT {
            let pool = &descriptor_pools[frame];
            let set = pool.allocate(&pbr_descriptor_layout, images.len() as u32);

            DescriptorWriter::default()
                .buffer(
                    0,
                    vk::DescriptorType::UNIFORM_BUFFER,
                    &uniform_buffers[0][frame],
                )
                .images(
                    1,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    &image_views,
                    &sampler,
                )
                .update(render_context.device.clone(), set);

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
                .update(render_context.device.clone(), set);

            light_descriptor_sets.push(set);
        }

        let mut blit_descriptor_sets = Vec::new();
        for frame in 0..MAX_FRAMES_IN_FLIGHT {
            let pool = &descriptor_pools[frame];
            let set = pool.allocate(&blit_descriptor_layout, 0);

            DescriptorWriter::default()
                .image(
                    0,
                    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    &image_views[env_index as usize],
                    &sampler,
                )
                .update(render_context.device.clone(), set);

            blit_descriptor_sets.push(set);
        }

        let color_format = Swapchain::get_format(&render_context.adapter, &render_context.surface);

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
                    &render_context.instance,
                    &render_context.adapter,
                    render_context.device.clone(),
                ),
        );

        let cubemap_image_view = ImageView::new(
            render_context.device.clone(),
            cubemap_image.clone(),
            vk::ImageViewType::TYPE_2D_ARRAY,
            vk::ImageAspectFlags::COLOR,
        );

        let equirect_to_cubemap_descriptor_layout =
            DescriptorSetLayout::builder(render_context.device.clone())
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
        let equirect_to_cubemap_pool = DescriptorPool::new(
            render_context.device.clone(),
            1,
            &equirect_to_cubemap_pool_sizes,
        );
        let equirect_to_cubemap_descriptor_set =
            equirect_to_cubemap_pool.allocate(&equirect_to_cubemap_descriptor_layout, 0);

        DescriptorWriter::default()
            .image(
                0,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                &image_views[env_index as usize],
                &sampler,
            )
            .image(
                1,
                vk::DescriptorType::STORAGE_IMAGE,
                vk::ImageLayout::GENERAL,
                &cubemap_image_view,
                &sampler,
            )
            .update(
                render_context.device.clone(),
                equirect_to_cubemap_descriptor_set,
            );

        let cubemap_image_encoder =
            Encoder::begin_single_time(render_context.device.clone(), &render_context.adapter);
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
            .build_compute_pipeline(render_context.device.clone());

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
            1,
        );
        cubemap_image_encoder.end_single_time(render_context.device.graphics_queue);

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
            .multisample_state(render_context.swapchain.msaa_samples)
            .color_blend_state(vk::ColorComponentFlags::RGBA)
            .formats(color_format, vk::Format::D32_SFLOAT_S8_UINT);

        let push_constant_range = vk::PushConstantRange::default()
            .offset(0)
            .size(size_of::<PushConstantData>() as u32)
            .stage_flags(vk::ShaderStageFlags::ALL);

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
            .push_constant_ranges(&[push_constant_range])
            .build_graphics_pipeline(render_context.device.clone());

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
            .build_graphics_pipeline(render_context.device.clone());

        let blit_pipeline = pipeline_builder
            .shader_stage(
                Vec::from(include_bytes!("shaders/spirv/blit.spv")),
                vk::ShaderStageFlags::VERTEX,
                c"vertex_main",
            )
            .shader_stage(
                Vec::from(include_bytes!("shaders/spirv/blit.spv")),
                vk::ShaderStageFlags::FRAGMENT,
                c"fragment_main",
            )
            .descriptor_set_layouts(&[blit_descriptor_layout.vk_layout])
            .push_constant_ranges(&[push_constant_range])
            .build_graphics_pipeline(render_context.device.clone());

        Self {
            _descriptor_pools: descriptor_pools,

            _pbr_descriptor_layout: pbr_descriptor_layout,
            _light_descriptor_layout: light_descriptor_layout,
            _blit_descriptor_layout: blit_descriptor_layout,

            pbr_descriptor_sets,
            light_descriptor_sets,
            blit_descriptor_sets,

            push_constant_data,

            pbr_pipeline,
            light_pipeline,
            blit_pipeline,

            _images: images,
            _image_views: image_views,

            _sampler: sampler,

            light_vertex_buffer,
            light_index_buffer,

            vertex_buffer,
            index_buffer,

            uniform_buffers,

            frame_in_flight: 0,

            framebuffer_resized: false,
            width,
            height,

            camera: Camera::default(),

            primitives,
            light_primitives,

            timer: Instant::now(),

            env_index,

            render_context,
        }
    }

    fn create_buffer<T: Copy>(
        instance: &Instance,
        adapter: &Adapter,
        device: Arc<Device>,
        graphics_queue: vk::Queue,
        data: &[T],
        usage: vk::BufferUsageFlags,
    ) -> Buffer {
        let size = (size_of::<T>() * data.len()) as vk::DeviceSize;

        let mut staging_buffer = Buffer::new(
            instance,
            adapter,
            device.clone(),
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        staging_buffer.map_memory();
        let data_ptr = staging_buffer
            .p_data
            .expect("No data pointer in staging buffer");

        let mut vertex_align =
            unsafe { Align::new(data_ptr, align_of::<T>() as vk::DeviceSize, size) };
        vertex_align.copy_from_slice(&data);

        staging_buffer.unmap_memory();

        let vertex_buffer = Buffer::new(
            instance,
            adapter,
            device.clone(),
            size,
            vk::BufferUsageFlags::TRANSFER_DST | usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        staging_buffer.copy(graphics_queue, adapter, &vertex_buffer);

        vertex_buffer
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
            .width(self.render_context.swapchain.extent.width as f32)
            .height(self.render_context.swapchain.extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);
        let viewports = [viewport];

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.render_context.swapchain.extent,
        };
        let scissors = [scissor];

        let light_pos_transform = Mat4::from_translation(Vec3::new(
            self.timer.elapsed().as_secs_f32().sin() * 5.0,
            3.0,
            -0.3,
        ));

        self.render_context.encoder.begin(self.frame_in_flight);

        Image::transition_layout(
            &self.render_context.encoder,
            self.render_context.swapchain.images[swapchain_image_index as usize],
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
            .image_view(self.render_context.swapchain.color_image_view.handle)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .resolve_mode(vk::ResolveModeFlags::AVERAGE)
            .resolve_image_view(
                self.render_context.swapchain.image_views[swapchain_image_index as usize],
            )
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
            .image_view(self.render_context.swapchain.depth_image_view.handle)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(clear_depth);

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: self.render_context.swapchain.extent.width,
                    height: self.render_context.swapchain.extent.height,
                },
            })
            .layer_count(1)
            .color_attachments(&color_attachments)
            .depth_attachment(&depth_attachment);

        self.render_context
            .encoder
            .cmd_begin_rendering(&rendering_info);

        self.render_context.encoder.cmd_set_viewport(0, &viewports);
        self.render_context.encoder.cmd_set_scissor(0, &scissors);

        // Blit
        self.render_context.encoder.cmd_bind_pipeline(
            vk::PipelineBindPoint::GRAPHICS,
            self.blit_pipeline.vk_pipeline,
        );

        self.render_context.encoder.cmd_bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            self.blit_pipeline.layout,
            0,
            &[self.blit_descriptor_sets[self.frame_in_flight]],
            &[],
        );

        self.update_push_constants();

        self.render_context.encoder.cmd_push_constants(
            self.blit_pipeline.layout,
            vk::ShaderStageFlags::ALL,
            0,
            bytemuck::bytes_of(&[self.push_constant_data]),
        );

        self.render_context.encoder.cmd_draw(3, 1, 0, 0);

        // PBR
        self.render_context.encoder.cmd_bind_pipeline(
            vk::PipelineBindPoint::GRAPHICS,
            self.pbr_pipeline.vk_pipeline,
        );

        for (primitive_index, primitive) in self.primitives.iter().enumerate() {
            self.render_context.encoder.cmd_bind_vertex_buffers(
                0,
                &[self.vertex_buffer.vk_buffer],
                &[0],
            );
            self.render_context.encoder.cmd_bind_index_buffer(
                self.index_buffer.vk_buffer,
                vk::DeviceSize::default(),
                vk::IndexType::UINT32,
            );

            self.update_uniform_buffer(primitive_index, light_pos_transform, false);

            self.render_context.encoder.cmd_bind_descriptor_sets(
                vk::PipelineBindPoint::GRAPHICS,
                self.pbr_pipeline.layout,
                0,
                &[self.pbr_descriptor_sets[self.frame_in_flight]],
                &[],
            );

            self.render_context.encoder.cmd_push_constants(
                self.blit_pipeline.layout,
                vk::ShaderStageFlags::ALL,
                0,
                bytemuck::bytes_of(&[self.push_constant_data]),
            );

            self.render_context.encoder.cmd_draw_indexed(
                primitive.index_count,
                1,
                primitive.first_index,
                0,
                0,
            );
        }

        // Light
        self.render_context.encoder.cmd_bind_pipeline(
            vk::PipelineBindPoint::GRAPHICS,
            self.light_pipeline.vk_pipeline,
        );

        for (primitive_index, primitive) in self.light_primitives.iter().enumerate() {
            self.render_context.encoder.cmd_bind_vertex_buffers(
                0,
                &[self.light_vertex_buffer.vk_buffer],
                &[0],
            );
            self.render_context.encoder.cmd_bind_index_buffer(
                self.light_index_buffer.vk_buffer,
                vk::DeviceSize::default(),
                vk::IndexType::UINT32,
            );

            self.update_uniform_buffer(primitive_index, light_pos_transform, true);

            self.render_context.encoder.cmd_bind_descriptor_sets(
                vk::PipelineBindPoint::GRAPHICS,
                self.light_pipeline.layout,
                0,
                &[self.light_descriptor_sets[self.frame_in_flight]],
                &[],
            );

            self.render_context.encoder.cmd_draw_indexed(
                primitive.index_count,
                1,
                primitive.first_index,
                0,
                0,
            );
        }

        self.render_context.encoder.cmd_end_rendering();

        Image::transition_layout(
            &self.render_context.encoder,
            self.render_context.swapchain.images[swapchain_image_index as usize],
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            1,
            1,
        );

        self.render_context.encoder.end();
    }

    fn update_uniform_buffer(
        &self,
        primitive_index: usize,
        light_pos_transform: Mat4,
        is_light: bool,
    ) {
        let primitive = &self.primitives[primitive_index];
        let primitive_model = primitive.model_matrix;

        let model = if is_light {
            light_pos_transform * primitive_model * Mat4::from_scale(Vec3::new(10.0, 10.0, 10.0))
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

        let pos = self.camera.pos;

        let uniform_buffer_data = UniformBufferData {
            model,
            view: self.camera.view(),
            proj,
            light_pos: light_pos_transform * Vec4::new(0.0, 0.0, 0.0, 1.0),
            cam_pos: Vec4::new(pos.x, pos.y, pos.z, 0.0),
            env_index: self.env_index,
            _pad0: Default::default(),
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

    fn update_push_constants(&mut self) {
        let mut proj = Mat4::perspective_rh(
            70.0_f32.to_radians(),
            self.width as f32 / self.height as f32,
            0.01,
            1000.0,
        );
        proj.y_axis *= -1.0;

        let view = Mat4::from_quat(self.camera.quat.conjugate());

        let inv_view_proj = (proj * view).inverse();

        self.push_constant_data.resolution = [self.width as f32, self.height as f32];
        self.push_constant_data.inv_view_proj = inv_view_proj.to_cols_array_2d();
    }

    pub fn draw_frame(&mut self) {
        unsafe_vk_try!(self.render_context.device.ash_device.wait_for_fences(
            &[self.render_context.in_flight_fences[self.frame_in_flight].vk_fence],
            true,
            u64::MAX,
        ));

        let image_index: u32;

        match self.render_context.swapchain.acquire_next_image(
            &self.render_context.image_available_semaphores[self.frame_in_flight],
        ) {
            None => {
                warn!("Recreating swapchain...");
                self.recreate_swapchain();
                return;
            }
            Some(index) => image_index = index,
        }

        unsafe_vk_try!(
            self.render_context
                .device
                .ash_device
                .reset_fences(&[
                    self.render_context.in_flight_fences[self.frame_in_flight].vk_fence
                ])
        );

        unsafe_vk_try!(self.render_context.device.ash_device.reset_command_buffer(
            self.render_context.encoder.command_buffers[self.frame_in_flight],
            vk::CommandBufferResetFlags::empty(),
        ));

        self.record_command_buffer(image_index);

        let signal_semaphore =
            &self.render_context.render_finished_semaphores[image_index as usize];

        self.render_context.device.submit_graphics(
            self.render_context.encoder.command_buffers[self.frame_in_flight],
            &self.render_context.image_available_semaphores[self.frame_in_flight],
            signal_semaphore,
            &self.render_context.in_flight_fences[self.frame_in_flight],
        );

        if !self
            .render_context
            .swapchain
            .present(image_index, signal_semaphore)
        {
            self.recreate_swapchain();
        }

        if self.framebuffer_resized {
            self.framebuffer_resized = false;
            self.recreate_swapchain();
        }

        self.frame_in_flight = (self.frame_in_flight + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    fn recreate_swapchain(&mut self) {
        self.render_context.swapchain.recreate(
            &self.render_context.instance,
            &self.render_context.adapter,
            &self.render_context.surface,
            self.width,
            self.height,
            self.render_context.swapchain.msaa_samples,
        );
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        self.render_context.device.wait_idle();
    }
}
