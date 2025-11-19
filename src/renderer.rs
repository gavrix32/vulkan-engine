use crate::camera::Camera;
use crate::vertex::Vertex;
use crate::vulkan::adapter::Adapter;
use crate::vulkan::buffer::Buffer;
use crate::vulkan::descriptor::{DescriptorPool, DescriptorSetLayout, DescriptorWriter};
use crate::vulkan::device::Device;
use crate::vulkan::encoder::Encoder;
use crate::vulkan::image::Image;
use crate::vulkan::instance::Instance;
use crate::vulkan::pipeline::{Pipeline, PipelineBuilder};
use crate::vulkan::surface::Surface;
use crate::vulkan::swapchain::Swapchain;
use crate::vulkan::{sync, util};
use crate::{loader, unsafe_vk_try};
use ash::util::Align;
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use log::{info, warn};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::io::Cursor;
use std::sync::Arc;
use std::time::Instant;

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
    in_flight_fences: Vec<sync::Fence>,

    image_available_semaphores: Vec<sync::Semaphore>,
    render_finished_semaphores: Vec<sync::Semaphore>,

    uniform_buffers: Vec<Vec<Buffer>>,

    light_index_buffer: Buffer,
    light_vertex_buffer: Buffer,

    index_buffer: Buffer,
    vertex_buffer: Buffer,

    encoder: Encoder,

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

    _images: Vec<Image>,

    swapchain: Swapchain,
    device: Arc<Device>,
    adapter: Adapter,
    surface: Surface,
    instance: Instance,

    frame_in_flight: usize,

    msaa_samples: vk::SampleCountFlags,

    pub framebuffer_resized: bool,
    pub width: u32,
    pub height: u32,

    pub camera: Camera,

    primitives: Vec<loader::PrimitiveInfo>,
    light_primitives: Vec<loader::PrimitiveInfo>,

    timer: Instant,

    env_index: u32,
}

impl Renderer {
    pub fn new(
        width: u32,
        height: u32,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        validation: bool,
    ) -> Self {
        let instance = Instance::new(display_handle, validation);
        let surface = Surface::new(&instance, display_handle, window_handle);
        let adapter = Adapter::new(&instance, &surface);
        let device = Arc::new(Device::new(&instance, &adapter));
        let msaa_samples = adapter.max_usable_sample_count(&instance);

        let swapchain = Swapchain::new(
            &instance,
            &adapter,
            device.clone(),
            &surface,
            width,
            height,
            msaa_samples,
        );

        let encoder = Encoder::new(device.clone(), &adapter, MAX_FRAMES_IN_FLIGHT);

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

        let mut images = Vec::new();
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

            let image = Image::from_bytes(
                image_bytes,
                image_width,
                image_height,
                vk::Format::R8G8B8A8_SRGB,
                &instance,
                &adapter,
                device.clone(),
                true,
                vk::SampleCountFlags::TYPE_1,
                vk::Filter::LINEAR,
                vk::Filter::LINEAR,
                1,
            );

            images.push(image);
        }

        let env_index = images.len() as u32;
        let env_image = Image::read_rgba32(
            &mut Cursor::new(include_bytes!(
                "../resources/textures/the_sky_is_on_fire_4k.hdr"
            )),
            &instance,
            &adapter,
            device.clone(),
            true,
            vk::SampleCountFlags::TYPE_1,
            vk::Filter::LINEAR,
            vk::Filter::LINEAR,
        );
        images.push(env_image);

        let placeholder_image = Image::read_rgba8(
            &mut Cursor::new(include_bytes!("../resources/textures/placeholder.png")),
            &instance,
            &adapter,
            device.clone(),
            true,
            vk::SampleCountFlags::TYPE_1,
            vk::Filter::NEAREST,
            vk::Filter::NEAREST,
        );
        images.push(placeholder_image);

        info!("Textures: {}, Size: {} MB", images.len(), size_mb);

        let vertex_buffer = Self::create_buffer(
            &instance,
            &adapter,
            device.clone(),
            device.graphics_queue,
            &vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        );

        let index_buffer = Self::create_buffer(
            &instance,
            &adapter,
            device.clone(),
            device.graphics_queue,
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
            &instance,
            &adapter,
            device.clone(),
            device.graphics_queue,
            &light_vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        );

        let light_index_buffer = Self::create_buffer(
            &instance,
            &adapter,
            device.clone(),
            device.graphics_queue,
            &light_indices,
            vk::BufferUsageFlags::INDEX_BUFFER,
        );

        let uniform_buffers = Self::create_uniform_buffers(&instance, &adapter, device.clone());

        let pbr_descriptor_layout = DescriptorSetLayout::builder(device.clone())
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

        let light_descriptor_layout = DescriptorSetLayout::builder(device.clone())
            .binding(
                0,
                vk::DescriptorType::UNIFORM_BUFFER,
                1,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                vk::DescriptorBindingFlags::empty(),
            )
            .build();

        let blit_descriptor_layout = DescriptorSetLayout::builder(device.clone())
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
            descriptor_pools.push(DescriptorPool::new(device.clone(), 3, &pool_sizes));
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
                    &images,
                )
                .update(device.clone(), set);

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
                .update(device.clone(), set);

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
                    &images[env_index as usize],
                )
                .update(device.clone(), set);

            blit_descriptor_sets.push(set);
        }

        let color_format = Swapchain::get_format(&adapter, &surface);

        let binding_descriptions = [Vertex::get_binding_description()];
        let attribute_descriptions = Vertex::get_attribute_descriptions();

        // Equirectangular to cubemap

        let cubemap_size = 1024;
        let faces_count = 6;

        let cubemap_image = Image::new(
            &instance,
            &adapter,
            device.clone(),
            cubemap_size,
            cubemap_size,
            vk::Format::R16G16B16A16_SFLOAT,
            vk::ImageType::TYPE_2D,
            vk::ImageViewType::TYPE_2D_ARRAY,
            faces_count,
            vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
            vk::ImageCreateFlags::CUBE_COMPATIBLE,
            false,
            vk::SampleCountFlags::TYPE_1,
        )
        .create_sampler(1, vk::Filter::LINEAR, vk::Filter::LINEAR);

        let equirect_to_cubemap_descriptor_layout = DescriptorSetLayout::builder(device.clone())
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
            DescriptorPool::new(device.clone(), 1, &equirect_to_cubemap_pool_sizes);
        let equirect_to_cubemap_descriptor_set =
            equirect_to_cubemap_pool.allocate(&equirect_to_cubemap_descriptor_layout, 0);

        DescriptorWriter::default()
            .image(
                0,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                &images[env_index as usize],
            )
            .image(
                1,
                vk::DescriptorType::STORAGE_IMAGE,
                vk::ImageLayout::GENERAL,
                &cubemap_image,
            )
            .update(device.clone(), equirect_to_cubemap_descriptor_set);

        let cubemap_image_encoder = Encoder::begin_single_time(device.clone(), &adapter);
        Image::transition_layout(
            &cubemap_image_encoder,
            cubemap_image.vk_image,
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
            .build_compute_pipeline(device.clone());

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
            cubemap_image.vk_image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            1,
            1,
        );
        cubemap_image_encoder.end_single_time(device.graphics_queue);

        // TODO: HDR in RGBA16F

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
            .multisample_state(msaa_samples)
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
            .build_graphics_pipeline(device.clone());

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
            .build_graphics_pipeline(device.clone());

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
            .build_graphics_pipeline(device.clone());

        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) =
            Self::create_sync_objects(device.clone(), swapchain.images.len());

        Self {
            swapchain,
            device,
            adapter,
            surface,
            instance,

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

            encoder,

            _images: images,

            light_vertex_buffer,
            light_index_buffer,

            vertex_buffer,
            index_buffer,

            uniform_buffers,

            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            frame_in_flight: 0,

            msaa_samples,

            framebuffer_resized: false,
            width,
            height,

            camera: Camera::default(),

            primitives,
            light_primitives,

            timer: Instant::now(),

            env_index,
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
        let data_ptr = staging_buffer.p_data.unwrap();

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

    fn create_sync_objects(
        device: Arc<Device>,
        swapchain_image_count: usize,
    ) -> (Vec<sync::Semaphore>, Vec<sync::Semaphore>, Vec<sync::Fence>) {
        let mut image_available_semaphores: Vec<sync::Semaphore> =
            Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut render_finished_semaphores: Vec<sync::Semaphore> =
            Vec::with_capacity(swapchain_image_count);
        let mut in_flight_fences: Vec<sync::Fence> = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            image_available_semaphores.push(sync::Semaphore::new(device.clone()));
            in_flight_fences.push(sync::Fence::new(device.clone(), true));
        }
        for _ in 0..swapchain_image_count {
            render_finished_semaphores.push(sync::Semaphore::new(device.clone()));
        }

        (
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
        )
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
            .image_view(self.swapchain.color_image.view)
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
            .image_view(self.swapchain.depth_image.view)
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

        // Blit
        self.encoder.cmd_bind_pipeline(
            vk::PipelineBindPoint::GRAPHICS,
            self.blit_pipeline.vk_pipeline,
        );

        self.encoder.cmd_bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            self.blit_pipeline.layout,
            0,
            &[self.blit_descriptor_sets[self.frame_in_flight]],
            &[],
        );

        self.update_push_constants();

        self.encoder.cmd_push_constants(
            self.blit_pipeline.layout,
            vk::ShaderStageFlags::ALL,
            0,
            bytemuck::bytes_of(&[self.push_constant_data]),
        );

        self.encoder.cmd_draw(3, 1, 0, 0);

        // PBR
        self.encoder.cmd_bind_pipeline(
            vk::PipelineBindPoint::GRAPHICS,
            self.pbr_pipeline.vk_pipeline,
        );

        for (primitive_index, primitive) in self.primitives.iter().enumerate() {
            self.encoder
                .cmd_bind_vertex_buffers(0, &[self.vertex_buffer.vk_buffer], &[0]);
            self.encoder.cmd_bind_index_buffer(
                self.index_buffer.vk_buffer,
                vk::DeviceSize::default(),
                vk::IndexType::UINT32,
            );

            self.update_uniform_buffer(primitive_index, light_pos_transform, false);

            self.encoder.cmd_bind_descriptor_sets(
                vk::PipelineBindPoint::GRAPHICS,
                self.pbr_pipeline.layout,
                0,
                &[self.pbr_descriptor_sets[self.frame_in_flight]],
                &[],
            );

            self.encoder.cmd_push_constants(
                self.blit_pipeline.layout,
                vk::ShaderStageFlags::ALL,
                0,
                bytemuck::bytes_of(&[self.push_constant_data]),
            );

            self.encoder
                .cmd_draw_indexed(primitive.index_count, 1, primitive.first_index, 0, 0);
        }

        // Light
        self.encoder.cmd_bind_pipeline(
            vk::PipelineBindPoint::GRAPHICS,
            self.light_pipeline.vk_pipeline,
        );

        for (primitive_index, primitive) in self.light_primitives.iter().enumerate() {
            self.encoder
                .cmd_bind_vertex_buffers(0, &[self.light_vertex_buffer.vk_buffer], &[0]);
            self.encoder.cmd_bind_index_buffer(
                self.light_index_buffer.vk_buffer,
                vk::DeviceSize::default(),
                vk::IndexType::UINT32,
            );

            self.update_uniform_buffer(primitive_index, light_pos_transform, true);

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
                uniform_buffer.p_data.unwrap(),
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
        unsafe_vk_try!(self.device.ash_device.wait_for_fences(
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
            self.device
                .ash_device
                .reset_fences(&[self.in_flight_fences[self.frame_in_flight].vk_fence])
        );

        unsafe_vk_try!(self.device.ash_device.reset_command_buffer(
            self.encoder.command_buffers[self.frame_in_flight],
            vk::CommandBufferResetFlags::empty(),
        ));

        self.record_command_buffer(image_index);

        let signal_semaphore = &self.render_finished_semaphores[image_index as usize];

        self.device.submit_graphics(
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
            &self.instance,
            &self.adapter,
            &self.surface,
            self.width,
            self.height,
            self.msaa_samples,
        );
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        self.device.wait_idle();
    }
}
