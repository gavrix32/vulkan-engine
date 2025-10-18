use crate::camera::Camera;
use crate::vulkan::adapter::Adapter;
use crate::vulkan::buffer::Buffer;
use crate::vulkan::descriptor::Descriptor;
use crate::vulkan::device::Device;
use crate::vulkan::encoder::Encoder;
use crate::vulkan::image::Image;
use crate::vulkan::instance::Instance;
use crate::vulkan::pipeline::Pipeline;
use crate::vulkan::surface::Surface;
use crate::vulkan::swapchain::Swapchain;
use crate::vulkan::sync;
use crate::vulkan::vertex::Vertex;
use crate::{loader, unsafe_vk_try};
use ash::util::Align;
use ash::vk;
use glam::{Mat4, Vec3, Vec4};
use log::{info, warn};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::io::Cursor;
use std::sync::Arc;
use std::time::Instant;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

fn rgb_to_rgba(rgb_data: &[u8]) -> Vec<u8> {
    rgb_data
        .chunks_exact(3)
        .flat_map(|rgb_pixel| [rgb_pixel[0], rgb_pixel[1], rgb_pixel[2], 255])
        .collect()
}

fn r_to_rgba(rgb_data: &[u8]) -> Vec<u8> {
    rgb_data
        .chunks_exact(1)
        .flat_map(|rgb_pixel| [rgb_pixel[0], 0, 0, 255])
        .collect()
}

fn rg_to_rgba(rgb_data: &[u8]) -> Vec<u8> {
    rgb_data
        .chunks_exact(2)
        .flat_map(|rgb_pixel| [rgb_pixel[0], rgb_pixel[1], 0, 255])
        .collect()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct UniformBufferData {
    model: Mat4,
    view: Mat4,
    proj: Mat4,
    light_pos: Vec4,
    cam_pos: Vec4,
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

    _images: Vec<Image>,

    encoder: Encoder,

    light_pipeline: Pipeline,

    pbr_pipeline: Pipeline,
    descriptor_sets: Vec<Vec<vk::DescriptorSet>>,
    light_descriptor_sets: Vec<Vec<vk::DescriptorSet>>,
    _descriptor: Descriptor,
    _light_descriptor: Descriptor,

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
        let msaa_samples = Self::get_max_usable_sample_count(&instance, &adapter);

        let swapchain = Swapchain::new(
            &instance,
            &adapter,
            device.clone(),
            &surface,
            // render_pass.vk_render_pass,
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
                    gltf::image::Format::R8G8B8 => {
                        (&rgb_to_rgba(&image_data.pixels), image_width, image_height)
                    }
                    gltf::image::Format::R8G8 => {
                        (&rg_to_rgba(&image_data.pixels), image_width, image_height)
                    }
                    gltf::image::Format::R8 => {
                        (&r_to_rgba(&image_data.pixels), image_width, image_height)
                    }
                    _ => panic!("Unsupported texture format: {:?}", image_data.format),
                }
            };

            let image = Image::from_bytes(
                image_bytes,
                image_width,
                image_height,
                &instance,
                &adapter,
                device.clone(),
                true,
                vk::SampleCountFlags::TYPE_1,
                vk::Filter::LINEAR,
                vk::Filter::LINEAR,
            );

            images.push(image);
        }

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

        let uniform_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);

        let albedo_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let normal_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let metallic_roughness_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(3)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let bindings = [
            uniform_layout_binding,
            albedo_layout_binding,
            normal_layout_binding,
            metallic_roughness_layout_binding,
        ];

        let descriptor_count = primitives.len() * MAX_FRAMES_IN_FLIGHT;
        let light_descriptor_count = light_primitives.len() * MAX_FRAMES_IN_FLIGHT;

        let uniform_buffer_descriptor_pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(descriptor_count as u32);

        let sampler_descriptor_pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count((descriptor_count * 3) as u32);

        let descriptor_pool_sizes = [
            uniform_buffer_descriptor_pool_size,
            sampler_descriptor_pool_size,
        ];

        let descriptor = Descriptor::new(
            device.clone(),
            &bindings,
            &descriptor_pool_sizes,
            descriptor_count,
        );
        let light_descriptor = Descriptor::new(
            device.clone(),
            &bindings,
            &descriptor_pool_sizes,
            light_descriptor_count,
        );

        let color_format = Swapchain::get_format(&adapter, &surface);

        let pbr_pipeline = Pipeline::new(
            device.clone(),
            Vec::from(include_bytes!("shaders/spirv/vertex.spv")),
            Vec::from(include_bytes!("shaders/spirv/fragment.spv")),
            color_format,
            vk::Format::D32_SFLOAT_S8_UINT,
            &[descriptor.layout],
            msaa_samples,
        );

        let light_pipeline = Pipeline::new(
            device.clone(),
            Vec::from(include_bytes!("shaders/spirv/light_vertex.spv")),
            Vec::from(include_bytes!("shaders/spirv/light_fragment.spv")),
            color_format,
            vk::Format::D32_SFLOAT_S8_UINT,
            &[descriptor.layout],
            msaa_samples,
        );

        let descriptor_sets =
            Self::create_descriptor_sets(&descriptor, &uniform_buffers[0], &images, &primitives);
        let light_descriptor_sets = Self::create_descriptor_sets(
            &light_descriptor,
            &uniform_buffers[1],
            &images,
            &light_primitives,
        );

        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) =
            Self::create_sync_objects(device.clone(), swapchain.images.len());

        Self {
            swapchain,
            device,
            adapter,
            surface,
            instance,

            _descriptor: descriptor,
            _light_descriptor: light_descriptor,
            descriptor_sets,
            light_descriptor_sets,
            pbr_pipeline,
            light_pipeline,

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
        }
    }

    fn get_max_usable_sample_count(instance: &Instance, adapter: &Adapter) -> vk::SampleCountFlags {
        let properties = unsafe {
            instance
                .ash_instance
                .get_physical_device_properties(adapter.physical_device)
        };
        let counts = properties.limits.framebuffer_color_sample_counts
            & properties.limits.framebuffer_depth_sample_counts;

        if counts.contains(vk::SampleCountFlags::TYPE_64) {
            return vk::SampleCountFlags::TYPE_64;
        }
        if counts.contains(vk::SampleCountFlags::TYPE_32) {
            return vk::SampleCountFlags::TYPE_32;
        }
        if counts.contains(vk::SampleCountFlags::TYPE_16) {
            return vk::SampleCountFlags::TYPE_16;
        }
        if counts.contains(vk::SampleCountFlags::TYPE_8) {
            return vk::SampleCountFlags::TYPE_8;
        }
        if counts.contains(vk::SampleCountFlags::TYPE_4) {
            return vk::SampleCountFlags::TYPE_4;
        }
        if counts.contains(vk::SampleCountFlags::TYPE_2) {
            return vk::SampleCountFlags::TYPE_2;
        }

        vk::SampleCountFlags::TYPE_1
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

    // descriptor.rs
    fn create_descriptor_sets(
        descriptor: &Descriptor,
        uniform_buffers: &Vec<Buffer>,
        images: &[Image],
        primitives: &[loader::PrimitiveInfo],
    ) -> Vec<Vec<vk::DescriptorSet>> {
        let flat_descriptor_sets = &descriptor.sets;

        let mut descriptor_sets =
            vec![vec![vk::DescriptorSet::null(); MAX_FRAMES_IN_FLIGHT]; primitives.len()];

        for (primitive_index, primitive) in primitives.iter().enumerate() {
            for frame in 0..MAX_FRAMES_IN_FLIGHT {
                let flat_index = primitive_index * MAX_FRAMES_IN_FLIGHT + frame;

                let descriptor_set = flat_descriptor_sets[flat_index];
                descriptor_sets[primitive_index][frame] = descriptor_set;

                let ubo_info = vk::DescriptorBufferInfo::default()
                    .buffer(uniform_buffers[frame].vk_buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE);
                let ubo_infos = [ubo_info];
                let write_ubo = vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .buffer_info(&ubo_infos);

                let albedo = &images[primitive.albedo_index.unwrap_or(images.len() - 1)];
                let albedo_info = vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(albedo.view)
                    .sampler(albedo.sampler.unwrap());
                let albedo_infos = [albedo_info];
                let write_albedo = vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .image_info(&albedo_infos);

                let normal = &images[primitive.normal_index.unwrap_or(images.len() - 1)];
                let normal_info = vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(normal.view)
                    .sampler(normal.sampler.unwrap());
                let normal_infos = [normal_info];
                let write_normal = vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(2)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .image_info(&normal_infos);

                let metallic_roughness = &images[primitive
                    .metallic_roughness_index
                    .unwrap_or(images.len() - 1)];
                let metallic_roughness_info = vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(metallic_roughness.view)
                    .sampler(metallic_roughness.sampler.unwrap());
                let metallic_roughness_infos = [metallic_roughness_info];
                let write_metallic_roughness = vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(3)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .image_info(&metallic_roughness_infos);

                let descriptor_writes = [
                    write_ubo,
                    write_albedo,
                    write_normal,
                    write_metallic_roughness,
                ];

                descriptor.update(&descriptor_writes);
            }
        }
        descriptor_sets
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

    fn record_command_buffer(&mut self, image_index: u32) {
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
            self.swapchain.images[image_index as usize],
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
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
            .resolve_image_view(self.swapchain.image_views[image_index as usize])
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

            self.update_uniform_buffer(primitive.model_matrix, light_pos_transform, false);

            self.encoder.cmd_bind_descriptor_sets(
                vk::PipelineBindPoint::GRAPHICS,
                self.pbr_pipeline.layout,
                0,
                &[self.descriptor_sets[primitive_index][self.frame_in_flight]],
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

        for primitive in &self.light_primitives {
            self.encoder
                .cmd_bind_vertex_buffers(0, &[self.light_vertex_buffer.vk_buffer], &[0]);
            self.encoder.cmd_bind_index_buffer(
                self.light_index_buffer.vk_buffer,
                vk::DeviceSize::default(),
                vk::IndexType::UINT32,
            );

            self.update_uniform_buffer(primitive.model_matrix, light_pos_transform, true);

            self.encoder.cmd_bind_descriptor_sets(
                vk::PipelineBindPoint::GRAPHICS,
                self.light_pipeline.layout,
                0,
                &[self.light_descriptor_sets[0][self.frame_in_flight]],
                &[],
            );

            self.encoder
                .cmd_draw_indexed(primitive.index_count, 1, primitive.first_index, 0, 0);
        }

        self.encoder.cmd_end_rendering();

        Image::transition_layout(
            &self.encoder,
            self.swapchain.images[image_index as usize],
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            1,
        );

        self.encoder.end();
    }

    fn update_uniform_buffer(&self, model: Mat4, light_pos_transform: Mat4, is_light: bool) {
        let model = if is_light {
            light_pos_transform * model * Mat4::from_scale(Vec3::new(0.1, 0.1, 0.1))
        } else {
            model
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

    pub fn draw_frame(&mut self) {
        unsafe_vk_try!(self.device.ash_device.wait_for_fences(
            &[self.in_flight_fences[self.frame_in_flight].vk_fence],
            true,
            u64::MAX,
        ));

        let image_index: u32;

        match self
            .swapchain
            .acquire_next_image(self.image_available_semaphores[self.frame_in_flight].vk_semaphore)
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

        let wait_semaphores = [self.image_available_semaphores[self.frame_in_flight].vk_semaphore];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [self.encoder.command_buffers[self.frame_in_flight]];
        let signal_semaphores =
            [self.render_finished_semaphores[image_index as usize].vk_semaphore];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);
        let submit_infos = [submit_info];

        unsafe_vk_try!(self.device.ash_device.queue_submit(
            self.device.graphics_queue,
            &submit_infos,
            self.in_flight_fences[self.frame_in_flight].vk_fence,
        ));

        let swapchains = [self.swapchain.swapchain_khr];
        let image_indices = [image_index];

        let present_info_khr = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let present_result = unsafe {
            self.swapchain
                .swapchain_device
                .queue_present(self.device.present_queue, &present_info_khr)
        };

        match present_result {
            Ok(is_suboptimal) => {
                if is_suboptimal {
                    warn!("Swapchain is suboptimal, recreating...");
                    self.recreate_swapchain();
                }
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                warn!("Swapchain is out of date, recreating...");
                self.recreate_swapchain();
            }
            Err(e) => {
                panic!("Failed to present swapchain image: {:?}", e);
            }
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
