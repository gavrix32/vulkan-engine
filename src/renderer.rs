use crate::vulkan::adapter::Adapter;
use crate::vulkan::buffer::Buffer;
use crate::vulkan::device::Device;
use crate::vulkan::image::Image;
use crate::vulkan::instance::Instance;
use crate::vulkan::pipeline::Pipeline;
use crate::vulkan::render_pass::RenderPass;
use crate::vulkan::surface::Surface;
use crate::vulkan::swapchain::Swapchain;
use ash::util::Align;
use ash::vk;
use glam::{Mat4, Vec4};
use gltf::{Node, buffer};
use log::{info, warn};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::mem::offset_of;
use std::sync::Arc;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

#[derive(Copy, Clone)]
pub(crate) struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
    tex_coord: [f32; 2],
}

impl Vertex {
    pub(crate) fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    pub(crate) fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Vertex, position) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Vertex, color) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(offset_of!(Vertex, tex_coord) as u32),
        ]
    }
}

struct PrimitiveInfo {
    first_index: u32,
    index_count: u32,
    texture_index: Option<usize>,
}

fn traverse_node(
    node: Node,
    buffers: &Vec<buffer::Data>,
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u32>,
    primitive_infos: &mut Vec<PrimitiveInfo>,
    parent_transform: Mat4,
) {
    info!("Node: {}", node.name().unwrap_or("Unnamed"));

    let local_transform = Mat4::from_cols_array_2d(&node.transform().matrix());
    let global_transform = parent_transform * local_transform;

    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            let material = primitive.material();

            let texture_index = material
                .pbr_metallic_roughness()
                .base_color_texture()
                .map(|info| info.texture().source().index());

            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let tex_coords = reader
                .read_tex_coords(0)
                .map(|tc| tc.into_f32().collect())
                .unwrap_or_else(|| vec![[0.0, 0.0]]);

            let first_index = indices.len() as u32;

            let indices_iter = reader.read_indices().unwrap().into_u32();
            let index_count = indices_iter.len() as u32;

            for index in indices_iter {
                indices.push(vertices.len() as u32 + index);
            }

            for (i, pos) in reader.read_positions().unwrap().enumerate() {
                let trans_pos =
                    (global_transform * Vec4::new(pos[0], pos[1], pos[2], 1.0)).to_array();

                vertices.push(Vertex {
                    position: [trans_pos[0], trans_pos[1], trans_pos[2]],
                    color: [1.0, 1.0, 1.0],
                    tex_coord: tex_coords.get(i).copied().unwrap_or([0.0, 0.0]),
                });
            }

            let primitive_info = PrimitiveInfo {
                first_index,
                index_count,
                texture_index,
            };
            primitive_infos.push(primitive_info);
        }
    }
    for child in node.children() {
        traverse_node(
            child,
            buffers,
            vertices,
            indices,
            primitive_infos,
            global_transform,
        );
    }
}

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
#[allow(dead_code)]
#[derive(Clone, Copy)]
struct UniformBufferData {
    model: Mat4,
    view: Mat4,
    proj: Mat4,
}

pub struct Renderer {
    uniform_buffers: Vec<Buffer>,
    index_buffer: Buffer,
    vertex_buffer: Buffer,

    _images: Vec<Image>,

    command_buffers: Vec<vk::CommandBuffer>,
    command_pool: vk::CommandPool,

    pipeline: Pipeline,
    descriptor_sets: Vec<Vec<vk::DescriptorSet>>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,

    swapchain: Swapchain,
    render_pass: RenderPass,
    device: Arc<Device>,
    adapter: Adapter,
    surface: Surface,
    instance: Instance,

    ///////////////////////////
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    frame_in_flight_index: usize,

    pub framebuffer_resized: bool,
    pub width: u32,
    pub height: u32,

    pub camera_view: Mat4,

    primitive_infos: Vec<PrimitiveInfo>,
}

impl Renderer {
    pub fn new(
        width: u32,
        height: u32,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> Self {
        let instance = Instance::new(display_handle);
        let surface = Surface::new(&instance, display_handle, window_handle);
        let adapter = Adapter::new(&instance, &surface);
        let device = Arc::new(Device::new(&instance, &adapter));
        let render_pass =
            RenderPass::new(device.clone(), Swapchain::get_format(&adapter, &surface));

        let swapchain = Swapchain::new(
            &instance,
            &adapter,
            device.clone(),
            &surface,
            render_pass.vk_render_pass,
            width,
            height,
        );

        let descriptor_set_layout = Self::create_descriptor_set_layout(&device.ash_device);
        let descriptor_set_layouts = [descriptor_set_layout];

        let pipeline = Pipeline::new(device.clone(), &render_pass, &descriptor_set_layouts);

        let command_pool =
            Self::create_command_pool(&device.ash_device, &adapter.queue_family_indices);

        info!("Importing model");
        let (document, buffers_data, images_data) =
            gltf::import_slice(include_bytes!("../models/sponza.glb"))
                .expect("Failed to load model");

        let mut vertices: Vec<Vertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        let mut primitive_infos: Vec<PrimitiveInfo> = Vec::new();

        info!("Parsing model");
        for scene in document.scenes() {
            info!("Scene: {}", scene.name().unwrap_or("Unnamed"));
            for node in scene.nodes() {
                traverse_node(
                    node,
                    &buffers_data,
                    &mut vertices,
                    &mut indices,
                    &mut primitive_infos,
                    Mat4::IDENTITY,
                );
            }
        }
        info!("Vertices: {}, Indices: {}", vertices.len(), indices.len());

        let albedo_images_data: Vec<_> = primitive_infos
            .iter()
            .map(|option_i| option_i.texture_index.map(|i| &images_data[i]))
            .collect();

        let mut images = Vec::new();

        for albedo_image_data in albedo_images_data {
            if let Some(albedo_image_data) = albedo_image_data {
                let (image_bytes, image_width, image_height) = {
                    let image_data = albedo_image_data;
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
                    command_pool,
                    true,
                );

                images.push(image);
            } else {
                let image = Image::from_bytes(
                    &[255, 0, 255, 255],
                    1,
                    1,
                    &instance,
                    &adapter,
                    device.clone(),
                    command_pool,
                    true,
                );

                images.push(image);
            }
        }

        let vertex_buffer = Self::create_vertex_buffer(
            &instance,
            &adapter,
            device.clone(),
            device.graphics_queue,
            command_pool,
            &vertices,
        );

        let index_buffer = Self::create_index_buffer(
            &instance,
            &adapter,
            device.clone(),
            device.graphics_queue,
            command_pool,
            &indices,
        );

        let uniform_buffers = Self::create_uniform_buffers(&instance, &adapter, device.clone());

        let descriptor_pool = Self::create_descriptor_pool(
            &device.ash_device,
            (MAX_FRAMES_IN_FLIGHT * images.len()) as u32,
        );
        let descriptor_sets = Self::create_descriptor_sets(
            &device.ash_device,
            descriptor_set_layout,
            descriptor_pool,
            &uniform_buffers,
            &images,
        );

        let command_buffers = Self::create_command_buffers(&device.ash_device, command_pool);

        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) =
            Self::create_sync_objects(&device.ash_device, swapchain.images.len());

        Self {
            swapchain,
            render_pass,
            device,
            adapter,
            surface,
            instance,

            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            pipeline,

            command_pool,
            command_buffers,

            _images: images,

            vertex_buffer,
            index_buffer,
            uniform_buffers,

            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            frame_in_flight_index: 0,

            framebuffer_resized: false,
            width,
            height,

            camera_view: Mat4::IDENTITY,
            primitive_infos,
        }
    }

    fn create_descriptor_set_layout(device: &ash::Device) -> vk::DescriptorSetLayout {
        let uniform_buffer_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let sampler_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let bindings = [uniform_buffer_layout_binding, sampler_layout_binding];

        let descriptor_set_layout_create_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        unsafe { device.create_descriptor_set_layout(&descriptor_set_layout_create_info, None) }
            .expect("Failed to create descriptor set layout")
    }

    fn create_command_pool(
        device: &ash::Device,
        queue_family_indices: &QueueFamilyIndices,
    ) -> vk::CommandPool {
        let command_pool_create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_indices.graphics_family.unwrap());

        unsafe { device.create_command_pool(&command_pool_create_info, None) }
            .expect("Failed to create command pool")
    }

    fn create_vertex_buffer(
        instance: &Instance,
        adapter: &Adapter,
        device: Arc<Device>,
        graphics_queue: vk::Queue,
        command_pool: vk::CommandPool,
        vertices: &Vec<Vertex>,
    ) -> Buffer {
        let size = (size_of::<Vertex>() * vertices.len()) as vk::DeviceSize;

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
            unsafe { Align::new(data_ptr, align_of::<Vertex>() as vk::DeviceSize, size) };
        vertex_align.copy_from_slice(&vertices);

        staging_buffer.unmap_memory();

        let vertex_buffer = Buffer::new(
            instance,
            adapter,
            device.clone(),
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        staging_buffer.copy(graphics_queue, &vertex_buffer, command_pool);

        vertex_buffer
    }

    fn create_index_buffer(
        instance: &Instance,
        adapter: &Adapter,
        device: Arc<Device>,
        graphics_queue: vk::Queue,
        command_pool: vk::CommandPool,
        indices: &Vec<u32>,
    ) -> Buffer {
        let buffer_size = (size_of::<u32>() * indices.len()) as vk::DeviceSize;

        let mut staging_buffer = Buffer::new(
            instance,
            adapter,
            device.clone(),
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        staging_buffer.map_memory();
        let data_ptr = staging_buffer.p_data.unwrap();

        let mut vertex_align =
            unsafe { Align::new(data_ptr, align_of::<u32>() as vk::DeviceSize, buffer_size) };
        vertex_align.copy_from_slice(&indices);

        staging_buffer.unmap_memory();

        let index_buffer = Buffer::new(
            instance,
            adapter,
            device.clone(),
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        staging_buffer.copy(graphics_queue, &index_buffer, command_pool);

        index_buffer
    }

    fn create_uniform_buffers(
        instance: &Instance,
        adapter: &Adapter,
        device: Arc<Device>,
    ) -> Vec<Buffer> {
        let buffer_size = size_of::<UniformBufferData>() as vk::DeviceSize;

        let mut uniform_buffers: Vec<Buffer> = Vec::new();

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let mut uniform_buffer = Buffer::new(
                instance,
                adapter,
                device.clone(),
                buffer_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            uniform_buffer.map_memory();
            uniform_buffers.push(uniform_buffer);
        }

        uniform_buffers
    }

    fn create_descriptor_pool(device: &ash::Device, pool_size: u32) -> vk::DescriptorPool {
        let uniform_buffer_descriptor_pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(pool_size);

        let sampler_descriptor_pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(pool_size);

        let descriptor_pool_sizes = [
            uniform_buffer_descriptor_pool_size,
            sampler_descriptor_pool_size,
        ];

        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&descriptor_pool_sizes)
            .max_sets(pool_size);

        unsafe { device.create_descriptor_pool(&descriptor_pool_create_info, None) }
            .expect("Failed to create descriptor pool")
    }

    fn create_descriptor_sets(
        device: &ash::Device,
        descriptor_set_layout: vk::DescriptorSetLayout,
        descriptor_pool: vk::DescriptorPool,
        uniform_buffers: &Vec<Buffer>,
        images: &[Image],
    ) -> Vec<Vec<vk::DescriptorSet>> {
        let descriptor_set_layouts =
            vec![descriptor_set_layout; images.len() * MAX_FRAMES_IN_FLIGHT];

        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&descriptor_set_layouts);

        let flat_descriptor_sets = unsafe { device.allocate_descriptor_sets(&allocate_info) }
            .expect("Failed to allocate descriptor sets");

        let mut descriptor_sets =
            vec![vec![vk::DescriptorSet::null(); MAX_FRAMES_IN_FLIGHT]; images.len()];

        for (image_idx, image) in images.iter().enumerate() {
            for frame in 0..MAX_FRAMES_IN_FLIGHT {
                let flat_index = image_idx * MAX_FRAMES_IN_FLIGHT + frame;

                let descriptor_set = flat_descriptor_sets[flat_index];
                descriptor_sets[image_idx][frame] = descriptor_set;

                let descriptor_buffer_info = vk::DescriptorBufferInfo::default()
                    .buffer(uniform_buffers[frame].vk_buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE);
                let descriptor_buffer_infos = [descriptor_buffer_info];

                let write_descriptor_buffer = vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .buffer_info(&descriptor_buffer_infos);

                let descriptor_image_info = vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(image.view)
                    .sampler(image.sampler.unwrap());
                let descriptor_image_infos = [descriptor_image_info];

                let write_descriptor_image = vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .image_info(&descriptor_image_infos);

                let descriptor_writes = [write_descriptor_buffer, write_descriptor_image];

                unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) };
            }
        }
        descriptor_sets
    }

    fn create_command_buffers(
        device: &ash::Device,
        command_pool: vk::CommandPool,
    ) -> Vec<vk::CommandBuffer> {
        let command_buffers: Vec<vk::CommandBuffer> = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(command_buffers.capacity() as u32);

        unsafe { device.allocate_command_buffers(&command_buffer_allocate_info) }
            .expect("Failed to allocate command buffers")
    }

    fn create_sync_objects(
        device: &ash::Device,
        swapchain_image_count: usize,
    ) -> (Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>) {
        let mut image_available_semaphores: Vec<vk::Semaphore> =
            Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut render_finished_semaphores: Vec<vk::Semaphore> =
            Vec::with_capacity(swapchain_image_count);
        let mut in_flight_fences: Vec<vk::Fence> = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        let semaphore_create_info = vk::SemaphoreCreateInfo::default();
        let fence_create_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            image_available_semaphores.push(
                unsafe { device.create_semaphore(&semaphore_create_info, None) }
                    .expect("Failed to create image available semaphore"),
            );
            in_flight_fences.push(
                unsafe { device.create_fence(&fence_create_info, None) }
                    .expect("Failed to create in flight fence"),
            );
        }
        for _ in 0..swapchain_image_count {
            render_finished_semaphores.push(
                unsafe { device.create_semaphore(&semaphore_create_info, None) }
                    .expect("Failed to create render finished semaphore"),
            );
        }

        (
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
        )
    }

    fn record_command_buffer(&self, image_index: u32) {
        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };
        let clear_depth = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };
        let clear_values = [clear_color, clear_depth];

        let render_pass_begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass.vk_render_pass)
            .framebuffer(self.swapchain.framebuffers[image_index as usize])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain.extent,
            })
            .clear_values(&clear_values);

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

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default();

        let command_buffer = self.command_buffers[self.frame_in_flight_index];

        unsafe {
            self.device
                .ash_device
                .begin_command_buffer(command_buffer, &command_buffer_begin_info)
        }
        .expect("Failed to begin recording command buffer");

        unsafe {
            self.device.ash_device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );

            self.device.ash_device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.vk_pipeline,
            );

            self.device
                .ash_device
                .cmd_set_viewport(command_buffer, 0, &viewports);
            self.device
                .ash_device
                .cmd_set_scissor(command_buffer, 0, &scissors);

            for (i, primitive_info) in self.primitive_infos.iter().enumerate() {
                self.device.ash_device.cmd_bind_vertex_buffers(
                    command_buffer,
                    0,
                    &[self.vertex_buffer.vk_buffer],
                    &[0],
                );
                self.device.ash_device.cmd_bind_index_buffer(
                    command_buffer,
                    self.index_buffer.vk_buffer,
                    vk::DeviceSize::default(),
                    vk::IndexType::UINT32,
                );

                self.device.ash_device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.pipeline.layout,
                    0,
                    &[self.descriptor_sets[i][self.frame_in_flight_index]],
                    &[],
                );

                self.device.ash_device.cmd_draw_indexed(
                    command_buffer,
                    primitive_info.index_count,
                    1,
                    primitive_info.first_index,
                    0,
                    0,
                );
            }

            self.device.ash_device.cmd_end_render_pass(command_buffer);
        };

        unsafe { self.device.ash_device.end_command_buffer(command_buffer) }
            .expect("Failed to end recording command buffer");
    }

    fn update_uniform_buffer(&self) {
        let model =
            // Mat4::from_rotation_y(self.timer.elapsed().as_secs_f32() * 90.0_f32.to_radians());
            Mat4::IDENTITY;
        // let view = Mat4::look_at_rh(
        //     glam::Vec3::new(0.0, 2.0, 3.0),
        //     glam::Vec3::ZERO,
        //     glam::Vec3::Y,
        // );
        let mut proj = Mat4::perspective_rh(
            70.0_f32.to_radians(),
            self.width as f32 / self.height as f32,
            0.01,
            1000.0,
        );
        proj.y_axis *= -1.0;

        let uniform_buffer_data = UniformBufferData {
            model,
            view: self.camera_view,
            proj,
        };

        let mut uniform_align = unsafe {
            Align::new(
                self.uniform_buffers[self.frame_in_flight_index]
                    .p_data
                    .unwrap(),
                align_of::<f32>() as vk::DeviceSize,
                size_of::<UniformBufferData>() as vk::DeviceSize,
            )
        };
        uniform_align.copy_from_slice(&[uniform_buffer_data]);
    }

    pub fn draw_frame(&mut self) {
        unsafe {
            self.device.ash_device.wait_for_fences(
                &[self.in_flight_fences[self.frame_in_flight_index]],
                true,
                u64::MAX,
            )
        }
        .unwrap();

        let image_index: u32;

        match self
            .swapchain
            .acquire_next_image(self.image_available_semaphores[self.frame_in_flight_index])
        {
            None => {
                warn!("Recreating swapchain...");
                self.recreate_swapchain();
                return;
            }
            Some(index) => image_index = index,
        }

        unsafe {
            self.device
                .ash_device
                .reset_fences(&[self.in_flight_fences[self.frame_in_flight_index]])
        }
        .unwrap();

        unsafe {
            self.device.ash_device.reset_command_buffer(
                self.command_buffers[self.frame_in_flight_index],
                vk::CommandBufferResetFlags::empty(),
            )
        }
        .unwrap();

        self.record_command_buffer(image_index);

        let wait_semaphores = [self.image_available_semaphores[self.frame_in_flight_index]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [self.command_buffers[self.frame_in_flight_index]];
        let signal_semaphores = [self.render_finished_semaphores[image_index as usize]];

        self.update_uniform_buffer();

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);
        let submit_infos = [submit_info];

        unsafe {
            self.device.ash_device.queue_submit(
                self.device.graphics_queue,
                &submit_infos,
                self.in_flight_fences[self.frame_in_flight_index],
            )
        }
        .unwrap();

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

        self.frame_in_flight_index = (self.frame_in_flight_index + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    fn recreate_swapchain(&mut self) {
        self.swapchain.recreate(
            &self.instance,
            &self.adapter,
            &self.surface,
            self.render_pass.vk_render_pass,
            self.width,
            self.height,
        );
    }
}

pub struct QueueFamilyIndices {
    pub graphics_family: Option<u32>,
    pub present_family: Option<u32>,
}

impl QueueFamilyIndices {
    pub(crate) fn find_queue_families(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface: &Surface,
    ) -> Self {
        let mut indices = Self {
            graphics_family: None,
            present_family: None,
        };

        let queue_families = unsafe {
            instance
                .ash_instance
                .get_physical_device_queue_family_properties(physical_device)
        };

        let mut i = 0;
        for queue_family in queue_families {
            if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                indices.graphics_family = Some(i);
            }

            let present_support = unsafe {
                surface
                    .surface_instance
                    .get_physical_device_surface_support(physical_device, i, surface.surface_khr)
            }
            .expect("Failed to get adapter surface support");
            if present_support {
                indices.present_family = Some(i);
            }

            if indices.is_complete() {
                break;
            }
            i += 1;
        }

        Self {
            graphics_family: indices.graphics_family,
            present_family: indices.present_family,
        }
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.device.wait_idle();

            self.device
                .ash_device
                .destroy_descriptor_pool(self.descriptor_pool, None);

            self.device
                .ash_device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.device
                    .ash_device
                    .destroy_semaphore(self.image_available_semaphores[i], None);

                self.device
                    .ash_device
                    .destroy_fence(self.in_flight_fences[i], None);
            }

            for i in 0..self.swapchain.images.len() {
                self.device
                    .ash_device
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
            }

            self.device
                .ash_device
                .destroy_command_pool(self.command_pool, None);
        }
    }
}
