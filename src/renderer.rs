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
use gltf::{Document, buffer};
use log::warn;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::mem::offset_of;
use std::sync::Arc;
use std::time::Instant;

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

// fn add_indices(accessor: &Accessor, buffer_data: &[buffer::Data], dst: &mut Vec<u32>) {
//     let view = accessor.view().unwrap();
//     let buffer = &buffer_data[view.buffer().index()];
//     let offset = view.offset() + accessor.offset();
//     let length = accessor.count();
//     let stride = accessor.size();
//
//     let bytes = &buffer[offset..offset + length * stride];
//     match accessor.data_type() {
//         DataType::U8 => {
//             let slice: &[u8] = bytemuck::cast_slice(bytes);
//             dst.extend(slice.iter().map(|&i| i as u32));
//         }
//         DataType::U16 => {
//             let slice: &[u16] = bytemuck::cast_slice(bytes);
//             dst.extend(slice.iter().map(|&i| i as u32));
//         }
//         DataType::U32 => {
//             let slice: &[u32] = bytemuck::cast_slice(bytes);
//             dst.extend(slice.iter().cloned());
//         }
//         _ => panic!("Unsupported index type"),
//     }
// let slice: &[u16] = bytemuck::cast_slice(bytes);
// dst.extend(slice.iter().map(|&i| i as u32));
// }

fn get_mesh_data(document: Document, buffer_data: Vec<buffer::Data>) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|buffer| Some(&buffer_data[buffer.index()]));

            let positions: Vec<[f32; 3]> = reader.read_positions().unwrap().collect();
            let tex_coords: Vec<[f32; 2]> = reader
                .read_tex_coords(0)
                .map(|tc| tc.into_f32().collect())
                .unwrap_or_default();

            if let Some(read_indices) = reader.read_indices() {
                for index in read_indices.into_u32() {
                    let i = index as usize;
                    let vertex = Vertex {
                        position: positions[i],
                        color: [1.0, 1.0, 1.0],
                        tex_coord: tex_coords.get(i).copied().unwrap_or([0.0, 0.0]),
                    };
                    indices.push(vertices.len() as u32);
                    vertices.push(vertex);
                }
            }

            // // indices
            // if let Some(accessor) = primitive.indices() {
            //     match accessor.data_type() {
            //         DataType::U8 | DataType::U16 | DataType::U32 => {
            //             add_indices(&accessor, &buffer_data, &mut indices)
            //         }
            //         _ => panic!("Unsupported index type"),
            //     }
            // }
            //
            // // texture coords
            // let texture_coords: Option<&[[f32; 2]]> = primitive
            //     .get(&gltf::Semantic::TexCoords(0))
            //     .map(|accessor| {
            //         let view = accessor.view().unwrap();
            //         let buffer = &buffer_data[view.buffer().index()];
            //         let offset = view.offset() + accessor.offset();
            //         let bytes = &buffer[offset..offset + accessor.count() * 8]; // 2 * f32 == 8 bytes
            //         bytemuck::cast_slice(bytes)
            //     });
            //
            // // vertices
            // if let Some(accessor) = primitive.get(&gltf::Semantic::Positions) {
            //     let view = accessor.view().unwrap();
            //     let buffer = &buffer_data[view.buffer().index()];
            //     let offset = view.offset() + accessor.offset();
            //     let length = accessor.count();
            //     let stride = view.stride().unwrap_or(size_of::<&[f32; 3]>());
            //
            //     // let bytes = &buffer[offset..offset + length * 12];
            //     // let positions: &[[f32; 3]] = bytemuck::cast_slice(bytes);
            //
            //     for i in 0..length {
            //         let start = offset + i * stride;
            //         let end = start + size_of::<[f32; 3]>();
            //         let bytes = &buffer[start..end];
            //         let position: [f32; 3] = *bytemuck::from_bytes(bytes);
            //
            //         vertices.push(Vertex {
            //             position,
            //             color: [1.0, 1.0, 1.0],
            //             tex_coord: texture_coords.map_or([0.0, 0.0], |tc| tc[i]),
            //         });
            //     }

            // vertices = Vec::with_capacity(length);
            // for i in 0..length {
            //     let start = offset + i * stride;
            //     let end = start + size_of::<&[f32; 3]>();
            //     let bytes = &buffer[start..end];
            //
            //     vertices.push(Vertex {
            //         position: *bytemuck::from_bytes(bytes),
            //         color: [1.0, 1.0, 1.0],
            //         tex_coord: texture_coords.map_or([0.0, 0.0], |tc| tc[i]),
            //     });
            // }
            // }
        }
    }
    (vertices, indices)
}

fn rgb_to_rgba(rgb_data: &[u8]) -> Vec<u8> {
    rgb_data
        .chunks_exact(3)
        .flat_map(|rgb_pixel| [rgb_pixel[0], rgb_pixel[1], rgb_pixel[2], 255])
        .collect()
}

#[repr(C)]
#[allow(dead_code)]
#[derive(Clone, Copy)]
struct UniformBufferData {
    model: glam::Mat4,
    view: glam::Mat4,
    proj: glam::Mat4,
}

pub struct Renderer {
    uniform_buffers: Vec<Buffer>,
    index_buffer: Buffer,
    vertex_buffer: Buffer,

    _image: Image,

    command_buffers: Vec<vk::CommandBuffer>,
    command_pool: vk::CommandPool,

    pipeline: Pipeline,
    descriptor_sets: Vec<vk::DescriptorSet>,
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

    timer: Instant,

    pub framebuffer_resized: bool,
    pub width: u32,
    pub height: u32,

    index_count: u32,
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

        let (document, buffers, images) =
            gltf::import_slice(include_bytes!("../models/viking_room.glb"))
                .expect("Failed to load model");

        let (vertices, indices) = get_mesh_data(document, buffers);

        let image_data = &images[0];

        let image_bytes = match image_data.format {
            gltf::image::Format::R8G8B8A8 => &image_data.pixels,
            gltf::image::Format::R8G8B8 => &rgb_to_rgba(&image_data.pixels),
            _ => panic!("Unsupported texture format: {:?}", image_data.format),
        };

        let image = Image::from_bytes(
            image_bytes,
            image_data.width,
            image_data.height,
            &instance,
            &adapter,
            device.clone(),
            command_pool,
        );

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

        let descriptor_pool = Self::create_descriptor_pool(&device.ash_device);
        let descriptor_sets = Self::create_descriptor_sets(
            &device.ash_device,
            descriptor_set_layout,
            descriptor_pool,
            &uniform_buffers,
            &image,
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

            _image: image,

            vertex_buffer,
            index_buffer,
            uniform_buffers,

            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            frame_in_flight_index: 0,

            timer: Instant::now(),

            framebuffer_resized: false,
            width,
            height,

            index_count: indices.len() as u32,
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

    fn create_descriptor_pool(device: &ash::Device) -> vk::DescriptorPool {
        let uniform_buffer_descriptor_pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32);

        let sampler_descriptor_pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32);

        let descriptor_pool_sizes = [
            uniform_buffer_descriptor_pool_size,
            sampler_descriptor_pool_size,
        ];

        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&descriptor_pool_sizes)
            .max_sets(MAX_FRAMES_IN_FLIGHT as u32);

        unsafe { device.create_descriptor_pool(&descriptor_pool_create_info, None) }
            .expect("Failed to create descriptor pool")
    }

    fn create_descriptor_sets(
        device: &ash::Device,
        descriptor_set_layout: vk::DescriptorSetLayout,
        descriptor_pool: vk::DescriptorPool,
        uniform_buffers: &Vec<Buffer>,
        image: &Image,
    ) -> Vec<vk::DescriptorSet> {
        let mut descriptor_set_layouts = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            descriptor_set_layouts.push(descriptor_set_layout);
        }

        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&descriptor_set_layouts);

        let descriptor_sets =
            unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info) }
                .expect("Failed to allocate descriptor sets");

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let descriptor_buffer_info = vk::DescriptorBufferInfo::default()
                .buffer(uniform_buffers[i].vk_buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let descriptor_buffer_infos = [descriptor_buffer_info];

            let write_descriptor_buffer = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_sets[i])
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
                .dst_set(descriptor_sets[i])
                .dst_binding(1)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .image_info(&descriptor_image_infos);

            let descriptor_writes = [write_descriptor_buffer, write_descriptor_image];

            unsafe { device.update_descriptor_sets(&descriptor_writes, &[]) };
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
                &[self.descriptor_sets[self.frame_in_flight_index]],
                &[],
            );

            self.device
                .ash_device
                .cmd_draw_indexed(command_buffer, self.index_count, 1, 0, 0, 0);

            self.device.ash_device.cmd_end_render_pass(command_buffer);
        };

        unsafe { self.device.ash_device.end_command_buffer(command_buffer) }
            .expect("Failed to end recording command buffer");
    }

    fn update_uniform_buffer(&self) {
        let model =
            glam::Mat4::from_rotation_z(self.timer.elapsed().as_secs_f32() * 90.0_f32.to_radians());
        let view = glam::Mat4::look_at_rh(
            glam::Vec3::new(2.0, 2.0, 2.0),
            glam::Vec3::ZERO,
            glam::Vec3::Z,
        );
        let mut proj = glam::Mat4::perspective_rh(
            45.0_f32.to_radians(),
            self.width as f32 / self.height as f32,
            0.1,
            10.0,
        );
        let mut proj_array = proj.to_cols_array_2d();
        proj_array[1][1] *= -1.0;
        proj = glam::Mat4::from_cols_array_2d(&proj_array);

        let uniform_buffer_data = UniformBufferData { model, view, proj };

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
