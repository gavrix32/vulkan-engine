use ash::util::Align;
use ash::vk;
use log::warn;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::ffi;
use std::mem::offset_of;
use std::time::Instant;
use crate::vulkan::adapter::Adapter;
use crate::vulkan::device::Device;
use crate::vulkan::instance::Instance;
use crate::vulkan::surface::Surface;
use crate::vulkan::swapchain::Swapchain;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 2],
    color: [f32; 3],
}

impl Vertex {
    fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(offset_of!(Vertex, position) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Vertex, color) as u32),
        ]
    }
}

const VERTICES: [Vertex; 4] = [
    Vertex {
        position: [-0.5, -0.5],
        color: [1.0, 0.0, 0.0],
    },
    Vertex {
        position: [0.5, -0.5],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        position: [0.5, 0.5],
        color: [0.0, 0.0, 1.0],
    },
    Vertex {
        position: [-0.5, 0.5],
        color: [1.0, 1.0, 1.0],
    },
];

const INDICES: [u16; 6] = [0, 1, 2, 2, 3, 0];

#[repr(C)]
#[allow(dead_code)]
#[derive(Clone, Copy)]
struct UniformBufferData {
    model: glam::Mat4,
    view: glam::Mat4,
    proj: glam::Mat4,
}

pub struct State {
    swapchain: Swapchain,
    render_pass: vk::RenderPass,
    device: Device,
    adapter: Adapter,
    surface: Surface,
    instance: Instance,

    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,

    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,

    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,

    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,
    uniform_buffers_mapped: Vec<*mut ffi::c_void>,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    frame_in_flight_index: usize,

    timer: Instant,

    pub framebuffer_resized: bool,
    pub width: u32,
    pub height: u32,
}

impl State {
    pub fn new(
        width: u32,
        height: u32,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> Self {
        let instance = Instance::new(display_handle);
        let surface = Surface::new(&instance, display_handle, window_handle);
        let adapter = Adapter::new(&instance, &surface);
        let device = Device::new(&instance, &adapter);
        let render_pass = Self::create_render_pass(&device.ash_device, Swapchain::get_format(&adapter, &surface));
        let swapchain = Swapchain::new(&instance, &adapter, &device, &surface, render_pass, width, height);

        let descriptor_set_layout = Self::create_descriptor_set_layout(&device.ash_device);
        let descriptor_set_layouts = [descriptor_set_layout];

        let (pipeline_layout, pipeline) =
            Self::create_graphics_pipeline(&device.ash_device, render_pass, &descriptor_set_layouts);

        let command_pool = Self::create_command_pool(&device.ash_device, &adapter.queue_family_indices);

        let (vertex_buffer, vertex_buffer_memory) = Self::create_vertex_buffer(
            &instance.ash_instance,
            adapter.physical_device,
            &device.ash_device,
            device.graphics_queue,
            command_pool,
        );

        let (index_buffer, index_buffer_memory) = Self::create_index_buffer(
            &instance.ash_instance,
            adapter.physical_device,
            &device.ash_device,
            device.graphics_queue,
            command_pool,
        );

        let (uniform_buffers, uniform_buffers_memory, uniform_buffers_mapped) =
            Self::create_uniform_buffers(&instance.ash_instance, adapter.physical_device, &device.ash_device);

        let descriptor_pool = Self::create_descriptor_pool(&device.ash_device);
        let descriptor_sets = Self::create_descriptor_sets(
            &device.ash_device,
            descriptor_set_layout,
            descriptor_pool,
            &uniform_buffers,
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
            pipeline_layout,
            pipeline,

            command_pool,
            command_buffers,

            vertex_buffer,
            vertex_buffer_memory,

            index_buffer,
            index_buffer_memory,

            uniform_buffers,
            uniform_buffers_memory,
            uniform_buffers_mapped,

            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            frame_in_flight_index: 0,

            timer: Instant::now(),

            framebuffer_resized: false,
            width,
            height,
        }
    }

    fn create_render_pass(
        device: &ash::Device,
        swapchain_image_format: vk::Format,
    ) -> vk::RenderPass {
        let color_attachment_description = vk::AttachmentDescription::default()
            .format(swapchain_image_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
        let color_attachment_descriptions = [color_attachment_description];

        let color_attachment_reference = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let color_attachment_references = [color_attachment_reference];

        let subpass_description = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_references);
        let subpass_descriptions = [subpass_description];

        let subpass_dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
        let subpass_dependencies = [subpass_dependency];

        let render_pass_create_info = vk::RenderPassCreateInfo::default()
            .attachments(&color_attachment_descriptions)
            .subpasses(&subpass_descriptions)
            .dependencies(&subpass_dependencies);

        unsafe { device.create_render_pass(&render_pass_create_info, None) }
            .expect("Failed to create render pass")
    }

    fn create_descriptor_set_layout(device: &ash::Device) -> vk::DescriptorSetLayout {
        let uniform_buffer_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);
        let uniform_buffer_layout_bindings = [uniform_buffer_layout_binding];

        let descriptor_set_layout_create_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&uniform_buffer_layout_bindings);

        unsafe { device.create_descriptor_set_layout(&descriptor_set_layout_create_info, None) }
            .expect("Failed to create descriptor set layout")
    }

    fn create_shader_module(device: &ash::Device, words: &[u32]) -> vk::ShaderModule {
        let shader_module_create_info = vk::ShaderModuleCreateInfo::default().code(words);
        unsafe { device.create_shader_module(&shader_module_create_info, None) }
            .expect("Failed to create shader module")
    }

    fn create_graphics_pipeline(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
    ) -> (vk::PipelineLayout, vk::Pipeline) {
        let mut vertex_shader_file = std::fs::File::open("src/shaders/spirv/vertex.spv")
            .expect("Failed to open vertex shader file");
        let vertex_shader_code = ash::util::read_spv(&mut vertex_shader_file).unwrap();

        let mut fragment_shader_file = std::fs::File::open("src/shaders/spirv/fragment.spv")
            .expect("Failed to open fragment shader file");
        let fragment_shader_code = ash::util::read_spv(&mut fragment_shader_file).unwrap();

        let vertex_shader_module =
            Self::create_shader_module(device, vertex_shader_code.as_slice());
        let fragment_shader_module =
            Self::create_shader_module(device, fragment_shader_code.as_slice());

        let vertex_shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader_module)
            .name(c"main");
        let fragment_shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader_module)
            .name(c"main");

        let shader_stage_create_infos = [vertex_shader_stage_info, fragment_shader_stage_info];

        let binding_descriptions = [Vertex::get_binding_description()];
        let attribute_descriptions = Vertex::get_attribute_descriptions();

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer_state = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment_state = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);
        let color_blend_attachment_states = [color_blend_attachment_state];

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&color_blend_attachment_states);

        let pipeline_layout_create_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&descriptor_set_layouts);

        let pipeline_layout =
            unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None) }
                .expect("Failed to create pipeline layout");

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stage_create_infos)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let graphics_pipelines = unsafe {
            device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_create_info],
                None,
            )
        }
        .expect("Failed to create graphics pipelines");

        unsafe {
            device.destroy_shader_module(vertex_shader_module, None);
            device.destroy_shader_module(fragment_shader_module, None);
        }

        (pipeline_layout, graphics_pipelines[0])
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

    fn find_memory_type(
        instance: &ash::Instance,
        adapter: vk::PhysicalDevice,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> u32 {
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(adapter) };

        for i in 0..memory_properties.memory_type_count {
            if (type_filter & (1 << i) != 0)
                && memory_properties.memory_types[i as usize].property_flags & properties
                    == properties
            {
                return i;
            }
        }
        panic!("Failed to find suitable memory type");
    }

    fn create_buffer(
        instance: &ash::Instance,
        adapter: vk::PhysicalDevice,
        device: &ash::Device,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_flags: vk::MemoryPropertyFlags,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let vertex_buffer = unsafe { device.create_buffer(&buffer_create_info, None) }
            .expect("Failed to create buffer");

        let memory_requirements = unsafe { device.get_buffer_memory_requirements(vertex_buffer) };

        let memory_allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(Self::find_memory_type(
                instance,
                adapter,
                memory_requirements.memory_type_bits,
                memory_flags,
            ));

        let vertex_buffer_memory = unsafe { device.allocate_memory(&memory_allocate_info, None) }
            .expect("Failed to allocate buffer memory");

        unsafe { device.bind_buffer_memory(vertex_buffer, vertex_buffer_memory, 0) }
            .expect("Failed to bind buffer memory");

        (vertex_buffer, vertex_buffer_memory)
    }

    fn copy_buffer(
        device: &ash::Device,
        graphics_queue: vk::Queue,
        src_buffer: vk::Buffer,
        dst_buffer: vk::Buffer,
        size: vk::DeviceSize,
        command_pool: vk::CommandPool,
    ) {
        let mut command_buffers: Vec<vk::CommandBuffer> = Vec::with_capacity(1);

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(command_buffers.capacity() as u32);

        command_buffers = unsafe { device.allocate_command_buffers(&command_buffer_allocate_info) }
            .expect("Failed to allocate command buffers");

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { device.begin_command_buffer(command_buffers[0], &command_buffer_begin_info) }
            .expect("Failed to begin recording command buffer");

        let copy_region = vk::BufferCopy::default().size(size);

        unsafe {
            device.cmd_copy_buffer(command_buffers[0], src_buffer, dst_buffer, &[copy_region]);
            device.end_command_buffer(command_buffers[0])
        }
        .expect("Failed to end recording command buffer");

        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);

        unsafe {
            device
                .queue_submit(graphics_queue, &[submit_info], vk::Fence::null())
                .unwrap();
            device.queue_wait_idle(graphics_queue).unwrap();
            device.free_command_buffers(command_pool, &command_buffers);
        }
    }

    fn create_vertex_buffer(
        instance: &ash::Instance,
        adapter: vk::PhysicalDevice,
        device: &ash::Device,
        graphics_queue: vk::Queue,
        command_pool: vk::CommandPool,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let size = (size_of::<Vertex>() * VERTICES.len()) as vk::DeviceSize;

        let (staging_buffer, staging_buffer_memory) = Self::create_buffer(
            instance,
            adapter,
            device,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        let data_ptr = unsafe {
            device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())
        }
        .expect("Failed to map staging buffer memory");

        let mut vertex_align =
            unsafe { Align::new(data_ptr, align_of::<Vertex>() as vk::DeviceSize, size) };
        vertex_align.copy_from_slice(&VERTICES);

        unsafe { device.unmap_memory(staging_buffer_memory) };

        let (vertex_buffer, vertex_buffer_memory) = Self::create_buffer(
            instance,
            adapter,
            device,
            size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        Self::copy_buffer(
            device,
            graphics_queue,
            staging_buffer,
            vertex_buffer,
            size,
            command_pool,
        );

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        }

        (vertex_buffer, vertex_buffer_memory)
    }

    fn create_index_buffer(
        instance: &ash::Instance,
        adapter: vk::PhysicalDevice,
        device: &ash::Device,
        graphics_queue: vk::Queue,
        command_pool: vk::CommandPool,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_size = (size_of::<u16>() * INDICES.len()) as vk::DeviceSize;

        let (staging_buffer, staging_buffer_memory) = Self::create_buffer(
            instance,
            adapter,
            device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        let data_ptr = unsafe {
            device.map_memory(
                staging_buffer_memory,
                0,
                buffer_size,
                vk::MemoryMapFlags::empty(),
            )
        }
        .expect("Failed to map staging buffer memory");

        let mut vertex_align =
            unsafe { Align::new(data_ptr, align_of::<u16>() as vk::DeviceSize, buffer_size) };
        vertex_align.copy_from_slice(&INDICES);

        unsafe { device.unmap_memory(staging_buffer_memory) };

        let (index_buffer, index_buffer_memory) = Self::create_buffer(
            instance,
            adapter,
            device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        Self::copy_buffer(
            device,
            graphics_queue,
            staging_buffer,
            index_buffer,
            buffer_size,
            command_pool,
        );

        unsafe {
            device.destroy_buffer(staging_buffer, None);
            device.free_memory(staging_buffer_memory, None);
        }

        (index_buffer, index_buffer_memory)
    }

    fn create_uniform_buffers(
        instance: &ash::Instance,
        adapter: vk::PhysicalDevice,
        device: &ash::Device,
    ) -> (
        Vec<vk::Buffer>,
        Vec<vk::DeviceMemory>,
        Vec<*mut ffi::c_void>,
    ) {
        let buffer_size = size_of::<UniformBufferData>() as vk::DeviceSize;

        let mut uniform_buffers: Vec<vk::Buffer> = Vec::new();
        let mut uniform_buffers_memory: Vec<vk::DeviceMemory> = Vec::new();
        let mut uniform_buffers_mapped: Vec<*mut ffi::c_void> = Vec::new();

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let (buffer, buffer_memory) = Self::create_buffer(
                instance,
                adapter,
                device,
                buffer_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            uniform_buffers.push(buffer);
            uniform_buffers_memory.push(buffer_memory);

            uniform_buffers_mapped.push(
                unsafe {
                    device.map_memory(buffer_memory, 0, buffer_size, vk::MemoryMapFlags::empty())
                }
                .expect("Failed to map uniform buffer memory"),
            );
        }

        (
            uniform_buffers,
            uniform_buffers_memory,
            uniform_buffers_mapped,
        )
    }

    fn create_descriptor_pool(device: &ash::Device) -> vk::DescriptorPool {
        let descriptor_pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32);
        let descriptor_pool_sizes = [descriptor_pool_size];

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
        uniform_buffers: &Vec<vk::Buffer>,
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
                .buffer(uniform_buffers[i])
                .offset(0)
                .range(vk::WHOLE_SIZE);
            let descriptor_buffer_infos = [descriptor_buffer_info];

            let descriptor_write = vk::WriteDescriptorSet::default()
                .dst_set(descriptor_sets[i])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .buffer_info(&descriptor_buffer_infos);
            let descriptor_writes = [descriptor_write];

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
        let clear_color_values = [clear_color];

        let render_pass_begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.swapchain.framebuffers[image_index as usize])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain.extent,
            })
            .clear_values(&clear_color_values);

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
            self.device.ash_device
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
                self.pipeline,
            );

            self.device.ash_device.cmd_set_viewport(command_buffer, 0, &viewports);
            self.device.ash_device.cmd_set_scissor(command_buffer, 0, &scissors);

            self.device.ash_device
                .cmd_bind_vertex_buffers(command_buffer, 0, &[self.vertex_buffer], &[0]);
            self.device.ash_device.cmd_bind_index_buffer(
                command_buffer,
                self.index_buffer,
                vk::DeviceSize::default(),
                vk::IndexType::UINT16,
            );

            self.device.ash_device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_sets[self.frame_in_flight_index]],
                &[],
            );

            self.device.ash_device
                .cmd_draw_indexed(command_buffer, INDICES.len() as u32, 1, 0, 0, 0);

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
                self.uniform_buffers_mapped[self.frame_in_flight_index],
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

        let acquire_result = unsafe {
            self.swapchain.swapchain_device.acquire_next_image(
                self.swapchain.swapchain_khr,
                u64::MAX,
                self.image_available_semaphores[self.frame_in_flight_index],
                vk::Fence::null(),
            )
        };

        let image_index: u32;

        match acquire_result {
            Ok((index, is_suboptimal)) => {
                if is_suboptimal {
                    warn!("Swapchain is suboptimal, recreating...");
                    self.swapchain.recreate(&self.instance, &self.adapter, &self.device, &self.surface, self.render_pass, self.width, self.height);
                    return;
                }
                image_index = index;
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                warn!("Swapchain is out of date, recreating...");
                self.swapchain.recreate(&self.instance, &self.adapter, &self.device, &self.surface, self.render_pass, self.width, self.height);
                return;
            }
            Err(e) => {
                panic!("Failed to acquire next image: {:?}", e);
            }
        }

        unsafe {
            self.device.ash_device
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
            self.swapchain.swapchain_device
                .queue_present(self.device.present_queue, &present_info_khr)
        };

        match present_result {
            Ok(is_suboptimal) => {
                if is_suboptimal {
                    warn!("Swapchain is suboptimal, recreating...");
                    self.swapchain.recreate(&self.instance, &self.adapter, &self.device, &self.surface, self.render_pass, self.width, self.height);
                }
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                warn!("Swapchain is out of date, recreating...");
                self.swapchain.recreate(&self.instance, &self.adapter, &self.device, &self.surface, self.render_pass, self.width, self.height);
            }
            Err(e) => {
                panic!("Failed to present swapchain image: {:?}", e);
            }
        }

        if self.framebuffer_resized {
            self.framebuffer_resized = false;
            self.swapchain.recreate(&self.instance, &self.adapter, &self.device, &self.surface, self.render_pass, self.width, self.height);
        }

        self.frame_in_flight_index = (self.frame_in_flight_index + 1) % MAX_FRAMES_IN_FLIGHT;
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

        let queue_families =
            unsafe { instance.ash_instance.get_physical_device_queue_family_properties(physical_device) };

        let mut i = 0;
        for queue_family in queue_families {
            if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                indices.graphics_family = Some(i);
            }

            let present_support =
                unsafe { surface.surface_instance.get_physical_device_surface_support(physical_device, i, surface.surface_khr) }
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

impl Drop for State {
    fn drop(&mut self) {
        unsafe {
            self.device.ash_device.device_wait_idle().unwrap();

            self.swapchain.cleanup(&self.device);

            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.device.ash_device.destroy_buffer(self.uniform_buffers[i], None);
                self.device.ash_device
                    .free_memory(self.uniform_buffers_memory[i], None);
            }

            self.device.ash_device
                .destroy_descriptor_pool(self.descriptor_pool, None);

            self.device.ash_device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            self.device.ash_device.destroy_buffer(self.index_buffer, None);
            self.device.ash_device.free_memory(self.index_buffer_memory, None);

            self.device.ash_device.destroy_buffer(self.vertex_buffer, None);
            self.device.ash_device.free_memory(self.vertex_buffer_memory, None);

            self.device.ash_device.destroy_pipeline(self.pipeline, None);
            self.device.ash_device
                .destroy_pipeline_layout(self.pipeline_layout, None);

            self.device.ash_device.destroy_render_pass(self.render_pass, None);

            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.device.ash_device
                    .destroy_semaphore(self.image_available_semaphores[i], None);

                self.device.ash_device.destroy_fence(self.in_flight_fences[i], None);
            }

            for i in 0..self.swapchain.images.len() {
                self.device.ash_device
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
            }

            self.device.ash_device.destroy_command_pool(self.command_pool, None);
        }
    }
}
