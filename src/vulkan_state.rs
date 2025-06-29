use crate::debug;
use ash::ext;
use ash::khr;
use ash::vk;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::collections::HashSet;
use std::ffi::{CStr, c_char};

const ADAPTER_EXTENSIONS: [&CStr; 2] = [khr::swapchain::NAME, khr::shader_draw_parameters::NAME];
const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct VulkanState {
    _entry: ash::Entry,
    instance: ash::Instance,
    debug_utils_instance_messenger:
        Option<(ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    surface_instance: khr::surface::Instance,
    surface_khr: vk::SurfaceKHR,
    _adapter: vk::PhysicalDevice,
    _queue_family_indices: QueueFamilyIndices,
    device: ash::Device,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    swapchain_device: khr::swapchain::Device,
    swapchain_khr: vk::SwapchainKHR,
    _swapchain_images: Vec<vk::Image>,
    _swapchain_image_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_image_views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    swapchain_framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,
}

impl VulkanState {
    pub fn new(
        width: u32,
        height: u32,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> Self {
        let entry = unsafe { ash::Entry::load() }.expect("Failed to load Vulkan library");

        let mut debug_utils_messenger_create_info = debug::create_debug_messenger_create_info();

        let instance = Self::create_instance(
            &entry,
            display_handle,
            &mut debug_utils_messenger_create_info,
        );

        let debug_utils_instance_messenger =
            debug::setup_debug_messenger(&entry, &instance, &debug_utils_messenger_create_info);

        let (surface, surface_khr) =
            Self::create_surface(&entry, &instance, display_handle, window_handle);

        let (adapter, queue_family_indices) = Self::pick_adapter(&instance, &surface, surface_khr);
        let (device, graphics_queue, present_queue) =
            Self::create_device(&instance, adapter, &queue_family_indices);

        let (swapchain, swapchain_khr, swapchain_images, swapchain_image_format, swapchain_extent) =
            Self::create_swapchain(
                &instance,
                adapter,
                &device,
                &queue_family_indices,
                &surface,
                surface_khr,
                width,
                height,
            );

        let swapchain_image_views =
            Self::create_image_views(&swapchain_images, &swapchain_image_format, &device);

        let render_pass = Self::create_render_pass(&device, swapchain_image_format);

        let (pipeline_layout, pipeline) = Self::create_graphics_pipeline(&device, render_pass);

        let swapchain_framebuffers = Self::create_framebuffers(
            &device,
            &swapchain_image_views,
            render_pass,
            swapchain_extent,
        );

        let command_pool = Self::create_command_pool(&device, &queue_family_indices);

        let command_buffers = Self::create_command_buffers(&device, command_pool);

        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) =
            Self::create_sync_objects(&device, swapchain_images.len());

        Self {
            _entry: entry,
            instance,
            debug_utils_instance_messenger,
            surface_instance: surface,
            surface_khr,
            _adapter: adapter,
            _queue_family_indices: queue_family_indices,
            device,
            graphics_queue,
            present_queue,
            swapchain_device: swapchain,
            swapchain_khr,
            _swapchain_images: swapchain_images,
            _swapchain_image_format: swapchain_image_format,
            swapchain_extent,
            swapchain_image_views,
            render_pass,
            pipeline_layout,
            pipeline,
            swapchain_framebuffers,
            command_pool,
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            current_frame: 0,
        }
    }

    fn create_instance(
        entry: &ash::Entry,
        display_handle: RawDisplayHandle,
        debug_create_info: &mut vk::DebugUtilsMessengerCreateInfoEXT<'static>,
    ) -> ash::Instance {
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Hello Triangle")
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(c"No Engine")
            .engine_version(0)
            .api_version(vk::make_api_version(0, 1, 3, 0));

        let required_extensions = Self::get_required_extensions(display_handle);

        let mut debug_instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&required_extensions);

        let layer_cstring_pointers = debug::get_validation_layer_cstring_pointers();

        if debug::ENABLE_VALIDATION_LAYERS {
            if !Self::check_validation_layer_support(&entry) {
                panic!("validation layers requested, but not available!");
            }
            debug_instance_create_info = debug_instance_create_info
                .enabled_layer_names(&layer_cstring_pointers.1)
                .push_next(debug_create_info);
        }

        unsafe { entry.create_instance(&debug_instance_create_info, None) }
            .expect("Failed to create VkInstance")
    }

    fn check_validation_layer_support(entry: &ash::Entry) -> bool {
        let available_layers = unsafe { entry.enumerate_instance_layer_properties() }
            .expect("Failed to enumerate layer properties");
        for layer_name in debug::VALIDATION_LAYERS {
            let mut layer_found = false;

            for layer_properties in &available_layers {
                let name = unsafe { CStr::from_ptr(layer_properties.layer_name.as_ptr()) };
                if layer_name == name.to_str().unwrap() {
                    layer_found = true;
                    break;
                }
            }
            if !layer_found {
                return false;
            }
        }
        true
    }

    fn get_required_extensions(display_handle: RawDisplayHandle) -> Vec<*const c_char> {
        let mut extensions = ash_window::enumerate_required_extensions(display_handle)
            .unwrap()
            .to_vec();

        if debug::ENABLE_VALIDATION_LAYERS {
            extensions.push(ext::debug_utils::NAME.as_ptr());
        }
        extensions
    }

    fn create_surface(
        entry: &ash::Entry,
        instance: &ash::Instance,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> (khr::surface::Instance, vk::SurfaceKHR) {
        let surface = khr::surface::Instance::new(&entry, &instance);
        let surface_khr = unsafe {
            ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)
        }
        .expect("Failed to create window surface");
        (surface, surface_khr)
    }

    fn pick_adapter(
        instance: &ash::Instance,
        surface: &khr::surface::Instance,
        surface_khr: vk::SurfaceKHR,
    ) -> (vk::PhysicalDevice, QueueFamilyIndices) {
        let adapters = unsafe { instance.enumerate_physical_devices() }
            .expect("Failed to enumerate physical devices");
        if adapters.len() == 0 {
            panic!("Failed to find GPUs with Vulkan support");
        }

        for adapter in adapters {
            let (adapter_suitable, queue_family_indices) =
                Self::is_adapter_suitable(instance, adapter, surface, surface_khr);
            if adapter_suitable {
                return (adapter, queue_family_indices);
            }
        }
        panic!("Failed to find a suitable GPU");
    }

    fn is_adapter_suitable(
        instance: &ash::Instance,
        adapter: vk::PhysicalDevice,
        surface: &khr::surface::Instance,
        surface_khr: vk::SurfaceKHR,
    ) -> (bool, QueueFamilyIndices) {
        let indices =
            QueueFamilyIndices::find_queue_families(instance, adapter, surface, surface_khr);
        let extensions_supported = Self::check_adapter_extensions_support(instance, adapter);
        let mut swapchain_adequate = false;
        if extensions_supported {
            let swapchain_support =
                SwapchainSupportDetails::query_swapchain_support(adapter, surface, surface_khr);
            swapchain_adequate = !swapchain_support.formats.is_empty()
                && !swapchain_support.present_modes.is_empty();
        }
        (
            indices.is_complete() && extensions_supported && swapchain_adequate,
            indices,
        )
    }

    fn check_adapter_extensions_support(
        instance: &ash::Instance,
        adapter: vk::PhysicalDevice,
    ) -> bool {
        let available_extensions =
            unsafe { instance.enumerate_device_extension_properties(adapter) }
                .expect("Failed to enumerate adapter extension properties");
        let mut required_extensions = HashSet::from(ADAPTER_EXTENSIONS);
        for extension in available_extensions {
            let extension_name_cstr = unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) };
            required_extensions.remove(extension_name_cstr);
        }
        required_extensions.is_empty()
    }

    fn create_device(
        instance: &ash::Instance,
        adapter: vk::PhysicalDevice,
        queue_family_indices: &QueueFamilyIndices,
    ) -> (ash::Device, vk::Queue, vk::Queue) {
        let queue_priority = 1.0f32;
        let queue_priorities = [queue_priority];

        let mut unique_queue_families = HashSet::new();
        unique_queue_families.insert(queue_family_indices.graphics_family.unwrap());
        unique_queue_families.insert(queue_family_indices.present_family.unwrap());

        let mut queue_create_infos = Vec::new();

        for queue_family in unique_queue_families {
            let queue_create_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family)
                .queue_priorities(&queue_priorities);
            queue_create_infos.push(queue_create_info);
        }

        let adapter_extension_name_pointers: Vec<*const c_char> =
            ADAPTER_EXTENSIONS.iter().map(|s| s.as_ptr()).collect();

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&*adapter_extension_name_pointers);

        let device = unsafe { instance.create_device(adapter, &device_create_info, None) }
            .expect("Failed to create device");

        let graphics_queue =
            unsafe { device.get_device_queue(queue_family_indices.graphics_family.unwrap(), 0) };
        let present_queue =
            unsafe { device.get_device_queue(queue_family_indices.present_family.unwrap(), 0) };

        (device, graphics_queue, present_queue)
    }

    fn create_swapchain(
        instance: &ash::Instance,
        adapter: vk::PhysicalDevice,
        device: &ash::Device,
        queue_family_indices: &QueueFamilyIndices,
        surface: &khr::surface::Instance,
        surface_khr: vk::SurfaceKHR,
        width: u32,
        height: u32,
    ) -> (
        khr::swapchain::Device,
        vk::SwapchainKHR,
        Vec<vk::Image>,
        vk::Format,
        vk::Extent2D,
    ) {
        let swapchain_support_details =
            SwapchainSupportDetails::query_swapchain_support(adapter, surface, surface_khr);

        let surface_format = Self::choose_surface_format(swapchain_support_details.formats);
        let present_mode = Self::choose_present_mode(swapchain_support_details.present_modes);
        let extent = Self::choose_extent(swapchain_support_details.capabilities, width, height);

        let mut image_count = swapchain_support_details.capabilities.min_image_count + 1;
        if swapchain_support_details.capabilities.max_image_count > 0
            && image_count > swapchain_support_details.capabilities.max_image_count
        {
            image_count = swapchain_support_details.capabilities.max_image_count;
        }

        let mut swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface_khr)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .pre_transform(swapchain_support_details.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());

        let indices = [
            queue_family_indices.graphics_family.unwrap(),
            queue_family_indices.present_family.unwrap(),
        ];

        if queue_family_indices.graphics_family != queue_family_indices.present_family {
            swapchain_create_info =
                swapchain_create_info.image_sharing_mode(vk::SharingMode::CONCURRENT);
            swapchain_create_info = swapchain_create_info.queue_family_indices(&indices);
        } else {
            swapchain_create_info =
                swapchain_create_info.image_sharing_mode(vk::SharingMode::EXCLUSIVE);
        }

        let swapchain = khr::swapchain::Device::new(instance, device);
        let swapchain_khr = unsafe { swapchain.create_swapchain(&swapchain_create_info, None) }
            .expect("Failed to create swapchain");
        let swapchain_images = unsafe { swapchain.get_swapchain_images(swapchain_khr) }
            .expect("Failed to get swapchain images");

        (
            swapchain,
            swapchain_khr,
            swapchain_images,
            surface_format.format,
            extent,
        )
    }

    fn create_image_views(
        swapchain_images: &Vec<vk::Image>,
        swapchain_image_format: &vk::Format,
        device: &ash::Device,
    ) -> Vec<vk::ImageView> {
        let mut image_views = Vec::new();

        for i in 0..swapchain_images.len() {
            let image_view_create_info = vk::ImageViewCreateInfo::default()
                .image(swapchain_images[i])
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(*swapchain_image_format)
                .components(
                    vk::ComponentMapping::default()
                        .r(vk::ComponentSwizzle::IDENTITY)
                        .g(vk::ComponentSwizzle::IDENTITY)
                        .b(vk::ComponentSwizzle::IDENTITY)
                        .a(vk::ComponentSwizzle::IDENTITY),
                )
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );

            let image_view = unsafe { device.create_image_view(&image_view_create_info, None) }
                .expect("Failed to create image views");
            image_views.push(image_view)
        }
        image_views
    }

    fn create_graphics_pipeline(
        device: &ash::Device,
        render_pass: vk::RenderPass,
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

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default();

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
            .front_face(vk::FrontFace::CLOCKWISE)
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

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default();

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

    fn create_framebuffers(
        device: &ash::Device,
        swapchain_image_views: &Vec<vk::ImageView>,
        render_pass: vk::RenderPass,
        swapchain_extent: vk::Extent2D,
    ) -> Vec<vk::Framebuffer> {
        let mut framebuffers = Vec::new();

        for i in 0..swapchain_image_views.len() {
            let attachments = [swapchain_image_views[i]];
            let framebuffer_create_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(swapchain_extent.width)
                .height(swapchain_extent.height)
                .layers(1);

            let framebuffer = unsafe { device.create_framebuffer(&framebuffer_create_info, None) }
                .expect("Failed to create framebuffer");

            framebuffers.push(framebuffer);
        }
        framebuffers
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

    fn record_command_buffer(
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        render_pass: vk::RenderPass,
        framebuffer: vk::Framebuffer,
        swapchain_extent: vk::Extent2D,
        pipeline: vk::Pipeline,
    ) {
        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };
        let clear_color_values = [clear_color];

        let render_pass_begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(render_pass)
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: swapchain_extent,
            })
            .clear_values(&clear_color_values);

        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(swapchain_extent.width as f32)
            .height(swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);
        let viewports = [viewport];

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain_extent,
        };
        let scissors = [scissor];

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default();

        unsafe { device.begin_command_buffer(command_buffer, &command_buffer_begin_info) }
            .expect("Failed to begin recording command buffer");

        unsafe {
            device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );

            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_set_viewport(command_buffer, 0, &viewports);
            device.cmd_set_scissor(command_buffer, 0, &scissors);
            device.cmd_draw(command_buffer, 3, 1, 0, 0);

            device.cmd_end_render_pass(command_buffer);
        };

        unsafe { device.end_command_buffer(command_buffer) }
            .expect("Failed to end recording command buffer");
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

    fn create_shader_module(device: &ash::Device, words: &[u32]) -> vk::ShaderModule {
        let shader_module_create_info = vk::ShaderModuleCreateInfo::default().code(words);
        unsafe { device.create_shader_module(&shader_module_create_info, None) }
            .expect("Failed to create shader module")
    }

    fn choose_surface_format(surface_formats: Vec<vk::SurfaceFormatKHR>) -> vk::SurfaceFormatKHR {
        for surface_format in &surface_formats {
            if surface_format.format == vk::Format::B8G8R8A8_SRGB
                && surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            {
                return *surface_format;
            }
        }
        surface_formats[0]
    }

    fn choose_present_mode(present_modes: Vec<vk::PresentModeKHR>) -> vk::PresentModeKHR {
        for present_mode in &present_modes {
            if *present_mode == vk::PresentModeKHR::IMMEDIATE {
                return *present_mode;
            }
        }
        vk::PresentModeKHR::FIFO
    }

    fn choose_extent(
        capabilities: vk::SurfaceCapabilitiesKHR,
        width: u32,
        height: u32,
    ) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            return capabilities.current_extent;
        }
        let mut actual_extent = vk::Extent2D { width, height };
        actual_extent.width = actual_extent.width.clamp(
            capabilities.min_image_extent.width,
            capabilities.max_image_extent.width,
        );
        actual_extent.height = actual_extent.height.clamp(
            capabilities.min_image_extent.height,
            capabilities.max_image_extent.height,
        );

        actual_extent
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
                    .expect("Failed to create image available semaphore")
            );
            in_flight_fences.push(
                unsafe { device.create_fence(&fence_create_info, None) }
                    .expect("Failed to create in flight fence")
            );
        }
        for _ in 0..swapchain_image_count {
            render_finished_semaphores.push(
                unsafe { device.create_semaphore(&semaphore_create_info, None) }
                    .expect("Failed to create render finished semaphore")
            );
        }
        
        (
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
        )
    }

    pub fn draw_frame(&mut self) {
        unsafe {
            self.device
                .wait_for_fences(&[self.in_flight_fences[self.current_frame]], true, u64::MAX)
        }
        .unwrap();

        unsafe { self.device.reset_fences(&[self.in_flight_fences[self.current_frame]]) }.unwrap();

        let (image_index, _) = unsafe {
            self.swapchain_device.acquire_next_image(
                self.swapchain_khr,
                u64::MAX,
                self.image_available_semaphores[self.current_frame],
                vk::Fence::null(),
            )
        }
        .unwrap();

        unsafe {
            self.device
                .reset_command_buffer(self.command_buffers[self.current_frame], vk::CommandBufferResetFlags::empty())
        }
        .unwrap();

        Self::record_command_buffer(
            &self.device,
            self.command_buffers[self.current_frame],
            self.render_pass,
            self.swapchain_framebuffers[image_index as usize],
            self.swapchain_extent,
            self.pipeline,
        );

        let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [self.command_buffers[self.current_frame]];
        let signal_semaphores = [self.render_finished_semaphores[image_index as usize]];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);
        let submit_infos = [submit_info];

        unsafe {
            self.device
                .queue_submit(self.graphics_queue, &submit_infos, self.in_flight_fences[self.current_frame])
        }
        .unwrap();

        let swapchains = [self.swapchain_khr];
        let image_indices = [image_index];

        let present_info_khr = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        unsafe {
            self.swapchain_device
                .queue_present(self.present_queue, &present_info_khr)
        }
        .unwrap();
        
        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
}

struct QueueFamilyIndices {
    graphics_family: Option<u32>,
    present_family: Option<u32>,
}

impl QueueFamilyIndices {
    fn find_queue_families(
        instance: &ash::Instance,
        adapter: vk::PhysicalDevice,
        surface: &khr::surface::Instance,
        surface_khr: vk::SurfaceKHR,
    ) -> Self {
        let mut indices = Self {
            graphics_family: None,
            present_family: None,
        };

        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(adapter) };

        let mut i = 0;
        for queue_family in queue_families {
            if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                indices.graphics_family = Some(i);
            }

            let present_support =
                unsafe { surface.get_physical_device_surface_support(adapter, i, surface_khr) }
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

    fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
    }
}

struct SwapchainSupportDetails {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupportDetails {
    fn query_swapchain_support(
        adapter: vk::PhysicalDevice,
        surface: &khr::surface::Instance,
        surface_khr: vk::SurfaceKHR,
    ) -> Self {
        unsafe {
            Self {
                capabilities: surface
                    .get_physical_device_surface_capabilities(adapter, surface_khr)
                    .expect("Failed to get adapter surface capabilities"),
                formats: surface
                    .get_physical_device_surface_formats(adapter, surface_khr)
                    .expect("Failed to get adapter surface formats"),
                present_modes: surface
                    .get_physical_device_surface_present_modes(adapter, surface_khr)
                    .expect("Failed to get adapter surface present modes"),
            }
        }
    }
}

impl Drop for VulkanState {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();

            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.device
                    .destroy_semaphore(self.image_available_semaphores[i], None);
                
                self.device.destroy_fence(self.in_flight_fences[i], None);
            }
            
            for i in 0..self._swapchain_images.len() {
                self.device
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
            }

            self.device.destroy_command_pool(self.command_pool, None);

            for framebuffer in &self.swapchain_framebuffers {
                self.device.destroy_framebuffer(*framebuffer, None);
            }

            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_render_pass(self.render_pass, None);

            for image_view in &self.swapchain_image_views {
                self.device.destroy_image_view(*image_view, None);
            }

            self.swapchain_device
                .destroy_swapchain(self.swapchain_khr, None);
            self.device.destroy_device(None);

            if let Some((instance, messenger)) = self.debug_utils_instance_messenger.take() {
                instance.destroy_debug_utils_messenger(messenger, None);
            }

            self.surface_instance
                .destroy_surface(self.surface_khr, None);
            self.instance.destroy_instance(None)
        };
    }
}
