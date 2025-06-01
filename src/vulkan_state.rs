use crate::debug;
use ash::ext;
use ash::khr;
use ash::vk;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::collections::HashSet;
use std::ffi::{CStr, c_char};

const ADAPTER_EXTENSIONS: [&CStr; 1] = [khr::swapchain::NAME];

pub struct VulkanState {
    _entry: ash::Entry,
    instance: ash::Instance,
    debug_utils_instance_messenger:
        Option<(ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    surface: khr::surface::Instance,
    surface_khr: vk::SurfaceKHR,
    _adapter: vk::PhysicalDevice,
    device: ash::Device,
    _graphics_queue: vk::Queue,
    _present_queue: vk::Queue,
    swapchain: khr::swapchain::Device,
    swapchain_khr: vk::SwapchainKHR,
}

impl VulkanState {
    pub fn new(
        width: u32,
        height: u32,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> Self {
        let entry = unsafe { ash::Entry::load().expect("Failed to load Vulkan library") };

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

        let adapter = Self::pick_adapter(&instance, &surface, surface_khr);
        let (device, graphics_queue, present_queue) =
            Self::create_device(&instance, adapter, &surface, surface_khr);

        let (swapchain, swapchain_khr) = Self::create_swapchain(
            &instance,
            adapter,
            &device,
            &surface,
            surface_khr,
            width,
            height,
        );

        Self {
            _entry: entry,
            instance,
            debug_utils_instance_messenger,
            surface,
            surface_khr,
            _adapter: adapter,
            device,
            _graphics_queue: graphics_queue,
            _present_queue: present_queue,
            swapchain,
            swapchain_khr,
        }
    }

    fn create_instance(
        entry: &ash::Entry,
        display_handle: RawDisplayHandle,
        debug_create_info: &mut vk::DebugUtilsMessengerCreateInfoEXT<'static>,
    ) -> ash::Instance {
        let app_info = vk::ApplicationInfo::default()
            .application_name(c"Hello Triangle")
            .application_version(0)
            .engine_name(c"No Engine")
            .engine_version(0)
            .application_version(vk::make_api_version(0, 1, 0, 0));

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

        unsafe {
            entry
                .create_instance(&debug_instance_create_info, None)
                .expect("Failed to create VkInstance")
        }
    }

    fn check_validation_layer_support(entry: &ash::Entry) -> bool {
        let available_layers = unsafe {
            entry
                .enumerate_instance_layer_properties()
                .expect("Failed to enumerate layer properties")
        };
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
                .expect("Failed to create window surface")
        };
        (surface, surface_khr)
    }

    fn pick_adapter(
        instance: &ash::Instance,
        surface: &khr::surface::Instance,
        surface_khr: vk::SurfaceKHR,
    ) -> vk::PhysicalDevice {
        let adapters = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("Failed to enumerate physical devices")
        };
        if adapters.len() == 0 {
            panic!("Failed to find GPUs with Vulkan support");
        }

        for adapter in adapters {
            if Self::is_adapter_suitable(instance, adapter, surface, surface_khr) {
                return adapter;
            }
        }
        panic!("Failed to find a suitable GPU");
    }

    fn is_adapter_suitable(
        instance: &ash::Instance,
        adapter: vk::PhysicalDevice,
        surface: &khr::surface::Instance,
        surface_khr: vk::SurfaceKHR,
    ) -> bool {
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
        indices.is_complete() && extensions_supported && swapchain_adequate
    }

    fn check_adapter_extensions_support(
        instance: &ash::Instance,
        adapter: vk::PhysicalDevice,
    ) -> bool {
        let available_extensions = unsafe {
            instance
                .enumerate_device_extension_properties(adapter)
                .expect("Failed to enumerate adapter extension properties")
        };
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
        surface: &khr::surface::Instance,
        surface_khr: vk::SurfaceKHR,
    ) -> (ash::Device, vk::Queue, vk::Queue) {
        let indices =
            QueueFamilyIndices::find_queue_families(instance, adapter, surface, surface_khr);

        let queue_priority = 1.0f32;
        let queue_priorities = [queue_priority];

        let mut unique_queue_families = HashSet::new();
        unique_queue_families.insert(indices.graphics_family.unwrap());
        unique_queue_families.insert(indices.present_family.unwrap());

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

        let device = unsafe {
            instance
                .create_device(adapter, &device_create_info, None)
                .expect("Failed to create device")
        };

        let graphics_queue =
            unsafe { device.get_device_queue(indices.graphics_family.unwrap(), 0) };
        let present_queue = unsafe { device.get_device_queue(indices.present_family.unwrap(), 0) };

        (device, graphics_queue, present_queue)
    }

    fn create_swapchain(
        instance: &ash::Instance,
        adapter: vk::PhysicalDevice,
        device: &ash::Device,
        surface: &khr::surface::Instance,
        surface_khr: vk::SurfaceKHR,
        width: u32,
        height: u32,
    ) -> (khr::swapchain::Device, vk::SwapchainKHR) {
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

        let indices =
            QueueFamilyIndices::find_queue_families(instance, adapter, surface, surface_khr);
        let queue_family_indices = [
            indices.graphics_family.unwrap(),
            indices.present_family.unwrap(),
        ];

        if indices.graphics_family != indices.present_family {
            swapchain_create_info =
                swapchain_create_info.image_sharing_mode(vk::SharingMode::CONCURRENT);
            swapchain_create_info =
                swapchain_create_info.queue_family_indices(&queue_family_indices);
        } else {
            swapchain_create_info =
                swapchain_create_info.image_sharing_mode(vk::SharingMode::EXCLUSIVE);
        }

        let swapchain = khr::swapchain::Device::new(instance, device);
        let swapchain_khr = unsafe {
            swapchain
                .create_swapchain(&swapchain_create_info, None)
                .expect("Failed to create swapchain")
        };

        (swapchain, swapchain_khr)
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

            let present_support = unsafe {
                surface
                    .get_physical_device_surface_support(adapter, i, surface_khr)
                    .expect("Failed to get adapter surface support")
            };
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
            self.swapchain.destroy_swapchain(self.swapchain_khr, None);
            self.device.destroy_device(None);
            if let Some((instance, messenger)) = self.debug_utils_instance_messenger.take() {
                instance.destroy_debug_utils_messenger(messenger, None);
            }
            self.surface.destroy_surface(self.surface_khr, None);
            self.instance.destroy_instance(None)
        };
    }
}
