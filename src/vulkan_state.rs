use crate::debug;
use ash::ext::debug_utils;
use ash::khr;
use ash::vk;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::ffi::{CStr, c_char};

pub struct VulkanState {
    _entry: ash::Entry,
    instance: ash::Instance,
    debug_utils_instance_messenger: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    surface: khr::surface::Instance,
    surface_khr: vk::SurfaceKHR,
    _adapter: vk::PhysicalDevice,
    device: ash::Device,
    _queue: vk::Queue,
}

impl VulkanState {
    pub fn new(display_handle: RawDisplayHandle, window_handle: RawWindowHandle) -> Self {
        let entry = unsafe { ash::Entry::load().expect("Failed to load Vulkan library") };

        let mut debug_utils_messenger_create_info = debug::create_debug_messenger_create_info();

        let instance = Self::create_instance(
            &entry,
            display_handle,
            &mut debug_utils_messenger_create_info,
        );

        let debug_utils_instance_messenger =
            debug::setup_debug_messenger(&entry, &instance, &debug_utils_messenger_create_info);

        // surface
        let (surface, surface_khr) =
            Self::create_surface(&entry, &instance, display_handle, window_handle);

        let adapter = Self::pick_adapter(&instance);
        let (device, queue) = Self::create_device(&instance, adapter);

        Self {
            _entry: entry,
            instance,
            debug_utils_instance_messenger,
            surface,
            surface_khr,
            _adapter: adapter,
            device,
            _queue: queue,
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
            extensions.push(debug_utils::NAME.as_ptr());
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

    fn pick_adapter(instance: &ash::Instance) -> vk::PhysicalDevice {
        let adapters = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("Failed to enumerate physical devices")
        };
        if adapters.len() == 0 {
            panic!("Failed to find GPUs with Vulkan support");
        }

        for adapter in adapters {
            if Self::is_adapter_suitable(instance, adapter) {
                return adapter;
            }
        }
        panic!("Failed to find a suitable GPU");
    }

    fn is_adapter_suitable(instance: &ash::Instance, adapter: vk::PhysicalDevice) -> bool {
        let indices = QueueFamilyIndices::find_queue_families(instance, adapter);
        indices.is_complete()
    }

    fn create_device(
        instance: &ash::Instance,
        adapter: vk::PhysicalDevice,
    ) -> (ash::Device, vk::Queue) {
        let indices = QueueFamilyIndices::find_queue_families(instance, adapter);

        let queue_priority = 1.0f32;
        let queue_priorities = [queue_priority];

        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(indices.graphics_family.unwrap())
            .queue_priorities(&queue_priorities);
        let queue_create_infos = [queue_create_info];

        let device_create_info =
            vk::DeviceCreateInfo::default().queue_create_infos(&queue_create_infos);

        let device = unsafe {
            instance
                .create_device(adapter, &device_create_info, None)
                .expect("Failed to create device")
        };

        let queue = unsafe { device.get_device_queue(indices.graphics_family.unwrap(), 0) };

        (device, queue)
    }
}

struct QueueFamilyIndices {
    graphics_family: Option<u32>,
}

impl QueueFamilyIndices {
    fn find_queue_families(instance: &ash::Instance, adapter: vk::PhysicalDevice) -> Self {
        let mut indices = Self {
            graphics_family: None,
        };

        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(adapter) };

        let mut i = 0;
        for queue_family in queue_families {
            if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                indices.graphics_family = Some(i);
            }
            if indices.is_complete() {
                break;
            }
            i += 1;
        }

        Self {
            graphics_family: indices.graphics_family,
        }
    }

    fn is_complete(&self) -> bool {
        self.graphics_family.is_some()
    }
}

impl Drop for VulkanState {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            if let Some((instance, messenger)) = self.debug_utils_instance_messenger.take() {
                instance.destroy_debug_utils_messenger(messenger, None);
            }
            self.surface.destroy_surface(self.surface_khr, None);
            self.instance.destroy_instance(None)
        };
    }
}
