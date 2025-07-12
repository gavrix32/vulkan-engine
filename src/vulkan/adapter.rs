use crate::state::QueueFamilyIndices;
use crate::vulkan::instance::Instance;
use crate::vulkan::surface::Surface;
use crate::vulkan::swapchain;
use ash::{khr, vk};
use std::collections::HashSet;
use std::ffi::CStr;

pub const DEVICE_EXTENSIONS: [&CStr; 2] = [khr::swapchain::NAME, khr::shader_draw_parameters::NAME];

pub struct Adapter {
    pub physical_device: vk::PhysicalDevice,
    pub queue_family_indices: QueueFamilyIndices,
}

impl Adapter {
    pub fn new(instance: &Instance, surface: &Surface) -> Self {
        let adapters = unsafe { instance.ash_instance.enumerate_physical_devices() }
            .expect("Failed to enumerate physical devices");
        if adapters.len() == 0 {
            panic!("Failed to find GPUs with Vulkan support");
        }

        for physical_device in adapters {
            let (adapter_suitable, queue_family_indices) =
                is_physical_device_suitable(instance, physical_device, &surface);
            if adapter_suitable {
                return Self {
                    physical_device,
                    queue_family_indices,
                };
            }
        }
        panic!("Failed to find a suitable GPU");
    }
}

fn is_physical_device_suitable(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    surface: &Surface,
) -> (bool, QueueFamilyIndices) {
    let indices = QueueFamilyIndices::find_queue_families(instance, physical_device, surface);

    let extensions_supported = check_physical_device_extensions_support(instance, physical_device);

    let mut swapchain_adequate = false;
    if extensions_supported {
        let swapchain_support = swapchain::SupportDetails::query_support(physical_device, surface);
        swapchain_adequate =
            !swapchain_support.formats.is_empty() && !swapchain_support.present_modes.is_empty();
    }
    (
        indices.is_complete() && extensions_supported && swapchain_adequate,
        indices,
    )
}

fn check_physical_device_extensions_support(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> bool {
    let available_extensions = unsafe {
        instance
            .ash_instance
            .enumerate_device_extension_properties(physical_device)
    }
    .expect("Failed to enumerate adapter extension properties");

    let mut required_extensions = HashSet::from(DEVICE_EXTENSIONS);

    for extension in available_extensions {
        let extension_name_cstr = unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) };
        required_extensions.remove(extension_name_cstr);
    }
    required_extensions.is_empty()
}
