use crate::unsafe_vk_try;
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
        let adapters = unsafe_vk_try!(instance.ash_instance.enumerate_physical_devices());
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

    pub fn max_usable_sample_count(&self, instance: &Instance) -> vk::SampleCountFlags {
        let properties = unsafe {
            instance
                .ash_instance
                .get_physical_device_properties(self.physical_device)
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
    let available_extensions = unsafe_vk_try!(
        instance
            .ash_instance
            .enumerate_device_extension_properties(physical_device)
    );

    let mut required_extensions = HashSet::from(DEVICE_EXTENSIONS);

    for extension in available_extensions {
        let extension_name_cstr = unsafe { CStr::from_ptr(extension.extension_name.as_ptr()) };
        required_extensions.remove(extension_name_cstr);
    }
    required_extensions.is_empty()
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

            let present_support = unsafe_vk_try!(
                surface
                    .surface_instance
                    .get_physical_device_surface_support(physical_device, i, surface.surface_khr)
            );
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
