use std::collections::HashSet;
use std::ffi::c_char;
use ash::vk;
use crate::vulkan::adapter::{Adapter, DEVICE_EXTENSIONS};
use crate::vulkan::instance::Instance;

pub struct Device {
    pub ash_device: ash::Device,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
}

impl Device {
    pub fn new(instance: &Instance, adapter: &Adapter) -> Self {
        let queue_priority = 1.0f32;
        let queue_priorities = [queue_priority];

        let mut unique_queue_families = HashSet::new();
        unique_queue_families.insert(adapter.queue_family_indices.graphics_family.unwrap());
        unique_queue_families.insert(adapter.queue_family_indices.present_family.unwrap());

        let mut queue_create_infos = Vec::new();

        for queue_family in unique_queue_families {
            let queue_create_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family)
                .queue_priorities(&queue_priorities);
            queue_create_infos.push(queue_create_info);
        }

        let adapter_extension_name_pointers: Vec<*const c_char> =
            DEVICE_EXTENSIONS.iter().map(|s| s.as_ptr()).collect();

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&*adapter_extension_name_pointers);

        let ash_device = unsafe { instance.ash_instance.create_device(adapter.physical_device, &device_create_info, None) }
            .expect("Failed to create device");

        let graphics_queue =
            unsafe { ash_device.get_device_queue(adapter.queue_family_indices.graphics_family.unwrap(), 0) };
        let present_queue =
            unsafe { ash_device.get_device_queue(adapter.queue_family_indices.present_family.unwrap(), 0) };

        Self {
            ash_device,
            graphics_queue,
            present_queue
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe { self.ash_device.destroy_device(None); }
    }
}