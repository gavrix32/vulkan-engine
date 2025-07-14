use crate::vulkan::adapter::Adapter;
use crate::vulkan::command_buffer::CommandBuffer;
use crate::vulkan::device::Device;
use crate::vulkan::instance::Instance;
use ash::vk;
use std::ffi;
use std::sync::Arc;

pub struct Buffer {
    device: Arc<Device>,
    pub vk_buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    pub p_data: Option<*mut ffi::c_void>,
}

impl Buffer {
    pub fn new(
        instance: &Instance,
        adapter: &Adapter,
        device: Arc<Device>,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_flags: vk::MemoryPropertyFlags,
    ) -> Self {
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.ash_device.create_buffer(&buffer_create_info, None) }
            .expect("Failed to create buffer");

        let memory_requirements =
            unsafe { device.ash_device.get_buffer_memory_requirements(buffer) };

        let memory_allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(Self::find_memory_type(
                instance,
                adapter,
                memory_requirements.memory_type_bits,
                memory_flags,
            ));

        let memory = unsafe {
            device
                .ash_device
                .allocate_memory(&memory_allocate_info, None)
        }
        .expect("Failed to allocate buffer memory");

        unsafe { device.ash_device.bind_buffer_memory(buffer, memory, 0) }
            .expect("Failed to bind buffer memory");

        Self {
            device,
            vk_buffer: buffer,
            memory,
            size,
            p_data: None,
        }
    }

    pub fn copy(
        &self,
        graphics_queue: vk::Queue,
        dst_buffer: &Self,
        command_pool: vk::CommandPool,
    ) {
        let command_buffers =
            CommandBuffer::begin_single_time_commands(self.device.clone(), command_pool);

        let copy_region = vk::BufferCopy::default().size(self.size);

        unsafe {
            self.device.ash_device.cmd_copy_buffer(
                command_buffers[0],
                self.vk_buffer,
                dst_buffer.vk_buffer,
                &[copy_region],
            );
        }

        CommandBuffer::end_single_time_commands(
            self.device.clone(),
            command_pool,
            command_buffers,
            graphics_queue,
        );
    }

    pub fn map_memory(&mut self) {
        self.p_data = Some(
            unsafe {
                self.device.ash_device.map_memory(
                    self.memory,
                    0,
                    self.size,
                    vk::MemoryMapFlags::empty(),
                )
            }
            .expect("Failed to map staging buffer memory"),
        );
    }

    pub fn unmap_memory(&self) {
        unsafe { self.device.ash_device.unmap_memory(self.memory) };
    }

    pub fn find_memory_type(
        instance: &Instance,
        adapter: &Adapter,
        type_filter: u32,
        properties: vk::MemoryPropertyFlags,
    ) -> u32 {
        let memory_properties = unsafe {
            instance
                .ash_instance
                .get_physical_device_memory_properties(adapter.physical_device)
        };

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
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.ash_device.destroy_buffer(self.vk_buffer, None);
            self.device.ash_device.free_memory(self.memory, None);
        }
    }
}
