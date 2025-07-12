use crate::vulkan::adapter::Adapter;
use crate::vulkan::device::Device;
use crate::vulkan::instance::Instance;
use ash::vk;
use std::ffi;

pub struct Buffer {
    pub vk_buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    pub p_data: Option<*mut ffi::c_void>,
}

impl Buffer {
    pub fn new(
        instance: &Instance,
        adapter: &Adapter,
        device: &Device,
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
            vk_buffer: buffer,
            memory,
            size,
            p_data: None,
        }
    }

    pub fn copy(
        &self,
        device: &Device,
        graphics_queue: vk::Queue,
        dst_buffer: &Self,
        command_pool: vk::CommandPool,
    ) {
        let mut command_buffers: Vec<vk::CommandBuffer> = Vec::with_capacity(1);

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(command_buffers.capacity() as u32);

        command_buffers = unsafe {
            device
                .ash_device
                .allocate_command_buffers(&command_buffer_allocate_info)
        }
        .expect("Failed to allocate command buffers");

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device
                .ash_device
                .begin_command_buffer(command_buffers[0], &command_buffer_begin_info)
        }
        .expect("Failed to begin recording command buffer");

        let copy_region = vk::BufferCopy::default().size(self.size);

        unsafe {
            device.ash_device.cmd_copy_buffer(
                command_buffers[0],
                self.vk_buffer,
                dst_buffer.vk_buffer,
                &[copy_region],
            );
            device.ash_device.end_command_buffer(command_buffers[0])
        }
        .expect("Failed to end recording command buffer");

        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);

        unsafe {
            device
                .ash_device
                .queue_submit(graphics_queue, &[submit_info], vk::Fence::null())
                .unwrap();
            device.ash_device.queue_wait_idle(graphics_queue).unwrap();
            device
                .ash_device
                .free_command_buffers(command_pool, &command_buffers);
        }
    }

    pub fn map_memory(&mut self, device: &Device) {
        self.p_data = Some(
            unsafe {
                device
                    .ash_device
                    .map_memory(self.memory, 0, self.size, vk::MemoryMapFlags::empty())
            }
            .expect("Failed to map staging buffer memory"),
        );
    }

    pub fn unmap_memory(&self, device: &Device) {
        unsafe { device.ash_device.unmap_memory(self.memory) };
    }

    fn find_memory_type(
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

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.ash_device.destroy_buffer(self.vk_buffer, None);
            device.ash_device.free_memory(self.memory, None);
        }
    }
}
