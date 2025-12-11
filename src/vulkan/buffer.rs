use crate::unsafe_vk_try;
use crate::vulkan::adapter::Adapter;
use crate::vulkan::device::Device;
use crate::vulkan::encoder::Encoder;
use crate::vulkan::pool::CmdPool;
use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use std::sync::{Arc, Mutex};

pub struct Buffer {
    device: Arc<Device>,
    pub handle: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub allocator: Arc<Mutex<Allocator>>,
    pub size: vk::DeviceSize,
}

impl Buffer {
    pub fn new(
        device: Arc<Device>,
        allocator: Arc<Mutex<Allocator>>,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Self {
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let handle = unsafe_vk_try!(device.handle.create_buffer(&buffer_create_info, None));

        let requirements = unsafe { device.handle.get_buffer_memory_requirements(handle) };

        let allocation = allocator
            .lock()
            .expect("Failed to lock GPU allocator mutex")
            .allocate(&AllocationCreateDesc {
                name: "Buffer allocation",
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .expect("Failed to allocate buffer");

        unsafe_vk_try!(device.handle.bind_buffer_memory(
            handle,
            allocation.memory(),
            allocation.offset()
        ));

        Self {
            device,
            handle,
            allocator,
            allocation: Some(allocation),
            size,
        }
    }

    pub fn copy(&self, graphics_queue: vk::Queue, adapter: &Adapter, dst_buffer: &Self) {
        let copy_region = vk::BufferCopy::default().size(self.size);

        let single_time_pool = CmdPool::new(self.device.clone(), &adapter);
        let encoder = Encoder::begin_single_time(self.device.clone(), &single_time_pool);
        encoder.cmd_copy_buffer(self.handle, dst_buffer.handle, &[copy_region]);
        encoder.end_single_time(graphics_queue);
    }

    pub fn update<T: bytemuck::NoUninit>(&mut self, data: &[T]) {
        let bytes = bytemuck::cast_slice(data);

        if let Some(allocation) = self.allocation.as_mut() {
            allocation
                .mapped_slice_mut()
                .expect("No data pointer in buffer")[..bytes.len()]
                .copy_from_slice(bytes);
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_buffer(self.handle, None);
            if let Some(allocation) = self.allocation.take() {
                self.allocator
                    .lock()
                    .expect("Failed to lock GPU allocator mutex")
                    .free(allocation)
                    .expect("Failed to free GPU allocation");
            }
        }
    }
}
