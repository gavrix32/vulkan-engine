use crate::vulkan::device::Device;
use ash::vk;
use std::sync::Arc;

pub struct Descriptor {
    device: Arc<Device>,
    pub layout: vk::DescriptorSetLayout,
    pub pool: vk::DescriptorPool,
    pub sets: Vec<vk::DescriptorSet>,
}

impl Descriptor {
    pub fn new(
        device: Arc<Device>,
        bindings: &[vk::DescriptorSetLayoutBinding],
        pool_sizes: &[vk::DescriptorPoolSize],
        max_sets: usize,
    ) -> Self {
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

        let layout = unsafe {
            device
                .ash_device
                .create_descriptor_set_layout(&layout_info, None)
        }
        .expect("Failed to create descriptor set layout");
        let layouts = vec![layout; max_sets];

        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(max_sets as u32);

        let pool = unsafe {
            device
                .ash_device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
        }
        .expect("Failed to create descriptor pool");

        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);

        let sets = unsafe { device.ash_device.allocate_descriptor_sets(&allocate_info) }
            .expect("Failed to allocate descriptor sets");

        Self {
            device,
            layout,
            pool,
            sets,
        }
    }

    pub fn update(&self, writes: &[vk::WriteDescriptorSet]) {
        unsafe { self.device.ash_device.update_descriptor_sets(&writes, &[]) };
    }
}

impl Drop for Descriptor {
    fn drop(&mut self) {
        unsafe {
            self.device
                .ash_device
                .destroy_descriptor_pool(self.pool, None);

            self.device
                .ash_device
                .destroy_descriptor_set_layout(self.layout, None);
        }
    }
}
