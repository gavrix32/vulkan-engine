use crate::unsafe_vk_try;
use crate::vulkan::device::Device;
use ash::vk;
use std::sync::Arc;

pub struct Descriptor {
    device: Arc<Device>,
    pub layout: vk::DescriptorSetLayout,
    pub pool: vk::DescriptorPool,
    pub set: vk::DescriptorSet,
}

impl Descriptor {
    pub fn new(
        device: Arc<Device>,
        bindings: &[vk::DescriptorSetLayoutBinding],
        binding_flags: &[vk::DescriptorBindingFlags],
        pool_sizes: &[vk::DescriptorPoolSize],
        variable_counts: &[u32],
    ) -> Self {
        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .pool_sizes(&pool_sizes)
            .max_sets(1);

        let pool = unsafe_vk_try!(
            device
                .ash_device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
        );

        let mut binding_flags_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(binding_flags);

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .bindings(&bindings)
            .push_next(&mut binding_flags_info);

        let layout = unsafe_vk_try!(
            device
                .ash_device
                .create_descriptor_set_layout(&layout_info, None)
        );
        let layouts = [layout];

        let mut variable_count_alloc_info =
            vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
                .descriptor_counts(variable_counts);

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts)
            .push_next(&mut variable_count_alloc_info);

        let sets = unsafe_vk_try!(device.ash_device.allocate_descriptor_sets(&alloc_info));
        let set = sets[0];

        Self {
            device,
            layout,
            pool,
            set,
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
