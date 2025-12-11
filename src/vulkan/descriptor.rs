use crate::unsafe_vk_try;
use crate::vulkan::buffer::Buffer;
use crate::vulkan::device::Device;
use crate::vulkan::image::{ImageView, Sampler};
use ash::vk;
use log::error;
use std::sync::Arc;

pub struct DescriptorSetLayout {
    device: Arc<Device>,
    pub handle: vk::DescriptorSetLayout,
}

impl DescriptorSetLayout {
    pub fn builder(device: Arc<Device>) -> DescriptorSetLayoutBuilder {
        DescriptorSetLayoutBuilder::new(device)
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        unsafe {
            self.device
                .handle
                .destroy_descriptor_set_layout(self.handle, None);
        }
    }
}

pub struct DescriptorSetLayoutBuilder {
    device: Arc<Device>,
    bindings: Vec<vk::DescriptorSetLayoutBinding<'static>>,
    binding_flags: Vec<vk::DescriptorBindingFlags>,
}

impl DescriptorSetLayoutBuilder {
    pub fn new(device: Arc<Device>) -> Self {
        Self {
            device,
            bindings: Vec::new(),
            binding_flags: Vec::new(),
        }
    }

    pub fn binding(
        mut self,
        binding: u32,
        ty: vk::DescriptorType,
        count: u32,
        stage_flags: vk::ShaderStageFlags,
        binding_flags: vk::DescriptorBindingFlags,
    ) -> Self {
        let layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(binding)
            .descriptor_type(ty)
            .descriptor_count(count)
            .stage_flags(stage_flags);

        self.bindings.push(layout_binding);
        self.binding_flags.push(binding_flags);
        self
    }

    pub fn build(self) -> DescriptorSetLayout {
        let mut binding_flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
            .binding_flags(&self.binding_flags);

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .bindings(&self.bindings)
            .push_next(&mut binding_flags_info);

        let vk_layout = unsafe_vk_try!(
            self.device
                .handle
                .create_descriptor_set_layout(&layout_info, None)
        );

        DescriptorSetLayout {
            device: self.device,
            handle: vk_layout,
        }
    }
}

#[derive(Default)]
pub struct DescriptorWriter<'a> {
    writes: Vec<vk::WriteDescriptorSet<'a>>,
    buffer_infos: Vec<vk::DescriptorBufferInfo>,
    image_infos: Vec<vk::DescriptorImageInfo>,
}

impl<'a> DescriptorWriter<'a> {
    pub fn buffer(mut self, binding: u32, ty: vk::DescriptorType, buffer: &'a Buffer) -> Self {
        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(buffer.handle)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        self.buffer_infos.push(buffer_info);

        let write = vk::WriteDescriptorSet::default()
            .dst_binding(binding)
            .descriptor_type(ty)
            .descriptor_count(1);

        self.writes.push(write);
        self
    }

    pub fn image(
        mut self,
        binding: u32,
        ty: vk::DescriptorType,
        layout: vk::ImageLayout,
        image_view: &'a ImageView,
        sampler: &'a Sampler,
    ) -> Self {
        let image_info = vk::DescriptorImageInfo::default()
            .image_layout(layout)
            .image_view(image_view.handle)
            .sampler(sampler.handle);
        self.image_infos.push(image_info);

        let write = vk::WriteDescriptorSet::default()
            .dst_binding(binding)
            .descriptor_type(ty)
            .descriptor_count(1);

        self.writes.push(write);
        self
    }

    pub fn images(
        mut self,
        binding: u32,
        ty: vk::DescriptorType,
        layout: vk::ImageLayout,
        image_views: &'a [ImageView],
        sampler: &'a Sampler,
    ) -> Self {
        for i in 0..image_views.len() {
            let image_info = vk::DescriptorImageInfo::default()
                .image_layout(layout)
                .image_view(image_views[i].handle)
                .sampler(sampler.handle);
            self.image_infos.push(image_info);
        }

        let write = vk::WriteDescriptorSet::default()
            .dst_binding(binding)
            .descriptor_type(ty)
            .descriptor_count(image_views.len() as u32);

        self.writes.push(write);
        self
    }

    pub fn update(mut self, device: Arc<Device>, set: vk::DescriptorSet) {
        for write in &mut self.writes {
            write.dst_set = set;
        }

        let mut buffer_index = 0;
        let mut image_index = 0;

        for write in &mut self.writes {
            match write.descriptor_type {
                vk::DescriptorType::UNIFORM_BUFFER
                | vk::DescriptorType::STORAGE_BUFFER
                | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => {
                    write.p_buffer_info = &self.buffer_infos[buffer_index];
                    buffer_index += write.descriptor_count as usize;
                }
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                | vk::DescriptorType::SAMPLED_IMAGE
                | vk::DescriptorType::STORAGE_IMAGE
                | vk::DescriptorType::INPUT_ATTACHMENT => {
                    write.p_image_info = &self.image_infos[image_index];
                    image_index += write.descriptor_count as usize;
                }
                _ => error!("Unknown descriptor type"),
            }
        }

        unsafe {
            device.handle.update_descriptor_sets(&self.writes, &[]);
        }
    }
}

pub struct DescriptorPool {
    device: Arc<Device>,
    vk_pool: vk::DescriptorPool,
}

impl DescriptorPool {
    pub fn new(device: Arc<Device>, max_sets: u32, pool_sizes: &[vk::DescriptorPoolSize]) -> Self {
        let pool_create_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .max_sets(max_sets)
            .pool_sizes(pool_sizes);

        let vk_pool = unsafe_vk_try!(
            device
                .handle
                .create_descriptor_pool(&pool_create_info, None)
        );

        Self { device, vk_pool }
    }

    pub fn allocate(&self, layout: &DescriptorSetLayout, variable_count: u32) -> vk::DescriptorSet {
        let layouts = [layout.handle];

        let variable_counts = [variable_count];

        let mut variable_count_alloc_info =
            vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
                .descriptor_counts(&variable_counts);

        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.vk_pool)
            .set_layouts(&layouts)
            .push_next(&mut variable_count_alloc_info);

        let sets = unsafe_vk_try!(self.device.handle.allocate_descriptor_sets(&allocate_info));

        sets[0]
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.device
                .handle
                .destroy_descriptor_pool(self.vk_pool, None);
        }
    }
}
