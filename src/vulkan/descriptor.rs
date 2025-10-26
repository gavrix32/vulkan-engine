use crate::unsafe_vk_try;
use crate::vulkan::buffer::Buffer;
use crate::vulkan::device::Device;
use crate::vulkan::image::Image;
use ash::vk;
use std::sync::Arc;

pub struct DescriptorSetLayout {
    device: Arc<Device>,
    pub vk_layout: vk::DescriptorSetLayout,
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
                .ash_device
                .destroy_descriptor_set_layout(self.vk_layout, None);
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
                .ash_device
                .create_descriptor_set_layout(&layout_info, None)
        );

        DescriptorSetLayout {
            device: self.device,
            vk_layout,
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
            .buffer(buffer.vk_buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE);
        self.buffer_infos.push(buffer_info);

        let write = vk::WriteDescriptorSet::default()
            .dst_binding(binding)
            .descriptor_type(ty)
            .buffer_info(unsafe {
                std::slice::from_raw_parts(self.buffer_infos.last().unwrap() as *const _, 1)
            });

        self.writes.push(write);
        self
    }

    pub fn image(
        mut self,
        binding: u32,
        ty: vk::DescriptorType,
        layout: vk::ImageLayout,
        image: &'a Image,
    ) -> Self {
        let image_info = vk::DescriptorImageInfo::default()
            .image_layout(layout)
            .image_view(image.view)
            .sampler(image.sampler.expect("Image must have a sampler"));
        self.image_infos.push(image_info);

        let write = vk::WriteDescriptorSet::default()
            .dst_binding(binding)
            .descriptor_type(ty)
            .image_info(unsafe {
                std::slice::from_raw_parts(self.image_infos.last().unwrap() as *const _, 1)
            });

        self.writes.push(write);
        self
    }

    pub fn images(
        mut self,
        binding: u32,
        ty: vk::DescriptorType,
        layout: vk::ImageLayout,
        images: &'a [Image],
    ) -> Self {
        let info_start_index = self.image_infos.len();

        for image in images {
            let image_info = vk::DescriptorImageInfo::default()
                .image_layout(layout)
                .image_view(image.view)
                .sampler(image.sampler.expect("Image in array must have a sampler"));
            self.image_infos.push(image_info);
        }

        let write = vk::WriteDescriptorSet::default()
            .dst_binding(binding)
            .descriptor_count(images.len() as u32)
            .descriptor_type(ty)
            .image_info(unsafe {
                std::slice::from_raw_parts(
                    self.image_infos.as_ptr().add(info_start_index),
                    images.len(),
                )
            });

        self.writes.push(write);
        self
    }

    pub fn update(mut self, device: Arc<Device>, set: vk::DescriptorSet) {
        for write in &mut self.writes {
            write.dst_set = set;
        }

        unsafe {
            device.ash_device.update_descriptor_sets(&self.writes, &[]);
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
                .ash_device
                .create_descriptor_pool(&pool_create_info, None)
        );

        Self { device, vk_pool }
    }

    pub fn allocate(&self, layout: &DescriptorSetLayout, variable_count: u32) -> vk::DescriptorSet {
        let layouts = [layout.vk_layout];

        let variable_counts = [variable_count];

        let mut variable_count_alloc_info =
            vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
                .descriptor_counts(&variable_counts);

        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.vk_pool)
            .set_layouts(&layouts)
            .push_next(&mut variable_count_alloc_info);

        let sets = unsafe_vk_try!(
            self.device
                .ash_device
                .allocate_descriptor_sets(&allocate_info)
        );

        sets[0]
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe {
            self.device
                .ash_device
                .destroy_descriptor_pool(self.vk_pool, None);
        }
    }
}
