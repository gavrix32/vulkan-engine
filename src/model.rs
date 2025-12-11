use crate::vulkan::buffer::Buffer;
use crate::vulkan::descriptor::DescriptorPool;
use crate::vulkan::image::{Image, ImageView, Sampler};
use ash::vk;
use glam::Mat4;
use std::sync::Arc;

pub struct Model {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub primitives: Vec<Primitive>,
    pub _sampler: Sampler,
    pub _images: Vec<Arc<Image>>,
    pub _image_views: Vec<ImageView>,
    pub _res_descriptor_pool: DescriptorPool,
    pub res_descriptor_set: vk::DescriptorSet,
}

pub(crate) struct Primitive {
    pub(crate) first_index: u32,
    pub(crate) index_count: u32,
    pub(crate) model_matrix: Mat4,
}
