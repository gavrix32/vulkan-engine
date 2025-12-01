use crate::vulkan::buffer::Buffer;
use crate::vulkan::image::{Image, ImageView};
use glam::Mat4;
use std::sync::Arc;

pub struct Mesh {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub primitives: Vec<Primitive>,
    pub images: Vec<Arc<Image>>,
    pub image_views: Vec<ImageView>,
}

pub(crate) struct Primitive {
    pub(crate) first_index: u32,
    pub(crate) index_count: u32,
    pub(crate) model_matrix: Mat4,
}
