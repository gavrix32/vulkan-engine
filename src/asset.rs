use crate::context::RenderContext;
use crate::model::{Model, Primitive};
use crate::parser;
use crate::vertex::Vertex;
use crate::vulkan::adapter::Adapter;
use crate::vulkan::buffer::Buffer;
use crate::vulkan::descriptor::{DescriptorPool, DescriptorWriter};
use crate::vulkan::device::Device;
use crate::vulkan::image::{Image, ImageBuilder, ImageView, Sampler};
use crate::vulkan::instance::Instance;
use crate::vulkan::util;
use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::Allocator;
use image::ImageReader;
use log::info;
use std::io::Cursor;
use std::sync::{Arc, Mutex};

pub struct AssetManager {
    ctx: Arc<RenderContext>,
    allocator: Arc<Mutex<Allocator>>,
}

impl AssetManager {
    pub fn new(ctx: Arc<RenderContext>, allocator: Arc<Mutex<Allocator>>) -> Self {
        Self { ctx, allocator }
    }

    // pub fn load_texture_rgba8<S: AsRef<[u8]>>(&self, slice: S) -> (Arc<Image>, ImageView) {
    //     let image_reader = ImageReader::new(Cursor::new(slice))
    //         .with_guessed_format()
    //         .expect("Failed to guess format")
    //         .decode()
    //         .expect("Failed to decode image")
    //         .to_rgba8();
    //     let bytes = bytemuck::cast_slice(image_reader.as_raw());
    //
    //     let image = create_image(
    //         &self.ctx.instance,
    //         &self.ctx.adapter,
    //         self.ctx.device.clone(),
    //         vk::Format::R8G8B8A8_SRGB,
    //         image_reader.width(),
    //         image_reader.height(),
    //         bytes,
    //     );
    //     let image_view = ImageView::new(
    //         self.ctx.device.clone(),
    //         image.clone(),
    //         vk::ImageViewType::TYPE_2D,
    //         vk::ImageAspectFlags::COLOR,
    //     );
    //
    //     (image, image_view)
    // }

    pub fn load_texture_rgba32f<S: AsRef<[u8]>>(&self, slice: S) -> (Arc<Image>, ImageView) {
        let image_reader = ImageReader::new(Cursor::new(slice))
            .with_guessed_format()
            .expect("Failed to guess format")
            .decode()
            .expect("Failed to decode image")
            .to_rgba32f();
        let bytes = bytemuck::cast_slice(image_reader.as_raw());

        let image = create_image(
            &self.ctx.instance,
            &self.ctx.adapter,
            self.ctx.device.clone(),
            self.allocator.clone(),
            vk::Format::R32G32B32A32_SFLOAT,
            image_reader.width(),
            image_reader.height(),
            bytes,
        );
        let image_view = ImageView::new(
            self.ctx.device.clone(),
            image.clone(),
            vk::ImageViewType::TYPE_2D,
            vk::ImageAspectFlags::COLOR,
            0,
            image.mip_levels,
        );

        (image, image_view)
    }

    pub fn load_gltf<S: AsRef<[u8]>>(&self, slice: S) -> Model {
        info!("Importing model");
        let (document, buffers_data, images_data) =
            gltf::import_slice(slice).expect("Failed to load model");

        let mut vertices: Vec<Vertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        let mut primitives: Vec<Primitive> = Vec::new();

        info!("Parsing model");
        parser::parse_model(
            &document,
            &buffers_data,
            &mut vertices,
            &mut indices,
            &mut primitives,
            images_data.len(),
        );
        info!("Vertices: {}, Indices: {}", vertices.len(), indices.len());

        info!("Generating tangents");
        let mut mesh_view = parser::MeshView {
            vertices: &mut vertices,
            indices: &indices,
        };
        mikktspace::generate_tangents(&mut mesh_view);

        let mut texture_formats = vec![vk::Format::R8G8B8A8_UNORM; images_data.len()];

        for material in document.materials() {
            if let Some(info) = material.pbr_metallic_roughness().base_color_texture() {
                let index = info.texture().source().index();
                if index < texture_formats.len() {
                    texture_formats[index] = vk::Format::R8G8B8A8_SRGB;
                }
            }
        }

        let mut images = Vec::<Arc<Image>>::new();
        let mut image_views = Vec::<ImageView>::new();

        let mut size_mb = 0;

        info!("Loading textures");
        for (i, image_data) in images_data.iter().enumerate() {
            size_mb += image_data.pixels.len() / 1024 / 1024;

            let (image_bytes, image_width, image_height) = {
                let image_width = image_data.width;
                let image_height = image_data.height;

                match image_data.format {
                    gltf::image::Format::R8G8B8A8 => {
                        (&image_data.pixels, image_width, image_height)
                    }
                    gltf::image::Format::R8G8B8 => (
                        &util::rgb_to_rgba(&image_data.pixels),
                        image_width,
                        image_height,
                    ),
                    gltf::image::Format::R8G8 => (
                        &util::rg_to_rgba(&image_data.pixels),
                        image_width,
                        image_height,
                    ),
                    gltf::image::Format::R8 => (
                        &util::r_to_rgba(&image_data.pixels),
                        image_width,
                        image_height,
                    ),
                    _ => panic!("Unsupported texture format: {:?}", image_data.format),
                }
            };

            let image = Arc::new(
                ImageBuilder::default()
                    .size(image_width, image_height)
                    .generate_mipmaps(true)
                    .format(texture_formats[i])
                    .usage(
                        vk::ImageUsageFlags::TRANSFER_SRC
                            | vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::SAMPLED,
                    )
                    .bytes(image_bytes)
                    .build(
                        &self.ctx.instance,
                        &self.ctx.adapter,
                        self.ctx.device.clone(),
                        self.allocator.clone(),
                        MemoryLocation::Unknown,
                    ),
            );

            images.push(image.clone());
            image_views.push(ImageView::new(
                self.ctx.device.clone(),
                image.clone(),
                vk::ImageViewType::TYPE_2D,
                vk::ImageAspectFlags::COLOR,
                0,
                image.mip_levels,
            ));
        }

        let placeholder_raw_bytes = include_bytes!("../resources/textures/placeholder.png");
        let placeholder_image_reader = ImageReader::new(Cursor::new(placeholder_raw_bytes))
            .with_guessed_format()
            .expect("Failed to guess format")
            .decode()
            .expect("Failed to decode image")
            .to_rgba8();
        let placeholder_bytes = bytemuck::cast_slice(placeholder_image_reader.as_raw());
        let placeholder_image = create_image(
            &self.ctx.instance,
            &self.ctx.adapter,
            self.ctx.device.clone(),
            self.allocator.clone(),
            vk::Format::R8G8B8A8_UNORM,
            placeholder_image_reader.width(),
            placeholder_image_reader.height(),
            placeholder_bytes,
        );
        images.push(placeholder_image.clone());

        image_views.push(ImageView::new(
            self.ctx.device.clone(),
            placeholder_image.clone(),
            vk::ImageViewType::TYPE_2D,
            vk::ImageAspectFlags::COLOR,
            0,
            placeholder_image.mip_levels,
        ));

        info!("Textures: {}, Size: {} MB", images.len(), size_mb);

        let vertex_buffer = create_buffer(
            &self.ctx.adapter,
            self.ctx.device.clone(),
            self.allocator.clone(),
            self.ctx.device.graphics_queue,
            &vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        );

        let index_buffer = create_buffer(
            &self.ctx.adapter,
            self.ctx.device.clone(),
            self.allocator.clone(),
            self.ctx.device.graphics_queue,
            &indices,
            vk::BufferUsageFlags::INDEX_BUFFER,
        );

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(images.len() as u32)];

        let descriptor_pool = DescriptorPool::new(self.ctx.device.clone(), 1, &pool_sizes);

        let descriptor_set =
            descriptor_pool.allocate(&self.ctx.res_descriptor_layout, images.len() as u32);

        let sampler = Sampler::new(
            self.ctx.device.clone(),
            vk::Filter::LINEAR,
            vk::Filter::LINEAR,
            vk::LOD_CLAMP_NONE,
        );

        DescriptorWriter::default()
            .images(
                0,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                &image_views,
                &sampler,
            )
            .update(self.ctx.device.clone(), descriptor_set);

        Model {
            vertex_buffer,
            index_buffer,
            primitives,
            _sampler: sampler,
            _images: images,
            _image_views: image_views,
            _res_descriptor_pool: descriptor_pool,
            res_descriptor_set: descriptor_set,
        }
    }
}

fn create_buffer<T: bytemuck::NoUninit>(
    adapter: &Adapter,
    device: Arc<Device>,
    allocator: Arc<Mutex<Allocator>>,
    graphics_queue: vk::Queue,
    data: &[T],
    usage: vk::BufferUsageFlags,
) -> Buffer {
    let size = (size_of::<T>() * data.len()) as vk::DeviceSize;

    let mut staging_buffer = Buffer::new(
        device.clone(),
        allocator.clone(),
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,
    );

    staging_buffer.update(&data);

    let vertex_buffer = Buffer::new(
        device.clone(),
        allocator.clone(),
        size,
        vk::BufferUsageFlags::TRANSFER_DST | usage,
        MemoryLocation::Unknown,
    );

    staging_buffer.copy(graphics_queue, adapter, &vertex_buffer);

    vertex_buffer
}

fn create_image(
    instance: &Instance,
    adapter: &Adapter,
    device: Arc<Device>,
    allocator: Arc<Mutex<Allocator>>,
    format: vk::Format,
    width: u32,
    height: u32,
    bytes: &[u8],
) -> Arc<Image> {
    Arc::new(
        ImageBuilder::default()
            .size(width, height)
            .generate_mipmaps(true)
            .format(format)
            .usage(
                vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED,
            )
            .bytes(bytes)
            .build(
                &instance,
                &adapter,
                device.clone(),
                allocator.clone(),
                MemoryLocation::Unknown,
            ),
    )
}
