use crate::context::RenderContext;
use crate::mesh::{Mesh, Primitive};
use crate::parser;
use crate::vertex::Vertex;
use crate::vulkan::adapter::Adapter;
use crate::vulkan::buffer::Buffer;
use crate::vulkan::device::Device;
use crate::vulkan::image::{Image, ImageBuilder, ImageView};
use crate::vulkan::instance::Instance;
use crate::vulkan::util;
use ash::util::Align;
use ash::vk;
use image::ImageReader;
use log::info;
use std::io::Cursor;
use std::sync::Arc;

pub struct AssetManager {
    ctx: Arc<RenderContext>,
}

impl AssetManager {
    pub fn new(ctx: Arc<RenderContext>) -> Self {
        Self { ctx }
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

    pub fn load_gltf<S: AsRef<[u8]>>(&self, slice: S) -> Mesh {
        info!("Importing model");
        let (document, buffers_data, images_data) =
            gltf::import_slice(slice).expect("Failed to load model");

        let mut vertices: Vec<Vertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();

        let mut primitives: Vec<Primitive> = Vec::new();

        info!("Parsing model");
        parser::parse_model(
            document,
            &buffers_data,
            &mut vertices,
            &mut indices,
            &mut primitives,
        );
        info!("Vertices: {}, Indices: {}", vertices.len(), indices.len());

        info!("Generating tangents");
        let mut mesh_view = parser::MeshView {
            vertices: &mut vertices,
            indices: &indices,
        };
        mikktspace::generate_tangents(&mut mesh_view);

        let mut images = Vec::<Arc<Image>>::new();
        let mut image_views = Vec::<ImageView>::new();

        let mut size_mb = 0;

        info!("Loading textures");
        for image_data in &images_data {
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
                    .format(vk::Format::R8G8B8A8_SRGB)
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
            vk::Format::R8G8B8A8_SRGB,
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
            &self.ctx.instance,
            &self.ctx.adapter,
            self.ctx.device.clone(),
            self.ctx.device.graphics_queue,
            &vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
        );

        let index_buffer = create_buffer(
            &self.ctx.instance,
            &self.ctx.adapter,
            self.ctx.device.clone(),
            self.ctx.device.graphics_queue,
            &indices,
            vk::BufferUsageFlags::INDEX_BUFFER,
        );

        Mesh {
            vertex_buffer,
            index_buffer,
            primitives,
            images,
            image_views,
        }
    }
}

fn create_buffer<T: Copy>(
    instance: &Instance,
    adapter: &Adapter,
    device: Arc<Device>,
    graphics_queue: vk::Queue,
    data: &[T],
    usage: vk::BufferUsageFlags,
) -> Buffer {
    let size = (size_of::<T>() * data.len()) as vk::DeviceSize;

    let mut staging_buffer = Buffer::new(
        instance,
        adapter,
        device.clone(),
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    );

    staging_buffer.map_memory();
    let data_ptr = staging_buffer
        .p_data
        .expect("No data pointer in staging buffer");

    let mut vertex_align = unsafe { Align::new(data_ptr, align_of::<T>() as vk::DeviceSize, size) };
    vertex_align.copy_from_slice(&data);

    staging_buffer.unmap_memory();

    let vertex_buffer = Buffer::new(
        instance,
        adapter,
        device.clone(),
        size,
        vk::BufferUsageFlags::TRANSFER_DST | usage,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    );

    staging_buffer.copy(graphics_queue, adapter, &vertex_buffer);

    vertex_buffer
}

fn create_image(
    instance: &Instance,
    adapter: &Adapter,
    device: Arc<Device>,
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
            .build(&instance, &adapter, device.clone()),
    )
}
