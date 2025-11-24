use crate::unsafe_vk_try;
use crate::vulkan::adapter::Adapter;
use crate::vulkan::buffer::Buffer;
use crate::vulkan::device::Device;
use crate::vulkan::encoder::Encoder;
use crate::vulkan::instance::Instance;
use ash::util::Align;
use ash::vk;
use log::error;
use std::cmp::max;
use std::sync::Arc;

// TODO: use allocator crate

pub struct Image {
    device: Arc<Device>,
    pub handle: vk::Image,
    memory: vk::DeviceMemory,

    // Metadata
    format: vk::Format,
    extent: vk::Extent3D,
    layers: u32,
    mip_levels: u32,
    layout: vk::ImageLayout,
}

impl Image {
    fn new(
        builder: ImageBuilder,
        instance: &Instance,
        adapter: &Adapter,
        device: Arc<Device>,
    ) -> Self {
        let extent = vk::Extent3D {
            width: builder.width,
            height: builder.height,
            depth: 1,
        };

        let mut mip_levels = 1;
        if builder.mipmapping {
            mip_levels = max(builder.width, builder.height).ilog2() + 1;
        }

        let layout = vk::ImageLayout::UNDEFINED;

        let mut usage = builder.usage;
        if builder.mipmapping {
            usage = vk::ImageUsageFlags::TRANSFER_SRC | usage
        }

        let image_create_info = vk::ImageCreateInfo::default()
            .flags(builder.flags)
            .image_type(builder.image_type)
            .extent(extent)
            .mip_levels(mip_levels)
            .array_layers(builder.layers)
            .format(builder.format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(layout)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(builder.samples);

        let vk_image = unsafe_vk_try!(device.ash_device.create_image(&image_create_info, None));

        let memory_requirements =
            unsafe { device.ash_device.get_image_memory_requirements(vk_image) };

        let memory_allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_requirements.size)
            .memory_type_index(Buffer::find_memory_type(
                instance,
                adapter,
                memory_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ));

        let memory = unsafe_vk_try!(
            device
                .ash_device
                .allocate_memory(&memory_allocate_info, None)
        );

        unsafe_vk_try!(device.ash_device.bind_image_memory(vk_image, memory, 0));

        Self {
            device,
            handle: vk_image,
            memory,

            format: builder.format,
            extent,
            layers: builder.layers,
            mip_levels,
            layout,
        }
    }

    fn load_bytes(self, bytes: &[u8], instance: &Instance, adapter: &Adapter) -> Self {
        let pixel_size = match self.format {
            vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => 4,
            vk::Format::R32G32B32A32_SFLOAT => 16,
            vk::Format::R32G32B32_SFLOAT => 12,
            _ => panic!("Unsupported format: {:?}", self.format),
        };
        let image_size = (self.extent.width * self.extent.height * pixel_size) as vk::DeviceSize;

        let mut staging_buffer = Buffer::new(
            instance,
            adapter,
            self.device.clone(),
            image_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        staging_buffer.map_memory();
        let data_ptr = staging_buffer.p_data.unwrap();

        let mut image_align =
            unsafe { Align::new(data_ptr, align_of::<u8>() as vk::DeviceSize, image_size) };
        image_align.copy_from_slice(bytes);

        staging_buffer.unmap_memory();

        let encoder = Encoder::begin_single_time(self.device.clone(), adapter);
        Self::transition_layout(
            &encoder,
            self.handle,
            self.layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            self.mip_levels,
            self.layers,
        );
        encoder.end_single_time(self.device.graphics_queue);

        copy_from_buffer(
            self.device.clone(),
            adapter,
            &staging_buffer,
            self.handle,
            self.extent,
        );
        if self.mip_levels != 1 {
            generate_mipmaps(
                instance,
                self.device.clone(),
                adapter,
                self.handle,
                self.format,
                self.extent.width,
                self.extent.height,
                self.mip_levels,
            );
        } else {
            let encoder = Encoder::begin_single_time(self.device.clone(), adapter);
            Self::transition_layout(
                &encoder,
                self.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                self.mip_levels,
                self.layers,
            );
            encoder.end_single_time(self.device.graphics_queue);
        }

        self
    }

    // TODO: обязаительно менять поле image.layout!!!
    pub fn transition_layout(
        encoder: &Encoder,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        mip_levels: u32,
        layer_count: u32,
    ) {
        // TODO: убрать эту залупу, чтобы нужно было вводить все stages и accesses вручную
        let (src_stage, dst_stage, src_access, dst_access) = match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::empty(),
                vk::AccessFlags2::TRANSFER_WRITE,
            ),
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::GENERAL) => (
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::empty(),
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
            ),
            (vk::ImageLayout::GENERAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_WRITE,
                vk::AccessFlags2::SHADER_READ,
            ),
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::PipelineStageFlags2::TRANSFER,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::AccessFlags2::SHADER_READ,
            ),
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => (
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::empty(),
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            ),
            (vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR) => (
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags2::empty(),
            ),
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL) => (
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
                vk::AccessFlags2::empty(),
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ),
            _ => panic!("Unsupported layout transition"),
        };

        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count,
            });

        let barriers = [barrier];
        let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);

        encoder.cmd_pipeline_barrier2(&dependency_info);
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            self.device.ash_device.destroy_image(self.handle, None);
            self.device.ash_device.free_memory(self.memory, None);
        }
    }
}

#[derive(Clone, Copy)]
pub struct ImageBuilder<'a> {
    image_type: vk::ImageType,
    flags: vk::ImageCreateFlags,
    width: u32,
    height: u32,
    layers: u32,
    mipmapping: bool,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    samples: vk::SampleCountFlags,
    bytes: &'a [u8],
}

impl Default for ImageBuilder<'_> {
    fn default() -> Self {
        Self {
            image_type: vk::ImageType::TYPE_2D,
            flags: vk::ImageCreateFlags::empty(),
            width: 0,
            height: 0,
            layers: 1,
            mipmapping: false,
            format: vk::Format::default(),
            usage: vk::ImageUsageFlags::default(),
            samples: vk::SampleCountFlags::TYPE_1,
            bytes: &[],
        }
    }
}

impl<'a> ImageBuilder<'a> {
    pub fn image_type(mut self, image_type: vk::ImageType) -> Self {
        self.image_type = image_type;
        self
    }

    pub fn flags(mut self, flags: vk::ImageCreateFlags) -> Self {
        self.flags = flags;
        self
    }

    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    pub fn layers(mut self, layers: u32) -> Self {
        self.layers = layers;
        self
    }

    pub fn mipmapping(mut self, mipmapping: bool) -> Self {
        self.mipmapping = mipmapping;
        self
    }

    pub fn format(mut self, format: vk::Format) -> Self {
        self.format = format;
        self
    }

    pub fn usage(mut self, usage: vk::ImageUsageFlags) -> Self {
        self.usage = usage;
        self
    }

    pub fn samples(mut self, samples: vk::SampleCountFlags) -> Self {
        self.samples = samples;
        self
    }

    pub fn bytes(mut self, bytes: &'a [u8]) -> Self {
        self.bytes = bytes;
        self
    }

    pub fn build(self, instance: &Instance, adapter: &Adapter, device: Arc<Device>) -> Image {
        let image = Image::new(self, instance, adapter, device);
        if !self.bytes.is_empty() {
            return image.load_bytes(self.bytes, instance, adapter);
        }
        image
    }
}

pub struct ImageView {
    device: Arc<Device>,
    pub handle: vk::ImageView,
    _image: Arc<Image>,
}

impl ImageView {
    pub fn new(
        device: Arc<Device>,
        image: Arc<Image>,
        ty: vk::ImageViewType,
        aspect: vk::ImageAspectFlags,
    ) -> Self {
        let image_view_create_info = vk::ImageViewCreateInfo::default()
            .image(image.handle)
            .view_type(ty)
            .format(image.format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: image.mip_levels,
                base_array_layer: 0,
                layer_count: image.layers,
            });

        let handle = unsafe_vk_try!(
            device
                .ash_device
                .create_image_view(&image_view_create_info, None)
        );

        Self {
            device,
            handle,
            _image: image,
        }
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        unsafe {
            self.device.ash_device.destroy_image_view(self.handle, None);
        }
    }
}

pub struct Sampler {
    device: Arc<Device>,
    pub handle: vk::Sampler,
}

impl Sampler {
    pub fn new(
        device: Arc<Device>,
        mag_filter: vk::Filter,
        min_filter: vk::Filter,
        max_lod: f32,
    ) -> Self {
        let sampler_create_info = vk::SamplerCreateInfo::default()
            .mag_filter(mag_filter)
            .min_filter(min_filter)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(false)
            .max_anisotropy(1.0)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(max_lod);

        let handle = unsafe_vk_try!(device.ash_device.create_sampler(&sampler_create_info, None));

        Self { device, handle }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe {
            self.device.ash_device.destroy_sampler(self.handle, None);
        }
    }
}

fn generate_mipmaps(
    instance: &Instance,
    device: Arc<Device>,
    adapter: &Adapter,
    vk_image: vk::Image,
    format: vk::Format,
    width: u32,
    height: u32,
    mip_levels: u32,
) {
    let format_props = unsafe {
        instance
            .ash_instance
            .get_physical_device_format_properties(adapter.physical_device, format)
    };
    if format_props.optimal_tiling_features & vk::FormatFeatureFlags::SAMPLED_IMAGE
        == vk::FormatFeatureFlags::empty()
    {
        error!("Texture image format does not support linear blitting");
    }

    let mut subresource_range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .level_count(1);

    let mut barrier = vk::ImageMemoryBarrier2::default().image(vk_image);

    let encoder = Encoder::begin_single_time(device.clone(), adapter);

    let mut mip_width = width as i32;
    let mut mip_height = height as i32;

    for i in 1..mip_levels {
        subresource_range = subresource_range.base_mip_level(i - 1);

        barrier = barrier
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .subresource_range(subresource_range);
        let barriers = [barrier];
        let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);

        encoder.cmd_pipeline_barrier2(&dependency_info);

        let blit = vk::ImageBlit::default()
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: mip_width,
                    y: mip_height,
                    z: 1,
                },
            ])
            .src_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(i - 1)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: if mip_width > 1 { mip_width / 2 } else { 1 },
                    y: if mip_height > 1 { mip_height / 2 } else { 1 },
                    z: 1,
                },
            ])
            .dst_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(i)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        encoder.cmd_blit_image(
            vk_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[blit],
            vk::Filter::LINEAR,
        );

        barrier = barrier
            .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let barriers = [barrier];
        let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);

        encoder.cmd_pipeline_barrier2(&dependency_info);

        if mip_width > 1 {
            mip_width /= 2
        }
        if mip_height > 1 {
            mip_height /= 2
        }
    }

    subresource_range = subresource_range.base_mip_level(mip_levels - 1);

    barrier = barrier
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .dst_access_mask(vk::AccessFlags2::SHADER_READ)
        .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .subresource_range(subresource_range);
    let barriers = [barrier];
    let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);

    encoder.cmd_pipeline_barrier2(&dependency_info);
    encoder.end_single_time(device.graphics_queue);
}

fn copy_from_buffer(
    device: Arc<Device>,
    adapter: &Adapter,
    buffer: &Buffer,
    image: vk::Image,
    extent: vk::Extent3D,
) {
    let region = vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(extent);

    let encoder = Encoder::begin_single_time(device.clone(), adapter);

    encoder.cmd_copy_buffer_to_image(
        buffer.vk_buffer,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &[region],
    );

    encoder.end_single_time(device.graphics_queue);
}
