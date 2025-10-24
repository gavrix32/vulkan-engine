use crate::unsafe_vk_try;
use crate::vulkan::adapter::Adapter;
use crate::vulkan::buffer::Buffer;
use crate::vulkan::device::Device;
use crate::vulkan::encoder::Encoder;
use crate::vulkan::instance::Instance;
use ash::util::Align;
use ash::vk;
use image::ImageReader;
use log::error;
use std::cmp::max;
use std::io;
use std::sync::Arc;

pub struct Image {
    device: Arc<Device>,
    vk_image: vk::Image,
    memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub sampler: Option<vk::Sampler>,
}

impl Image {
    pub fn new(
        instance: &Instance,
        adapter: &Adapter,
        device: Arc<Device>,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect: vk::ImageAspectFlags,
        mipmapping: bool,
        msaa_samples: vk::SampleCountFlags,
    ) -> Self {
        let layout = vk::ImageLayout::UNDEFINED;
        let image_type = vk::ImageType::TYPE_2D;
        let mip_levels = if mipmapping {
            max(width, height).ilog2() + 1
        } else {
            1
        };

        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(image_type)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(layout)
            .usage(if mipmapping {
                vk::ImageUsageFlags::TRANSFER_SRC | usage
            } else {
                usage
            })
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(msaa_samples);

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

        let view = create_image_view(device.clone(), vk_image, format, aspect, mip_levels);

        Self {
            device,
            vk_image,
            memory,
            view,
            sampler: None,
        }
    }

    pub fn read_rgba8<R: io::Seek + io::BufRead>(
        buffer: &mut R,
        instance: &Instance,
        adapter: &Adapter,
        device: Arc<Device>,
        mipmapping: bool,
        msaa_samples: vk::SampleCountFlags,
        mag_filter: vk::Filter,
        min_filter: vk::Filter,
    ) -> Self {
        let image = ImageReader::new(buffer)
            .with_guessed_format()
            .expect("Failed to guess format")
            .decode()
            .expect("Failed to decode image")
            .to_rgba8();
        let bytes = image.as_raw();

        Self::from_bytes(
            bytes,
            image.width(),
            image.height(),
            vk::Format::R8G8B8A8_SRGB,
            instance,
            adapter,
            device,
            mipmapping,
            msaa_samples,
            mag_filter,
            min_filter,
        )
    }

    pub fn read_rgba32<R: io::Seek + io::BufRead>(
        buffer: &mut R,
        instance: &Instance,
        adapter: &Adapter,
        device: Arc<Device>,
        mipmapping: bool,
        msaa_samples: vk::SampleCountFlags,
        mag_filter: vk::Filter,
        min_filter: vk::Filter,
    ) -> Self {
        let image = ImageReader::new(buffer)
            .with_guessed_format()
            .expect("Failed to guess format")
            .decode()
            .expect("Failed to decode image")
            .to_rgba32f();
        let bytes = bytemuck::cast_slice(image.as_raw());

        Self::from_bytes(
            bytes,
            image.width(),
            image.height(),
            vk::Format::R32G32B32A32_SFLOAT,
            instance,
            adapter,
            device,
            mipmapping,
            msaa_samples,
            mag_filter,
            min_filter,
        )
    }

    pub fn from_bytes(
        bytes: &[u8],
        width: u32,
        height: u32,
        format: vk::Format,
        instance: &Instance,
        adapter: &Adapter,
        device: Arc<Device>,
        mip_mapping: bool,
        msaa_samples: vk::SampleCountFlags,
        mag_filter: vk::Filter,
        min_filter: vk::Filter,
    ) -> Self {
        let pixel_size = match format {
            vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => 4,
            vk::Format::R32G32B32A32_SFLOAT => 16,
            vk::Format::R32G32B32_SFLOAT => 12,
            _ => panic!("Unsupported format: {:?}", format),
        };
        let image_size = (width * height * pixel_size) as vk::DeviceSize;

        let mip_levels = if mip_mapping {
            max(width, height).ilog2() + 1
        } else {
            1
        };

        let mut staging_buffer = Buffer::new(
            instance,
            adapter,
            device.clone(),
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

        let layout = vk::ImageLayout::UNDEFINED;

        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(1)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(layout)
            .usage(if mip_mapping {
                vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED
            } else {
                vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED
            })
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(msaa_samples);

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

        let encoder = Encoder::begin_single_time(device.clone(), adapter);
        Self::transition_layout(
            &encoder,
            vk_image,
            layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            mip_levels,
        );
        encoder.end_single_time(device.graphics_queue);

        copy_from_buffer(
            device.clone(),
            adapter,
            &staging_buffer,
            vk_image,
            width,
            height,
        );
        if mip_mapping {
            generate_mipmaps(
                instance,
                device.clone(),
                adapter,
                vk_image,
                format,
                width,
                height,
                mip_levels,
            );
        } else {
            let encoder = Encoder::begin_single_time(device.clone(), adapter);
            Self::transition_layout(
                &encoder,
                vk_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                mip_levels,
            );
            encoder.end_single_time(device.graphics_queue);
        }

        let view = create_image_view(
            device.clone(),
            vk_image,
            format,
            vk::ImageAspectFlags::COLOR,
            mip_levels,
        );
        let sampler = create_sampler(device.clone(), mip_levels, mag_filter, min_filter);

        Self {
            device,
            vk_image,
            memory,
            view,
            sampler: Some(sampler),
        }
    }

    pub fn transition_layout(
        encoder: &Encoder,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        mip_levels: u32,
    ) {
        let (src_stage, dst_stage, src_access, dst_access) = match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::PipelineStageFlags2::TRANSFER,
                vk::AccessFlags2::empty(),
                vk::AccessFlags2::TRANSFER_WRITE,
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
                layer_count: 1,
            });

        let barriers = [barrier];
        let dependency_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);

        encoder.cmd_pipeline_barrier2(&dependency_info);
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
    width: u32,
    height: u32,
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
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        });

    let encoder = Encoder::begin_single_time(device.clone(), adapter);

    encoder.cmd_copy_buffer_to_image(
        buffer.vk_buffer,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &[region],
    );

    encoder.end_single_time(device.graphics_queue);
}

fn create_image_view(
    device: Arc<Device>,
    image: vk::Image,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
    mip_levels: u32,
) -> vk::ImageView {
    let image_view_create_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: mip_levels,
            base_array_layer: 0,
            layer_count: 1,
        });

    unsafe_vk_try!(
        device
            .ash_device
            .create_image_view(&image_view_create_info, None)
    )
}

fn create_sampler(
    device: Arc<Device>,
    mip_levels: u32,
    mag_filter: vk::Filter,
    min_filter: vk::Filter,
) -> vk::Sampler {
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
        .max_lod(mip_levels as f32);

    unsafe_vk_try!(device.ash_device.create_sampler(&sampler_create_info, None))
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            if let Some(sampler) = self.sampler {
                self.device.ash_device.destroy_sampler(sampler, None);
            }
            self.device.ash_device.destroy_image_view(self.view, None);
            self.device.ash_device.destroy_image(self.vk_image, None);
            self.device.ash_device.free_memory(self.memory, None);
        }
    }
}
