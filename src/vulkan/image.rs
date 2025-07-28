use crate::vulkan::adapter::Adapter;
use crate::vulkan::buffer::Buffer;
use crate::vulkan::command_buffer::CommandBuffer;
use crate::vulkan::device::Device;
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

        let vk_image = unsafe { device.ash_device.create_image(&image_create_info, None) }
            .expect("Failed to create image");

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

        let memory = unsafe {
            device
                .ash_device
                .allocate_memory(&memory_allocate_info, None)
        }
        .expect("Failed to allocate image memory");

        unsafe { device.ash_device.bind_image_memory(vk_image, memory, 0) }
            .expect("Failed to bind image memory");

        let view = create_image_view(device.clone(), vk_image, format, aspect, mip_levels);

        Self {
            device,
            vk_image,
            memory,
            view,
            sampler: None,
        }
    }

    #[allow(unused)]
    pub fn read<R: io::Seek + io::BufRead>(
        buffer: &mut R,
        instance: &Instance,
        adapter: &Adapter,
        device: Arc<Device>,
        command_pool: vk::CommandPool,
        mipmapping: bool,
        msaa_samples: vk::SampleCountFlags,
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
            instance,
            adapter,
            device,
            command_pool,
            mipmapping,
            msaa_samples,
        )
    }

    pub fn from_bytes(
        bytes: &[u8],
        width: u32,
        height: u32,
        instance: &Instance,
        adapter: &Adapter,
        device: Arc<Device>,
        command_pool: vk::CommandPool,
        mip_mapping: bool,
        msaa_samples: vk::SampleCountFlags,
    ) -> Self {
        let size = (width * height * 4) as vk::DeviceSize;
        let mip_levels = if mip_mapping {
            max(width, height).ilog2() + 1
        } else {
            1
        };

        let mut staging_buffer = Buffer::new(
            instance,
            adapter,
            device.clone(),
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        staging_buffer.map_memory();
        let data_ptr = staging_buffer.p_data.unwrap();

        let mut image_align =
            unsafe { Align::new(data_ptr, align_of::<u8>() as vk::DeviceSize, size) };
        image_align.copy_from_slice(bytes);

        staging_buffer.unmap_memory();

        let format = vk::Format::R8G8B8A8_SRGB;
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

        let vk_image = unsafe { device.ash_device.create_image(&image_create_info, None) }
            .expect("Failed to create image");

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

        let memory = unsafe {
            device
                .ash_device
                .allocate_memory(&memory_allocate_info, None)
        }
        .expect("Failed to allocate image memory");

        unsafe { device.ash_device.bind_image_memory(vk_image, memory, 0) }
            .expect("Failed to bind image memory");

        transition_layout(
            device.clone(),
            command_pool,
            vk_image,
            layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            mip_levels,
        );
        copy_from_buffer(
            device.clone(),
            &staging_buffer,
            vk_image,
            width,
            height,
            command_pool,
        );
        if mip_mapping {
            generate_mipmaps(
                instance,
                device.clone(),
                adapter,
                command_pool,
                vk_image,
                format,
                width,
                height,
                mip_levels,
            );
        } else {
            transition_layout(
                device.clone(),
                command_pool,
                vk_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                mip_levels,
            );
        }

        let view = create_image_view(
            device.clone(),
            vk_image,
            format,
            vk::ImageAspectFlags::COLOR,
            mip_levels,
        );
        let sampler = create_sampler(device.clone(), mip_levels);

        Self {
            device,
            vk_image,
            memory,
            view,
            sampler: Some(sampler),
        }
    }
}

fn generate_mipmaps(
    instance: &Instance,
    device: Arc<Device>,
    adapter: &Adapter,
    command_pool: vk::CommandPool,
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

    let mut barrier = vk::ImageMemoryBarrier::default()
        .image(vk_image)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(subresource_range);

    let command_buffers = CommandBuffer::begin_single_time_commands(device.clone(), command_pool);

    let mut mip_width = width as i32;
    let mut mip_height = height as i32;

    for i in 1..mip_levels {
        subresource_range = subresource_range.base_mip_level(i - 1);

        barrier = barrier
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .subresource_range(subresource_range);

        unsafe {
            device.ash_device.cmd_pipeline_barrier(
                command_buffers[0],
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            )
        };

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

        unsafe {
            device.ash_device.cmd_blit_image(
                command_buffers[0],
                vk_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                vk::Filter::LINEAR,
            )
        };

        barrier = barrier
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_access_mask(vk::AccessFlags::TRANSFER_READ)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);

        unsafe {
            device.ash_device.cmd_pipeline_barrier(
                command_buffers[0],
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            )
        };

        if mip_width > 1 {
            mip_width /= 2
        }
        if mip_height > 1 {
            mip_height /= 2
        }
    }

    subresource_range = subresource_range.base_mip_level(mip_levels - 1);

    barrier = barrier
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .subresource_range(subresource_range);

    unsafe {
        device.ash_device.cmd_pipeline_barrier(
            command_buffers[0],
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        )
    };

    CommandBuffer::end_single_time_commands(
        device.clone(),
        command_pool,
        command_buffers,
        device.graphics_queue,
    );
}

fn transition_layout(
    device: Arc<Device>,
    command_pool: vk::CommandPool,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    mip_levels: u32,
) {
    let mut barrier = vk::ImageMemoryBarrier::default()
        .image(image)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: mip_levels,
            base_array_layer: 0,
            layer_count: 1,
        });

    let mut src_stage = vk::PipelineStageFlags::default();
    let mut dst_stage = vk::PipelineStageFlags::default();

    if old_layout == vk::ImageLayout::UNDEFINED
        && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
    {
        barrier = barrier
            .src_access_mask(vk::AccessFlags::NONE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);

        src_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
        dst_stage = vk::PipelineStageFlags::TRANSFER;
    } else if old_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
        && new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
    {
        barrier = barrier
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);

        src_stage = vk::PipelineStageFlags::TRANSFER;
        dst_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
    } else {
        error!("Unsupported layout transition");
    }

    let command_buffers = CommandBuffer::begin_single_time_commands(device.clone(), command_pool);

    unsafe {
        device.ash_device.cmd_pipeline_barrier(
            command_buffers[0],
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        )
    };

    CommandBuffer::end_single_time_commands(
        device.clone(),
        command_pool,
        command_buffers,
        device.graphics_queue,
    );
}

fn copy_from_buffer(
    device: Arc<Device>,
    buffer: &Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
    command_pool: vk::CommandPool,
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

    let command_buffers = CommandBuffer::begin_single_time_commands(device.clone(), command_pool);

    unsafe {
        device.ash_device.cmd_copy_buffer_to_image(
            command_buffers[0],
            buffer.vk_buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        )
    };

    CommandBuffer::end_single_time_commands(
        device.clone(),
        command_pool,
        command_buffers,
        device.graphics_queue,
    );
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

    unsafe {
        device
            .ash_device
            .create_image_view(&image_view_create_info, None)
    }
    .expect("Failed to create image view")
}

fn create_sampler(device: Arc<Device>, mip_levels: u32) -> vk::Sampler {
    let sampler_create_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
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

    unsafe { device.ash_device.create_sampler(&sampler_create_info, None) }
        .expect("Failed to create sampler")
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
