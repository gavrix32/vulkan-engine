use crate::vulkan::adapter::Adapter;
use crate::vulkan::buffer::Buffer;
use crate::vulkan::command_buffer::CommandBuffer;
use crate::vulkan::device::Device;
use crate::vulkan::instance::Instance;
use ash::util::Align;
use ash::vk;
use image::ImageReader;
use log::error;
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
    ) -> Self {
        let layout = vk::ImageLayout::UNDEFINED;

        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(layout)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1);

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

        let view = create_image_view(device.clone(), vk_image, format, aspect);

        Self {
            device,
            vk_image,
            memory,
            view,
            sampler: None,
        }
    }

    pub fn from_bytes<R: io::Seek + io::BufRead>(
        bytes: &mut R,
        instance: &Instance,
        adapter: &Adapter,
        device: Arc<Device>,
        command_pool: vk::CommandPool,
    ) -> Self {
        let image = ImageReader::new(bytes)
            .with_guessed_format()
            .expect("Failed to guess format")
            .decode()
            .expect("Failed to decode image");
        let size = (image.width() * image.height() * 4) as vk::DeviceSize;

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
        image_align.copy_from_slice(image.to_rgba8().as_raw());

        staging_buffer.unmap_memory();

        let format = vk::Format::R8G8B8A8_SRGB;
        let layout = vk::ImageLayout::UNDEFINED;

        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: image.width(),
                height: image.height(),
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(layout)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1);

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
        );
        copy_from_buffer(
            device.clone(),
            &staging_buffer,
            vk_image,
            image.width(),
            image.height(),
            command_pool,
        );
        transition_layout(
            device.clone(),
            command_pool,
            vk_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );

        let view = create_image_view(
            device.clone(),
            vk_image,
            format,
            vk::ImageAspectFlags::COLOR,
        );
        let sampler = create_sampler(device.clone());

        Self {
            device,
            vk_image,
            memory,
            view,
            sampler: Some(sampler),
        }
    }
}

fn transition_layout(
    device: Arc<Device>,
    command_pool: vk::CommandPool,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let mut barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
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
) -> vk::ImageView {
    let image_view_create_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: 1,
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

fn create_sampler(device: Arc<Device>) -> vk::Sampler {
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
        .max_lod(0.0);

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
