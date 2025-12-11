use crate::vulkan::adapter::Adapter;
use crate::vulkan::descriptor::DescriptorSetLayout;
use crate::vulkan::device::Device;
use crate::vulkan::instance::Instance;
use crate::vulkan::surface::Surface;
use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::sync::{Arc, Mutex};

const MAX_TEXTURES: u32 = 1024;

pub struct RenderContext {
    pub ibl_descriptor_layout: DescriptorSetLayout,
    pub res_descriptor_layout: DescriptorSetLayout,
    pub sky_descriptor_layout: DescriptorSetLayout,

    pub allocator: Arc<Mutex<Allocator>>,
    pub device: Arc<Device>,
    pub adapter: Adapter,
    pub surface: Surface,
    pub instance: Instance,
}

impl RenderContext {
    pub fn new(
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
        validation: bool,
    ) -> Self {
        let instance = Instance::new(display_handle, validation);
        let surface = Surface::new(&instance, display_handle, window_handle);
        let adapter = Adapter::new(&instance, &surface);
        let device = Arc::new(Device::new(&instance, &adapter));
        let allocator = Arc::new(Mutex::new(
            Allocator::new(&AllocatorCreateDesc {
                instance: instance.handle.clone(),
                device: device.handle.clone(),
                physical_device: adapter.handle,
                debug_settings: Default::default(),
                buffer_device_address: false,
                allocation_sizes: Default::default(),
            })
            .expect("Failed to create GPU allocator"),
        ));

        let ibl_descriptor_layout = DescriptorSetLayout::builder(device.clone())
            // Cubemap
            .binding(
                0,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                1,
                vk::ShaderStageFlags::FRAGMENT,
                vk::DescriptorBindingFlags::empty(),
            )
            // Irradiance
            .binding(
                1,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                1,
                vk::ShaderStageFlags::FRAGMENT,
                vk::DescriptorBindingFlags::empty(),
            )
            // Prefilter
            .binding(
                2,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                1,
                vk::ShaderStageFlags::FRAGMENT,
                vk::DescriptorBindingFlags::empty(),
            )
            // BRDF LUT
            .binding(
                3,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                1,
                vk::ShaderStageFlags::FRAGMENT,
                vk::DescriptorBindingFlags::empty(),
            )
            .build();

        let res_descriptor_layout = DescriptorSetLayout::builder(device.clone())
            .binding(
                0,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                MAX_TEXTURES,
                vk::ShaderStageFlags::FRAGMENT,
                vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                    | vk::DescriptorBindingFlags::PARTIALLY_BOUND
                    | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT,
            )
            .build();

        let sky_descriptor_layout = DescriptorSetLayout::builder(device.clone())
            .binding(
                0,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                1,
                vk::ShaderStageFlags::FRAGMENT,
                vk::DescriptorBindingFlags::empty(),
            )
            .build();

        RenderContext {
            device,
            adapter,
            surface,
            instance,
            allocator,
            ibl_descriptor_layout,
            res_descriptor_layout,
            sky_descriptor_layout,
        }
    }
}
