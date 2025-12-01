use crate::vulkan::adapter::Adapter;
use crate::vulkan::device::Device;
use crate::vulkan::image::{Image, ImageBuilder, ImageView};
use crate::vulkan::instance::Instance;
use crate::vulkan::surface::Surface;
use crate::vulkan::sync::Semaphore;
use crate::{unsafe_vk_try, vk_try};
use ash::{khr, vk};
use log::warn;
use std::sync::Arc;

pub struct Swapchain {
    device: Arc<Device>,
    pub swapchain_device: khr::swapchain::Device,
    pub swapchain_khr: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub extent: vk::Extent2D,
    pub image_views: Vec<vk::ImageView>,
    pub color_image: Arc<Image>,
    pub depth_image: Arc<Image>,
    pub color_image_view: ImageView,
    pub depth_image_view: ImageView,
    pub msaa_samples: vk::SampleCountFlags,
}

impl Swapchain {
    pub fn new(
        instance: &Instance,
        adapter: &Adapter,
        device: Arc<Device>,
        surface: &Surface,
        width: u32,
        height: u32,
        msaa_samples: vk::SampleCountFlags,
    ) -> Self {
        let (
            swapchain_device,
            swapchain_khr,
            images,
            extent,
            image_views,
            color_image,
            depth_image,
            color_image_view,
            depth_image_view,
        ) = init(
            instance,
            adapter,
            device.clone(),
            surface,
            width,
            height,
            msaa_samples,
        );

        Self {
            device,
            swapchain_device,
            swapchain_khr,
            images,
            extent,
            image_views,
            color_image,
            depth_image,
            color_image_view,
            depth_image_view,
            msaa_samples,
        }
    }

    pub fn get_format(adapter: &Adapter, surface: &Surface) -> vk::Format {
        let support_details = SupportDetails::query_support(adapter.physical_device, surface);
        choose_surface_format(support_details.formats).format
    }

    pub(crate) fn destroy(&self) {
        for image_view in &self.image_views {
            unsafe { self.device.ash_device.destroy_image_view(*image_view, None) };
        }
        unsafe {
            self.swapchain_device
                .destroy_swapchain(self.swapchain_khr, None)
        };
    }

    pub fn recreate(
        &mut self,
        instance: &Instance,
        adapter: &Adapter,
        surface: &Surface,
        width: u32,
        height: u32,
        msaa_samples: vk::SampleCountFlags,
    ) {
        self.device.wait_idle();

        self.destroy();

        let (
            swapchain_device,
            swapchain_khr,
            images,
            extent,
            image_views,
            color_image,
            depth_image,
            color_image_view,
            depth_image_view,
        ) = init(
            instance,
            adapter,
            self.device.clone(),
            surface,
            width,
            height,
            msaa_samples,
        );

        self.swapchain_device = swapchain_device;
        self.swapchain_khr = swapchain_khr;
        self.images = images;
        self.extent = extent;
        self.image_views = image_views;
        self.color_image = color_image;
        self.depth_image = depth_image;
        self.color_image_view = color_image_view;
        self.depth_image_view = depth_image_view;
        self.msaa_samples = msaa_samples;
    }

    pub fn acquire_next_image(&mut self, signal_semaphore: &Semaphore) -> Option<u32> {
        let acquire_result = unsafe {
            self.swapchain_device.acquire_next_image(
                self.swapchain_khr,
                u64::MAX,
                signal_semaphore.vk_semaphore,
                vk::Fence::null(),
            )
        };

        let image_index: u32;

        match acquire_result {
            Ok((index, is_suboptimal)) => {
                if is_suboptimal {
                    warn!("Swapchain is suboptimal");
                    return None;
                }
                image_index = index;
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                warn!("Swapchain is out of date");
                return None;
            }
            Err(e) => {
                panic!("Failed to acquire next image: {:?}", e);
            }
        }

        Some(image_index)
    }

    pub fn present(&self, image_index: u32, wait_semaphore: &Semaphore) -> bool {
        let swapchains = [self.swapchain_khr];
        let image_indices = [image_index];
        let wait_semaphores = [wait_semaphore.vk_semaphore];

        let present_info_khr = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

        let present_result = unsafe {
            self.swapchain_device
                .queue_present(self.device.present_queue, &present_info_khr)
        };

        match present_result {
            Ok(is_suboptimal) => {
                if is_suboptimal {
                    warn!("Swapchain is suboptimal, recreating...");
                } else {
                    return true;
                }
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                warn!("Swapchain is out of date, recreating...");
            }
            Err(e) => {
                panic!("Failed to present swapchain image: {:?}", e);
            }
        }
        false
    }
}

fn init(
    instance: &Instance,
    adapter: &Adapter,
    device: Arc<Device>,
    surface: &Surface,
    width: u32,
    height: u32,
    msaa_samples: vk::SampleCountFlags,
) -> (
    khr::swapchain::Device,
    vk::SwapchainKHR,
    Vec<vk::Image>,
    vk::Extent2D,
    Vec<vk::ImageView>,
    Arc<Image>,
    Arc<Image>,
    ImageView,
    ImageView,
) {
    let support_details = SupportDetails::query_support(adapter.physical_device, surface);

    let surface_format = choose_surface_format(support_details.formats);
    let present_mode = choose_present_mode(support_details.present_modes);
    let extent = choose_extent(support_details.capabilities, width, height);

    let mut image_count = support_details.capabilities.min_image_count + 1;
    if support_details.capabilities.max_image_count > 0
        && image_count > support_details.capabilities.max_image_count
    {
        image_count = support_details.capabilities.max_image_count;
    }

    let mut create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(surface.surface_khr)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .pre_transform(support_details.capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    let indices = [
        adapter
            .queue_family_indices
            .graphics_family
            .expect("Graphics queue family not found"),
        adapter
            .queue_family_indices
            .present_family
            .expect("Present queue family not found"),
    ];

    if adapter.queue_family_indices.graphics_family != adapter.queue_family_indices.present_family {
        create_info = create_info.image_sharing_mode(vk::SharingMode::CONCURRENT);
        create_info = create_info.queue_family_indices(&indices);
    } else {
        create_info = create_info.image_sharing_mode(vk::SharingMode::EXCLUSIVE);
    }

    let swapchain_device = khr::swapchain::Device::new(&instance.ash_instance, &device.ash_device);
    let swapchain_khr = unsafe_vk_try!(swapchain_device.create_swapchain(&create_info, None));
    let images = unsafe_vk_try!(swapchain_device.get_swapchain_images(swapchain_khr));
    let image_views = create_image_views(&images, surface_format.format, device.clone());

    let color_image = Arc::new(
        ImageBuilder::default()
            .size(width, height)
            .format(surface_format.format)
            .usage(
                vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            )
            .samples(msaa_samples)
            .build(&instance, &adapter, device.clone()),
    );

    let depth_image = Arc::new(
        ImageBuilder::default()
            .size(width, height)
            .format(vk::Format::D32_SFLOAT_S8_UINT)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .samples(msaa_samples)
            .build(&instance, &adapter, device.clone()),
    );

    let color_image_view = ImageView::new(
        device.clone(),
        color_image.clone(),
        vk::ImageViewType::TYPE_2D,
        vk::ImageAspectFlags::COLOR,
    );

    let depth_image_view = ImageView::new(
        device.clone(),
        depth_image.clone(),
        vk::ImageViewType::TYPE_2D,
        vk::ImageAspectFlags::DEPTH,
    );

    (
        swapchain_device,
        swapchain_khr,
        images,
        extent,
        image_views,
        color_image,
        depth_image,
        color_image_view,
        depth_image_view,
    )
}

fn create_image_views(
    swapchain_images: &Vec<vk::Image>,
    swapchain_image_format: vk::Format,
    device: Arc<Device>,
) -> Vec<vk::ImageView> {
    let mut image_views = Vec::new();

    for i in 0..swapchain_images.len() {
        let image_view_create_info = vk::ImageViewCreateInfo::default()
            .image(swapchain_images[i])
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(swapchain_image_format)
            .components(
                vk::ComponentMapping::default()
                    .r(vk::ComponentSwizzle::IDENTITY)
                    .g(vk::ComponentSwizzle::IDENTITY)
                    .b(vk::ComponentSwizzle::IDENTITY)
                    .a(vk::ComponentSwizzle::IDENTITY),
            )
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );

        let image_view = unsafe_vk_try!(
            device
                .ash_device
                .create_image_view(&image_view_create_info, None)
        );
        image_views.push(image_view)
    }
    image_views
}

pub(crate) struct SupportDetails {
    pub(crate) capabilities: vk::SurfaceCapabilitiesKHR,
    pub(crate) formats: Vec<vk::SurfaceFormatKHR>,
    pub(crate) present_modes: Vec<vk::PresentModeKHR>,
}

impl SupportDetails {
    pub(crate) fn query_support(physical_device: vk::PhysicalDevice, surface: &Surface) -> Self {
        unsafe {
            Self {
                capabilities: vk_try!(
                    surface
                        .surface_instance
                        .get_physical_device_surface_capabilities(
                            physical_device,
                            surface.surface_khr
                        )
                ),
                formats: vk_try!(
                    surface
                        .surface_instance
                        .get_physical_device_surface_formats(physical_device, surface.surface_khr)
                ),
                present_modes: vk_try!(
                    surface
                        .surface_instance
                        .get_physical_device_surface_present_modes(
                            physical_device,
                            surface.surface_khr
                        )
                ),
            }
        }
    }
}

fn choose_surface_format(surface_formats: Vec<vk::SurfaceFormatKHR>) -> vk::SurfaceFormatKHR {
    for surface_format in &surface_formats {
        if surface_format.format == vk::Format::B8G8R8A8_SRGB
            && surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        {
            return *surface_format;
        }
    }
    surface_formats[0]
}

fn choose_present_mode(present_modes: Vec<vk::PresentModeKHR>) -> vk::PresentModeKHR {
    for present_mode in &present_modes {
        if *present_mode == vk::PresentModeKHR::IMMEDIATE {
            return *present_mode;
        }
    }
    vk::PresentModeKHR::FIFO
}

fn choose_extent(
    capabilities: vk::SurfaceCapabilitiesKHR,
    width: u32,
    height: u32,
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        return capabilities.current_extent;
    }
    let mut actual_extent = vk::Extent2D { width, height };
    actual_extent.width = actual_extent.width.clamp(
        capabilities.min_image_extent.width,
        capabilities.max_image_extent.width,
    );
    actual_extent.height = actual_extent.height.clamp(
        capabilities.min_image_extent.height,
        capabilities.max_image_extent.height,
    );
    actual_extent
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        self.destroy();
    }
}
