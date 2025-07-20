use crate::vulkan::device::Device;
use ash::vk;
use std::sync::Arc;

pub struct RenderPass {
    device: Arc<Device>,
    pub vk_render_pass: vk::RenderPass,
}

impl RenderPass {
    pub fn new(device: Arc<Device>, image_format: vk::Format) -> Self {
        let color_attachment_description = vk::AttachmentDescription::default()
            .format(image_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_attachment_reference = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let color_attachment_references = [color_attachment_reference];

        let depth_attachment_description = vk::AttachmentDescription::default()
            .format(vk::Format::D32_SFLOAT_S8_UINT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let depth_attachment_reference = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let attachment_descriptions = [color_attachment_description, depth_attachment_description];

        let subpass_description = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_references)
            .depth_stencil_attachment(&depth_attachment_reference);
        let subpass_descriptions = [subpass_description];

        let subpass_dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            );
        let subpass_dependencies = [subpass_dependency];

        let render_pass_create_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachment_descriptions)
            .subpasses(&subpass_descriptions)
            .dependencies(&subpass_dependencies);

        let vk_render_pass = unsafe {
            device
                .ash_device
                .create_render_pass(&render_pass_create_info, None)
        }
        .expect("Failed to create render pass");

        Self {
            device,
            vk_render_pass,
        }
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        unsafe {
            self.device
                .ash_device
                .destroy_render_pass(self.vk_render_pass, None);
        }
    }
}
