use crate::renderer::Vertex;
use crate::vulkan::device::Device;
use crate::vulkan::render_pass::RenderPass;
use ash::vk;
use std::io::Cursor;
use std::sync::Arc;

pub struct Pipeline {
    device: Arc<Device>,
    pub layout: vk::PipelineLayout,
    pub vk_pipeline: vk::Pipeline,
}

impl Pipeline {
    pub fn new(
        device: Arc<Device>,
        render_pass: &RenderPass,
        descriptor_set_layouts: &[vk::DescriptorSetLayout],
    ) -> Self {
        let vertex_shader_bytes = include_bytes!("../shaders/spirv/vertex.spv");
        let vertex_shader_code =
            ash::util::read_spv(&mut Cursor::new(vertex_shader_bytes)).unwrap();

        let fragment_shader_bytes = include_bytes!("../shaders/spirv/fragment.spv");
        let fragment_shader_code =
            ash::util::read_spv(&mut Cursor::new(fragment_shader_bytes)).unwrap();

        let vertex_shader_module =
            create_shader_module(device.clone(), vertex_shader_code.as_slice());
        let fragment_shader_module =
            create_shader_module(device.clone(), fragment_shader_code.as_slice());

        let vertex_shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_shader_module)
            .name(c"main");
        let fragment_shader_stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_shader_module)
            .name(c"main");

        let shader_stage_create_infos = [vertex_shader_stage_info, fragment_shader_stage_info];

        let binding_descriptions = [Vertex::get_binding_description()];
        let attribute_descriptions = Vertex::get_attribute_descriptions();

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer_state = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment_state = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);
        let color_blend_attachment_states = [color_blend_attachment_state];

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&color_blend_attachment_states);

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS);

        let layout_create_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&descriptor_set_layouts);

        let layout = unsafe {
            device
                .ash_device
                .create_pipeline_layout(&layout_create_info, None)
        }
        .expect("Failed to create pipeline layout");

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stage_create_infos)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&color_blend_state)
            .depth_stencil_state(&depth_stencil_state)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .render_pass(render_pass.vk_render_pass)
            .subpass(0);

        let pipelines = unsafe {
            device.ash_device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_create_info],
                None,
            )
        }
        .expect("Failed to create graphics pipelines");

        unsafe {
            device
                .ash_device
                .destroy_shader_module(vertex_shader_module, None);
            device
                .ash_device
                .destroy_shader_module(fragment_shader_module, None);
        }

        Self {
            device,
            layout,
            vk_pipeline: pipelines[0],
        }
    }
}

fn create_shader_module(device: Arc<Device>, words: &[u32]) -> vk::ShaderModule {
    let shader_module_create_info = vk::ShaderModuleCreateInfo::default().code(words);
    unsafe {
        device
            .ash_device
            .create_shader_module(&shader_module_create_info, None)
    }
    .expect("Failed to create shader module")
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.device
                .ash_device
                .destroy_pipeline(self.vk_pipeline, None);
            self.device
                .ash_device
                .destroy_pipeline_layout(self.layout, None);
        }
    }
}
