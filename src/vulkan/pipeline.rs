use crate::unsafe_vk_try;
use crate::vulkan::device::Device;
use ash::vk;
use std::ffi::CStr;
use std::io::Cursor;
use std::sync::Arc;

pub struct Pipeline {
    device: Arc<Device>,
    pub layout: vk::PipelineLayout,
    pub handle: vk::Pipeline,
}

#[derive(Default, Clone)]
pub struct PipelineBuilder<'a> {
    shader_stage_data: Vec<(Vec<u8>, vk::ShaderStageFlags, &'a CStr)>,
    vertex_input_state: vk::PipelineVertexInputStateCreateInfo<'a>,
    input_assembly_state: vk::PipelineInputAssemblyStateCreateInfo<'a>,
    dynamic_state: vk::PipelineDynamicStateCreateInfo<'a>,
    viewport_state: vk::PipelineViewportStateCreateInfo<'a>,
    rasterization_state: vk::PipelineRasterizationStateCreateInfo<'a>,
    multisample_state: vk::PipelineMultisampleStateCreateInfo<'a>,
    color_write_mask: vk::ColorComponentFlags,
    depth_stencil_state: vk::PipelineDepthStencilStateCreateInfo<'a>,
    descriptor_set_layouts: &'a [vk::DescriptorSetLayout],
    push_constant_ranges: &'a [vk::PushConstantRange],
    pipeline_layout_info: vk::PipelineLayoutCreateInfo<'a>,
    color_format: vk::Format,
    depth_format: vk::Format,
}

impl<'a> PipelineBuilder<'a> {
    pub fn shader_stage(
        mut self,
        bytes: Vec<u8>,
        stage: vk::ShaderStageFlags,
        name: &'a CStr,
    ) -> Self {
        self.shader_stage_data.push((bytes, stage, name));
        self
    }

    pub fn vertex_input(
        mut self,
        binding_descriptions: &'a [vk::VertexInputBindingDescription],
        attribute_descriptions: &'a [vk::VertexInputAttributeDescription],
    ) -> Self {
        self.vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(binding_descriptions)
            .vertex_attribute_descriptions(attribute_descriptions);
        self
    }

    pub fn input_assembly(
        mut self,
        topology: vk::PrimitiveTopology,
        primitive_restart: bool,
    ) -> Self {
        self.input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(topology)
            .primitive_restart_enable(primitive_restart);
        self
    }

    pub fn dynamic_states(mut self, states: &'a [vk::DynamicState]) -> Self {
        self.dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(states);
        self
    }

    pub fn viewport_state(mut self, viewport_count: u32, scissor_count: u32) -> Self {
        self.viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(viewport_count)
            .scissor_count(scissor_count);
        self
    }

    pub fn rasterization_state(
        mut self,
        depth_clamp: bool,
        rasterizer_discard: bool,
        polygon_mode: vk::PolygonMode,
        line_width: f32,
        cull_mode: vk::CullModeFlags,
        front_face: vk::FrontFace,
        depth_bias: bool,
    ) -> Self {
        self.rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(depth_clamp)
            .rasterizer_discard_enable(rasterizer_discard)
            .polygon_mode(polygon_mode)
            .line_width(line_width)
            .cull_mode(cull_mode)
            .front_face(front_face)
            .depth_bias_enable(depth_bias);
        self
    }

    pub fn multisample_state(mut self, msaa_samples: vk::SampleCountFlags) -> Self {
        self.multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(msaa_samples);
        self
    }

    pub fn color_blend_state(mut self, color_write_mask: vk::ColorComponentFlags) -> Self {
        self.color_write_mask = color_write_mask;
        self
    }

    pub fn descriptor_set_layouts(
        mut self,
        descriptor_set_layouts: &'a [vk::DescriptorSetLayout],
    ) -> Self {
        self.descriptor_set_layouts = descriptor_set_layouts;
        self.pipeline_layout_info = self
            .pipeline_layout_info
            .set_layouts(&self.descriptor_set_layouts);

        self
    }

    pub fn push_constant_ranges(
        mut self,
        push_constant_ranges: &'a [vk::PushConstantRange],
    ) -> Self {
        self.push_constant_ranges = push_constant_ranges;
        self.pipeline_layout_info = self
            .pipeline_layout_info
            .push_constant_ranges(&self.push_constant_ranges);

        self
    }

    pub fn depth_stencil_state(
        mut self,
        depth_test: bool,
        depth_write: bool,
        depth_compare_op: vk::CompareOp,
    ) -> Self {
        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(depth_test)
            .depth_write_enable(depth_write)
            .depth_compare_op(depth_compare_op);

        self.depth_stencil_state = depth_stencil_state;
        self
    }

    pub fn formats(mut self, color: vk::Format, depth: vk::Format) -> Self {
        self.color_format = color;
        self.depth_format = depth;
        self
    }

    pub fn build_graphics_pipeline(self, device: Arc<Device>) -> Pipeline {
        let mut shader_stages = Vec::new();

        for (bytes, stage, c_name) in self.shader_stage_data {
            let words =
                ash::util::read_spv(&mut Cursor::new(bytes)).expect("Failed to read SPIR-V");

            let shader_module_create_info = vk::ShaderModuleCreateInfo::default().code(&words);
            let module = unsafe_vk_try!(
                device
                    .handle
                    .create_shader_module(&shader_module_create_info, None)
            );

            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(stage)
                .module(module)
                .name(c_name);

            shader_stages.push(stage);
        }

        let color_blend_attachment_state = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(self.color_write_mask)
            .blend_enable(false);
        let color_blend_attachment_states = [color_blend_attachment_state];

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&color_blend_attachment_states);

        let layout = unsafe_vk_try!(
            device
                .handle
                .create_pipeline_layout(&self.pipeline_layout_info, None)
        );

        let color_formats = [self.color_format];

        let mut pipeline_rendering_create_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(self.depth_format);

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&self.vertex_input_state)
            .input_assembly_state(&self.input_assembly_state)
            .viewport_state(&self.viewport_state)
            .rasterization_state(&self.rasterization_state)
            .multisample_state(&self.multisample_state)
            .color_blend_state(&color_blend_state)
            .depth_stencil_state(&self.depth_stencil_state)
            .dynamic_state(&self.dynamic_state)
            .layout(layout)
            .push_next(&mut pipeline_rendering_create_info);

        let pipelines = unsafe_vk_try!(device.handle.create_graphics_pipelines(
            vk::PipelineCache::null(),
            &[pipeline_create_info],
            None,
        ));

        for stage in &shader_stages {
            unsafe {
                device.handle.destroy_shader_module(stage.module, None);
            }
        }

        Pipeline {
            device,
            layout,
            handle: pipelines[0],
        }
    }

    pub fn build_compute_pipeline(self, device: Arc<Device>) -> Pipeline {
        let (bytes, stage, c_name) = self
            .shader_stage_data
            .last()
            .expect("Shader stage data not found");

        let words = ash::util::read_spv(&mut Cursor::new(bytes)).expect("Failed to read SPIR-V");

        let shader_module_create_info = vk::ShaderModuleCreateInfo::default().code(&words);
        let module = unsafe_vk_try!(
            device
                .handle
                .create_shader_module(&shader_module_create_info, None)
        );

        let shader_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(*stage)
            .module(module)
            .name(c_name);

        let layout = unsafe_vk_try!(
            device
                .handle
                .create_pipeline_layout(&self.pipeline_layout_info, None)
        );

        let pipeline_create_info = vk::ComputePipelineCreateInfo::default()
            .stage(shader_stage)
            .layout(layout);

        let pipelines = unsafe_vk_try!(device.handle.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[pipeline_create_info],
            None,
        ));

        unsafe {
            device
                .handle
                .destroy_shader_module(shader_stage.module, None);
        }

        Pipeline {
            device,
            layout,
            handle: pipelines[0],
        }
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.handle.destroy_pipeline(self.handle, None);
            self.device
                .handle
                .destroy_pipeline_layout(self.layout, None);
        }
    }
}
