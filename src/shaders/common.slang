struct UniformBuffer {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    float4 light_pos;
    float4 cam_pos;
    uint env_index;
};
[vk::binding(0, 0)] ConstantBuffer<UniformBuffer> uniforms;