struct VSInput {
    float3 in_position;
    float3 in_normal;
    float4 in_tangent;
    float2 in_tex_coord;
};

struct VSOutput {
    float4 position : SV_Position;
    float4 out_light_pos;
};

struct UniformBuffer {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    float4 light_pos;
};
ConstantBuffer<UniformBuffer> ubo;

[shader("vertex")]
VSOutput main(VSInput input) {
    VSOutput output;

    float4x4 MVP = mul(mul(ubo.proj, ubo.view), ubo.model);
    output.position = mul(MVP, float4(input.in_position, 1.0));
    output.out_light_pos = ubo.light_pos;

    return output;
}