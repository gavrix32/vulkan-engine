struct VSInput {
    float3 in_position;
    float3 in_normal;
    float4 in_tangent;
    float2 in_tex_coord;
};

struct VSOutput {
    float4 position : SV_Position;
    float3 frag_pos;
    float2 tex_coord;
    float3 world_tangent;
    float3 world_bitangent;
    float3 world_normal;
};

struct UniformBuffer {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    float4 light_pos;
    float4 cam_pos;
};
ConstantBuffer<UniformBuffer> ubo;

[shader("vertex")]
VSOutput main(VSInput input) {
    VSOutput output;

    float4x4 MVP = mul(mul(ubo.proj, ubo.view), ubo.model);
    output.position = mul(MVP, float4(input.in_position, 1.0));

    output.frag_pos = mul(ubo.model, float4(input.in_position, 1.0)).xyz;
    output.tex_coord = input.in_tex_coord;

    output.world_tangent = normalize(mul(float3x3(ubo.model), input.in_tangent.xyz));
    output.world_normal = normalize(mul(float3x3(ubo.model), input.in_normal));
    output.world_bitangent = normalize(cross(output.world_normal, output.world_tangent) * input.in_tangent.w);

    return output;
}