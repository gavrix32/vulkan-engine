struct VSInput {
    float3 in_position;
    float3 in_normal;
    float4 in_tangent;
    float2 in_tex_coord;
};

struct VSOutput {
    float4 position : SV_Position;
    float3 out_position;
    float3 out_normal;
    float3 out_tangent;
    float out_tangent_sign;
    float2 out_tex_coord;
};

struct UniformBuffer {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
};
ConstantBuffer<UniformBuffer> ubo;

[shader("vertex")]
VSOutput main(VSInput input) {
    VSOutput output;

    float4x4 MVP = mul(mul(ubo.proj, ubo.view), ubo.model);
    output.position = mul(MVP, float4(input.in_position, 1.0));
    output.out_position = mul(ubo.model, float4(input.in_position, 1.0)).xyz;

    output.out_normal = normalize(mul(float3x3(ubo.model), input.in_normal));
    output.out_tangent = normalize(mul(float3x3(ubo.model), input.in_tangent.xyz));
    output.out_tangent_sign = input.in_tangent.w;
    output.out_tex_coord = input.in_tex_coord;

    return output;
}