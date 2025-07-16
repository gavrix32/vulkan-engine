struct VSInput {
    float3 in_position;
    float3 in_color;
    float2 in_tex_coord;
};

struct VSOutput {
    float4 position : SV_Position;
    float3 frag_color;
    float2 frag_tex_coord;
};

struct UniformBuffer {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
};
ConstantBuffer<UniformBuffer> uniform_buffer;

[shader("vertex")]
VSOutput main(VSInput input) {
    VSOutput output;
    output.position = mul(mul(mul(uniform_buffer.proj, uniform_buffer.view), uniform_buffer.model), float4(input.in_position, 1.0));
    output.frag_color = input.in_color;
    output.frag_tex_coord = input.in_tex_coord;
    return output;
}