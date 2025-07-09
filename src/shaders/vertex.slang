struct VSInput {
    float2 in_position : POSITION;
    float3 in_color : COLOR;
};

struct VSOutput {
    float4 position : SV_Position;
    float3 frag_color : COLOR0;
};

struct UniformData {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
}
ConstantBuffer<UniformData> uniform_buffer;

[shader("vertex")]
VSOutput main(VSInput input) {
    VSOutput output;
    output.position = mul(uniform_buffer.proj * uniform_buffer.view * uniform_buffer.model, float4(input.in_position, 0.0, 1.0));
    output.frag_color = input.in_color;
    return output;
}