struct VSInput {
    float2 in_position : POSITION;
    float3 in_color : COLOR;
};

struct VSOutput {
    float4 position : SV_Position;
    float3 frag_color : COLOR0;
};

[shader("vertex")]
VSOutput main(VSInput input) {
    VSOutput output;
    output.position = float4(input.in_position, 0.0, 1.0);
    output.frag_color = input.in_color;
    return output;
}