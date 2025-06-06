[shader("vertex")]
void main(
    uint vertex_index : SV_VertexID,
    out float4 out_position : SV_Position,
    out float3 frag_color : COLOR0,
) {
    float2 positions[3] = {
        float2(0.0, -0.5),
        float2(0.5, 0.5),
        float2(-0.5, 0.5),
    };

    float3 colors[3] = {
        float3(1.0, 0.0, 0.0),
        float3(0.0, 1.0, 0.0),
        float3(0.0, 0.0, 1.0),
    };

    out_position = float4(positions[vertex_index], 0.0, 1.0);
    frag_color = colors[vertex_index];
}
