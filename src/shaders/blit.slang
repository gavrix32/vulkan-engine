[vk::binding(0, 0)] Sampler2D env_texture;

static const float PI = 3.14159265;
static const float EPSILON = 0.00001;

struct VertexOutput {
    float4 sv_position : SV_Position;
};

[shader("vertex")]
VertexOutput vertex_main(uint vertexID : SV_VertexID) {
    VertexOutput output;

    const float2 positions[3] = {
        float2(-1.0, 3.0),
        float2(-1.0, -1.0),
        float2(3.0, -1.0)
    };

    const float2 tex_coords[3] = {
       float2(0.0, -1.0),
       float2(0.0, 2.0),
       float2(2.0, 2.0)
    };

    output.sv_position = float4(positions[vertexID], 0.0, 1.0);

    return output;
}

[shader("fragment")]
void fragment_main(
    float4 sv_position : SV_Position,
    out float4 out_color : SV_Target0
) {
    out_color = float4(1.0, 0.0, 0.0, 1.0);
}