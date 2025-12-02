import common;

[vk::binding(1, 0)]
SamplerCube cubemap_texture;

struct VertexOutput {
    float4 sv_position : SV_Position;
    float3 uvw;
};

static const float3 VERTICES[36] = {
    // Right face
    float3( 1, -1,  1), float3( 1, -1, -1), float3( 1,  1, -1),
    float3( 1,  1, -1), float3( 1,  1,  1), float3( 1, -1,  1),

    // Left face
    float3(-1, -1, -1), float3(-1, -1,  1), float3(-1,  1,  1),
    float3(-1,  1,  1), float3(-1,  1, -1), float3(-1, -1, -1),

    // Top face
    float3(-1,  1,  1), float3( 1,  1,  1), float3( 1,  1, -1),
    float3( 1,  1, -1), float3(-1,  1, -1), float3(-1,  1,  1),

    // Bottom face
    float3(-1, -1, -1), float3( 1, -1, -1), float3( 1, -1,  1),
    float3( 1, -1,  1), float3(-1, -1,  1), float3(-1, -1, -1),

    // Front face
    float3(-1, -1,  1), float3( 1, -1,  1), float3( 1,  1,  1),
    float3( 1,  1,  1), float3(-1,  1,  1), float3(-1, -1,  1),

    // Back face
    float3( 1, -1, -1), float3(-1, -1, -1), float3(-1,  1, -1),
    float3(-1,  1, -1), float3( 1,  1, -1), float3( 1, -1, -1),
};

[shader("vertex")]
VertexOutput vertex_main(uint vertexID : SV_VertexID) {
    VertexOutput output;

    float3 pos = VERTICES[vertexID];

    output.uvw = pos;

    float4x4 view_no_translation = uniforms.view;
    view_no_translation[0][3] = 0.0;
    view_no_translation[1][3] = 0.0;
    view_no_translation[2][3] = 0.0;

    float4 clip_pos = mul(uniforms.proj, mul(view_no_translation, float4(pos, 1.0)));

    output.sv_position = clip_pos.xyww;

    return output;
}

[shader("fragment")]
void fragment_main(
    VertexOutput input,
    out float4 out_color : SV_Target0
) {
    float3 color = cubemap_texture.SampleLevel(input.uvw, 0.0).rgb;

    color = aces_film(color * EXPOSURE);

    out_color = float4(color, 1.0);
}