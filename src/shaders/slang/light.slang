import common;

struct VertexInput {
    float3 in_position;
    float3 in_normal;
    float4 in_tangent;
    float2 in_tex_coord;
    uint3 in_material;
};

struct VertexOutput {
    float4 sv_position : SV_Position;
    float4 light_pos;
};

[shader("vertex")]
VertexOutput vertex_main(VertexInput input) {
    VertexOutput output;

    float4x4 MVP = mul(mul(uniforms.proj, uniforms.view), uniforms.model);
    output.sv_position = mul(MVP, float4(input.in_position, 1.0));
    output.light_pos = uniforms.light_pos;

    return output;
}

[shader("fragment")]
void fragment_main(
    out float4 out_color : SV_Target0,
    float4 out_light_pos,
) {
    out_color = float4(float3(1.0), 1.0);
}