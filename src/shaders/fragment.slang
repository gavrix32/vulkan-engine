[[vk::binding(1, 0)]] Sampler2D texture;

[shader("fragment")]
void main(
    out float4 out_color : SV_Target0,
    float3 frag_color,
    float2 frag_tex_coord,
) {
    out_color = texture.Sample(frag_tex_coord);
}