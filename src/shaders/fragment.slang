[[vk::binding(1, 0)]] Sampler2D albedo_texture;
[[vk::binding(2, 0)]] Sampler2D normal_texture;
[[vk::binding(3, 0)]] Sampler2D metallic_roughness_texture;

[shader("fragment")]
void main(
    out float4 out_color : SV_Target0,
    float3 out_position,
    float3 out_normal,
    float3 out_tangent,
    float out_tangent_sign,
    float2 out_tex_coord,
    float4 out_light_pos,
) {
    float3 albedo = albedo_texture.Sample(out_tex_coord).rgb;
    float3 normal = normal_texture.Sample(out_tex_coord).rgb;
    float3 metallic_roughness = metallic_roughness_texture.Sample(out_tex_coord).rgb;

    float3 T = out_tangent;
    float3 B = normalize(cross(out_normal, out_tangent) * out_tangent_sign);
    float3 N = out_normal;
    float3x3 TBN = float3x3(T, B, N);

    float3 final_normal = mul(normal, TBN);
    if (all(normal == float3(1.0, 0.0, 1.0))) {
        final_normal = N;
    }

    float3 point = out_light_pos.xyz;
    float3 light_dir = normalize(point - out_position);

    float3 ambient = 0.1;
    float3 diffuse = max(dot(final_normal, light_dir), 0.0);
    //out_color = float4((ambient + diffuse) * albedo, 1.0);
    out_color = float4(metallic_roughness, 1.0);
}