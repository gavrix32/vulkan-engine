[[vk::binding(1, 0)]] Sampler2D albedo_texture;
[[vk::binding(2, 0)]] Sampler2D normal_texture;
[[vk::binding(3, 0)]] Sampler2D metallic_roughness_texture;

struct UniformBuffer {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    float4 light_pos;
    float4 cam_pos;
};
ConstantBuffer<UniformBuffer> ubo;

static const float PI = 3.14159265;

// x - fragment position
// N - Normal
// L - light vector
// V - view vector
// H - halfway vector
// Lo - outgoing light
// Li - incoming light
// Le - emmited light
// F0 - base reflectance
// kD - diffuse fraction
// kS - specular fraction

// Approximation of the Fresnel Factor
float3 fresnel_schlick(float3 F0, float HdotV) {
    return F0 + (1.0 - F0) * pow(1.0 - HdotV, 5.0);
}

// GGX/Trowbridge-Reitz Normal Distribution Function
float distribution_GGX(float3 N, float NdotH, float alpha) {
    float alpha2 = pow(alpha, 2.0);
    float NdotH2 = pow(NdotH, 2.0);

    float numerator = alpha2;
    float denominator = PI * pow(NdotH2 * (alpha2 - 1.0) + 1.0, 2.0);

    return numerator / max(denominator, 0.000001);
}

// Smith Model
float geometry_smith(float3 N, float NdotL, float NdotV, float alpha) {
    return geometry_schlick_beckmann(N, NdotL, alpha) * geometry_schlick_beckmann(N, NdotV, alpha);
}

// Schlick-Beckmann Geometry Shadowing Function
float geometry_schlick_beckmann(float3 N, float NdotX, float alpha) {
    float k = alpha / 2.0;

    float numerator = NdotX;
    float denominator = NdotX * (1.0 - k) + k;

    return numerator / max(denominator, 0.000001);
}

[shader("fragment")]
void main(
    out float4 out_color: SV_Target0,
    float3 frag_pos,
    float2 tex_coord,
    float3 world_tangent,
    float3 world_bitangent,
    float3 world_normal,
) {
    float3 albedo = albedo_texture.Sample(tex_coord).rgb;

    float3 tangent_normal = normal_texture.Sample(tex_coord).rgb;
    float3x3 TBN = transpose(float3x3(
        normalize(world_tangent),
        normalize(world_bitangent),
        normalize(world_normal)
    ));
    float3 normal = normalize(mul(TBN, tangent_normal));

    // placeholder check
    if (all(tangent_normal == float3(1.0, 0.0, 0.0)) ||
        all(tangent_normal == float3(0.0, 0.0, 0.0))
    ) {
        normal = normalize(world_normal);
    }

    float3 metallic_roughness = metallic_roughness_texture.Sample(tex_coord).rgb;
    float metallic = metallic_roughness.b;
    float roughness = metallic_roughness.g;

    float3 light_color = 8.0;

    float3 N = normal;
    float3 V = normalize(ubo.cam_pos.xyz - frag_pos);
    float3 L = normalize(ubo.light_pos.xyz - frag_pos);
    float3 H = normalize(V + L);

    float3 F0 = 0.04;
    F0 = lerp(F0, albedo, metallic);

    float NdotL = saturate(dot(N, L));
    float NdotV = saturate(dot(N, V));
    float NdotH = saturate(dot(N, H));
    float HdotV = saturate(dot(H, V));

    float alpha = pow(roughness, 2.0);

    float D = distribution_GGX(N, NdotH, alpha);
    float G = geometry_smith(N, NdotL, NdotV, alpha);
    float3 F = fresnel_schlick(F0, HdotV);

    float3 lambert_diffuse = albedo / PI;

    float3 cook_torrance_numerator = D * G * F;
    float3 cook_torrance_denominator = 4.0 * NdotV * NdotL;
    float3 cook_torrance_specular = cook_torrance_numerator / max(cook_torrance_denominator, 0.000001);

    float3 kS = F;
    float3 kD = (1.0 - kS) * (1.0 - metallic);

    float3 BRDF = kD * lambert_diffuse + cook_torrance_specular;

    float distance = length(ubo.light_pos.xyz - frag_pos);
    float attenuation = 1.0 / pow(distance, 2.0);
    float3 Li = light_color * attenuation;
    float3 Le = 0.03 * albedo;
    float3 Lo = Le + saturate(BRDF * Li * NdotL);

    out_color = float4(Lo, 1.0);
}