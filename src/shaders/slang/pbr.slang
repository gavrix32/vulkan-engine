import common;

struct UniformBuffer {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    float4 light_pos;
    float4 cam_pos;
};

[vk::binding(0, 0)]
ConstantBuffer<UniformBuffer> uniforms;

[vk::binding(0, 1)]
SamplerCube cubemap_texture;

[vk::binding(1, 1)]
SamplerCube irradiance_texture;

[vk::binding(2, 1)]
SamplerCube prefilter_texture;

[vk::binding(3, 1)]
Sampler2D lut_texture;

[vk::binding(0, 2)]
Sampler2D textures[];

struct VertexInput {
    float3 in_position;
    float3 in_normal;
    float4 in_tangent;
    float2 in_tex_coord;
    uint3 in_material_indices;
};

struct VertexOutput {
    float4 sv_position : SV_Position;
    float3 frag_pos;
    float2 tex_coord;
    float3 world_tangent;
    float3 world_bitangent;
    float3 world_normal;
    uint3 material_indices;
};

[shader("vertex")]
VertexOutput vertex_main(VertexInput input) {
    VertexOutput output;

    float4x4 MVP = mul(mul(uniforms.proj, uniforms.view), uniforms.model);
    output.sv_position = mul(MVP, float4(input.in_position, 1.0));

    output.frag_pos = mul(uniforms.model, float4(input.in_position, 1.0)).xyz;
    output.tex_coord = input.in_tex_coord;

    output.world_tangent = normalize(mul(float3x3(uniforms.model), input.in_tangent.xyz));
    output.world_normal = normalize(mul(float3x3(uniforms.model), input.in_normal));
    output.world_bitangent = normalize(cross(output.world_normal, output.world_tangent) * input.in_tangent.w);

    output.material_indices = input.in_material_indices;

    return output;
}

// x - fragment position
// N - normal
// L - light vector
// V - view vector
// H - halfway vector
// Lo - outgoing light
// Li - incoming light
// Le - emitted light
// F0 - base reflectance
// kD - diffuse fraction
// kS - specular fraction

// Schlick Fresnel Approximation
float3 fresnel_schlick(float3 F0, float HdotV) {
    return F0 + (1.0 - F0) * pow(1.0 - HdotV, 5.0);
}

float3 fresnel_schlick_roughness(float NdotV, float3 F0, float roughness) {
    return F0 + (max(float3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - NdotV, 0.0, 1.0), 5.0);
}

[shader("fragment")]
void fragment_main(
    out float4 out_color : SV_Target0,
    float4 sv_position : SV_Position,
    VertexOutput input,
) {
    uint albedo_index = input.material_indices[0];
    uint normal_index = input.material_indices[1];
    uint metallic_roughness_index = input.material_indices[2];

    float3 albedo = textures[albedo_index].Sample(input.tex_coord).rgb;

    float4 tangent_normal = textures[normal_index].Sample(input.tex_coord);
    tangent_normal.rgb = tangent_normal.rgb * 2.0 - 1.0;

    float3x3 TBN = float3x3(
        input.world_tangent,
        input.world_bitangent,
        input.world_normal,
    );
    float3 normal = normalize(mul(tangent_normal.rgb, TBN));

    if (tangent_normal.a == 0.0) {
        normal = input.world_normal;
    }

    float3 metallic_roughness = textures[metallic_roughness_index].Sample(input.tex_coord).rgb;
    float metallic = metallic_roughness.b;
    float roughness = metallic_roughness.g;

    float3 light_color = 8.0;

    float3 N = normal;
    float3 V = normalize(uniforms.cam_pos.xyz - input.frag_pos);
    float3 L = normalize(uniforms.light_pos.xyz - input.frag_pos);
    float3 H = normalize(V + L);

    float3 F0 = 0.04;
    F0 = lerp(F0, albedo, metallic);

    float NdotL = saturate(dot(N, L));
    float NdotV = saturate(dot(N, V));
    float NdotH = saturate(dot(N, H));
    float HdotV = saturate(dot(H, V));

    float alpha = pow(roughness, 2.0);

    float D = distribution_GGX(N, NdotH, alpha);
    float G = geometry_smith(NdotL, NdotV, alpha);
    float3 F = fresnel_schlick(F0, HdotV);
    float3 R = normalize(reflect(-V, N));

    float3 numerator = D * G * F;
    float3 denominator = 4.0 * NdotV * NdotL;
    float3 cook_torrance_specular = numerator / max(denominator, EPSILON);

    float3 kS = F;
    float3 kD = (1.0 - kS) * (1.0 - metallic);

    float distance = length(uniforms.light_pos.xyz - input.frag_pos);
    float attenuation = 1.0 / pow(distance, 2.0);
    float3 radiance = light_color * attenuation;

    float3 Lo = (kD * albedo / PI + cook_torrance_specular) * radiance * NdotL;


    float3 kS_IBL = fresnel_schlick_roughness(NdotV, F0, roughness);
    float3 kD_IBL = (1.0 - kS_IBL) * (1.0 - metallic);

    float3 irradiance = irradiance_texture.SampleLevel(N, 0.0).rgb;
    float3 diffuse = irradiance * albedo;

    float3 prefiltered_color = prefilter_texture.SampleLevel(R, roughness * MAX_MIP_LEVELS).rgb;

    float2 brdf = lut_texture.SampleLevel(float2(NdotV, roughness), 0.0).rg;
    float3 specular = prefiltered_color * (kS_IBL * brdf.x + brdf.y);
    float3 ambient = kD_IBL * diffuse + specular;

    float3 color = ambient/* + Lo*/;

    color = neutral(color * EXPOSURE);

    out_color = float4(color, 1.0);
}