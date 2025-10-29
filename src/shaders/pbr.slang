import common;

[vk::binding(1, 0)] Sampler2D textures[];

static const float PI = 3.14159265;
static const float EPSILON = 0.00001;

struct VertexInput {
    float3 in_position;
    float3 in_normal;
    float4 in_tangent;
    float2 in_tex_coord;
    uint3 in_material;
};

struct VertexOutput {
    float4 sv_position : SV_Position;
    float3 frag_pos;
    float2 tex_coord;
    float3 world_tangent;
    float3 world_bitangent;
    float3 world_normal;
    uint3 material;
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

    output.material = input.in_material;

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

// GGX/Trowbridge-Reitz Normal Distribution Function
float distribution_GGX(float3 N, float NdotH, float alpha) {
    float alpha2 = pow(alpha, 2.0);
    float NdotH2 = pow(NdotH, 2.0);

    float numerator = alpha2;
    float denominator = PI * pow(NdotH2 * (alpha2 - 1.0) + 1.0, 2.0);

    return numerator / max(denominator, EPSILON);
}

// Smith Geometry Shadowing Function (light and view components)
float geometry_smith(float3 N, float NdotL, float NdotV, float alpha) {
    return geometry_schlick_GGX(N, NdotL, alpha) * geometry_schlick_GGX(N, NdotV, alpha);
}

// Schlick-Beckmann Geometry Shadowing Function
float geometry_schlick_GGX(float3 N, float NdotX, float alpha) {
    float k = alpha / 2.0;

    float numerator = NdotX;
    float denominator = NdotX * (1.0 - k) + k;

    return numerator / max(denominator, EPSILON);
}

float2 sample_env(float3 dir) {
    float2 uv;
    uv.x = atan2(dir.z, dir.x) / (2.0 * PI) + 0.5;
    uv.y = asin(clamp(dir.y, -1.0, 1.0)) / PI + 0.5;
    return uv;
}

[shader("fragment")]
void fragment_main(
    out float4 out_color : SV_Target0,
    float4 sv_position : SV_Position,
    VertexOutput input,
) {
    uint albedo_index = input.material[0];
    uint normal_index = input.material[1];
    uint metallic_roughness_index = input.material[2];

    float3 albedo = textures[albedo_index].Sample(input.tex_coord).rgb;

    float3 tangent_normal = textures[normal_index].Sample(input.tex_coord).rgb;
    float3x3 TBN = transpose(float3x3(
        normalize(input.world_tangent),
        normalize(input.world_bitangent),
        normalize(input.world_normal)
    ));
    float3 normal = normalize(mul(TBN, tangent_normal));

    // placeholder check
    if (all(tangent_normal == float3(1.0, 0.0, 0.0)) ||
        all(tangent_normal == float3(0.0, 0.0, 0.0))
    ) {
        normal = normalize(input.world_normal);
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
    float G = geometry_smith(N, NdotL, NdotV, alpha);
    float3 F = fresnel_schlick(F0, HdotV);

    float3 lambert_diffuse = albedo / PI;

    float3 numerator = D * G * F;
    float3 denominator = 4.0 * NdotV * NdotL;
    float3 cook_torrance_specular = numerator / max(denominator, EPSILON);

    float3 kS = F;
    float3 kD = (1.0 - kS) * (1.0 - metallic);

    float3 BRDF = kD * lambert_diffuse + cook_torrance_specular;

    float distance = length(uniforms.light_pos.xyz - input.frag_pos);
    float attenuation = 1.0 / pow(distance, 2.0);
    float3 Li = light_color * attenuation;
    float3 Le = 0.02 * albedo;
    float3 Lo = Le + BRDF * Li * NdotL;

    // Environment
    //float3 R = reflect(-V, N);
    //float2 sph_uv = sample_env(V);
    //float3 env = textures[uniforms.env_index].Sample(sph_uv).rgb;
    //float exposure = 1.0;
    //env = float3(1.0) - exp(-env * exposure);

    //float depth = sv_position.a;
    //if (depth == 0.0) {
    //    Lo = env;
    //}

    out_color = float4(Lo, 1.0);
}