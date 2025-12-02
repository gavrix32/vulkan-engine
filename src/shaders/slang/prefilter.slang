[vk::binding(0, 0)]
SamplerCube cubemap_texture;

[vk::binding(1, 0)]
[vk::image_format("rgba32f")]
RWTexture2DArray<float4> output_image;

struct PushConstants {
    float roughness;
};
[vk::push_constant] ConstantBuffer<PushConstants> push_constants;

static const uint SAMPLE_COUNT = 4096;
static const uint MAX_MIP_LEVELS = 5;
static const float PI = 3.14159265;

float distribution_GGX(float3 N, float3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}

// https://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
float2 hammersley(float i, float N) {
    uint bits = uint(i);
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);

    float radical_inverse_VdC = float(bits) * 2.3283064365386963e-10; // / 0x100000000

    return float2(i / N, radical_inverse_VdC);
}

float3 importance_sample_GGX(float2 Xi, float3 N, float roughness) {
	float a = roughness * roughness;

	float phi = 2.0 * PI * Xi.x;
	float cos_theta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
	float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

	float3 H;
	H.x = cos(phi) * sin_theta;
	H.y = sin(phi) * sin_theta;
	H.z = cos_theta;

	float3 up = abs(N.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
	float3 tangent = normalize(cross(up, N));
	float3 bitangent = cross(N, tangent);

	float3 sample_vector = tangent * H.x + bitangent * H.y + N * H.z;
	return normalize(sample_vector);
}

float3 cosine_sample_hemisphere(float2 Xi){
    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt(1.0 - Xi.y);
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    return float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

float3 tangent_to_world(float3 vec, float3 N, float3 right, float3 up) {
    return right * vec.x + up * vec.y + N * vec.z;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_ID : SV_DispatchThreadID) {
    uint width, height, elements;
    output_image.GetDimensions(width, height, elements);

    float2 uv = (float2(thread_ID.xy) + 0.5) / float2(width, height) * 2.0 - 1.0;

    float3 N = float3(0.0);
    switch (thread_ID.z) {
        case 0: N = float3(1.0, -uv.y, -uv.x); break;
        case 1: N = float3(-1.0, -uv.y, uv.x); break;
        case 2: N = float3(uv.x, 1.0, uv.y); break;
        case 3: N = float3(uv.x, -1.0, -uv.y); break;
        case 4: N = float3(uv.x, -uv.y, 1.0); break;
        case 5: N = float3(-uv.x, -uv.y, -1.0); break;
    }
    N = normalize(N);

    float3 R = N;
    float3 V = R;

    float3 prefiltered_color = float3(0.0);
    float total_weight = 0.0;

    for (uint i = 0; i < SAMPLE_COUNT; ++i) {
        float2 Xi = hammersley(float(i), SAMPLE_COUNT);
        float3 H = importance_sample_GGX(Xi, N, push_constants.roughness);
        float3 L  = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.0) {
            float D = distribution_GGX(N, H, push_constants.roughness);
            float NdotH = max(dot(N, H), 0.0);
            float HdotV = max(dot(H, V), 0.0);
            float pdf = D * NdotH / (4.0 * HdotV) + 0.0001;

            float resolution = 1024.0;
            float texel = 4.0 * PI / (6.0 * resolution * resolution);
            float sample = 1.0 / (float(SAMPLE_COUNT) * pdf + 0.0001);

            float mip_level = push_constants.roughness == 0.0 ? 0.0 : 0.5 * log2(sample / texel);

            prefiltered_color += cubemap_texture.SampleLevel(L, mip_level).rgb * NdotL;
            total_weight += NdotL;
        }
    }
    prefiltered_color = prefiltered_color / total_weight;

    output_image[thread_ID] = float4(prefiltered_color, 1.0);
}