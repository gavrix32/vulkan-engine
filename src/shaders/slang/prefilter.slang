import common;

[vk::binding(0, 0)]
SamplerCube cubemap_texture;

[vk::binding(1, 0)]
[vk::image_format("rgba32f")]
RWTexture2DArray<float4> output_image;

struct PushConstants {
    float roughness;
};
[vk::push_constant] ConstantBuffer<PushConstants> push_constants;

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

        float NdotL = saturate(dot(N, L));
        float NdotH = saturate(dot(N, H));

        if (NdotL > 0.0) {
            float D = distribution_GGX(N, NdotH, push_constants.roughness);
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