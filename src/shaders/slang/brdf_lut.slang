import common;

[vk::binding(0, 0)]
[vk::image_format("rgba32f")]
RWTexture2D output_image;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_ID : SV_DispatchThreadID) {
    uint width, height;
    output_image.GetDimensions(width, height);

    float2 uv = (float2(thread_ID.xy) + 0.5) / float2(width, height);

    float NdotV = uv.x;
    float roughness = uv.y;

    float3 V;
    V.x = sqrt(1.0 - NdotV * NdotV);
    V.y = 0.0;
    V.z = NdotV;

    float A = 0.0;
    float B = 0.0;

    float3 N = float3(0.0, 0.0, 1.0);

    const uint LUT_SAMPLE_COUNT = 1024;
    for (uint i = 0; i < LUT_SAMPLE_COUNT; ++i) {
        float2 Xi = hammersley(i, float(LUT_SAMPLE_COUNT));
        float3 H = importance_sample_GGX(Xi, N, roughness);
        float3 L = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = saturate(L.z);
        float NdotH = saturate(H.z);
        float VdotH = saturate(dot(V, H));

        if(NdotL > 0.0) {
            float G = geometry_smith(NdotL, NdotV, roughness);
            float G_Vis = (G * VdotH) / (NdotH * NdotV);
            float Fc = pow(1.0 - VdotH, 5.0);

            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    A /= float(LUT_SAMPLE_COUNT);
    B /= float(LUT_SAMPLE_COUNT);

    output_image[thread_ID.xy] = float4(A, B, 0.0, 1.0);
}