[vk::binding(0, 0)]
Sampler2D env_texture;

[vk::binding(1, 0)]
[vk::image_format("rgba32f")]
RWTexture2DArray<float4> output_image;

static const float PI = 3.14159265;

float2 sample_env(float3 dir) {
    float2 uv;
    uv.x = atan2(dir.z, dir.x) / (2.0 * PI) + 0.5;
    uv.y = acos(dir.y) / PI;
    return uv;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 thread_ID : SV_DispatchThreadID) {
    uint width, height, elements;
    output_image.GetDimensions(width, height, elements);

    float2 uv = (float2(thread_ID.xy) + 0.5) / float2(width, height) * 2.0 - 1.0;

    float3 dir;
    switch (thread_ID.z) {
        case 0: dir = float3(1.0, -uv.y, -uv.x); break;
        case 1: dir = float3(-1.0, -uv.y, uv.x); break;
        case 2: dir = float3(uv.x, 1.0, uv.y); break;
        case 3: dir = float3(uv.x, -1.0, -uv.y); break;
        case 4: dir = float3(uv.x, -uv.y, 1.0); break;
        case 5: dir = float3(-uv.x, -uv.y, -1.0); break;
    }
    dir = normalize(dir);

    float2 sph_uv = sample_env(dir);
    float3 env = env_texture.SampleLevel(sph_uv, 0).rgb;

    output_image[thread_ID] = float4(env, 1.0);
}