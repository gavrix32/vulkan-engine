[vk::binding(0, 0)]
SamplerCube cubemap_texture;

[vk::binding(1, 0)]
[vk::image_format("rgba32f")]
RWTexture2DArray<float4> output_image;

static const uint SAMPLE_COUNT = 4096;
static const float PI = 3.14159265;

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

float3 cosine_sample_hemisphere(float2 Xi) {
    float phi = 2.0 * PI * Xi.x;
    float cos_theta = sqrt(1.0 - Xi.y);
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    return float3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
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

    float3 irradiance = float3(0.0);

    float3 up = float3(0.0, 1.0, 0.0);
    float3 right = normalize(cross(up, N));
    up = cross(N, right);

    float sample_delta = 0.005;
    float samples = 0.0;

    for (float phi = 0.0; phi < 2.0 * PI; phi += sample_delta) {
        for (float theta = 0.0; theta < 0.5 * PI; theta += sample_delta) {
            float3 tangent_sample = float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
            float3 sample_vector = tangent_sample.x * right + tangent_sample.y * up + tangent_sample.z * N;
            irradiance += cubemap_texture.SampleLevel(sample_vector, 0).rgb * cos(theta) * sin(theta);
            samples++;
        }
    }

    irradiance = PI * irradiance / samples;

    output_image[thread_ID] = float4(irradiance, 1.0);
}