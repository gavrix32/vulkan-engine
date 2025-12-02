struct UniformBuffer {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    float4 light_pos;
    float4 cam_pos;
};

[vk::binding(0, 0)]
ConstantBuffer<UniformBuffer> uniforms;

static const float PI = 3.14159265;
static const float EPSILON = 0.00001;
static const float MAX_REFLECTION_LOD = 4.0;
static const float EXPOSURE = 0.5;

float3 aces_film(float3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}