[vk::binding(0, 0)] Sampler2D env_texture;

struct PushConstants {
    float2 resolution;
    float4x4 inv_view_proj;
};
[vk::push_constant] ConstantBuffer<PushConstants> push_constants;

static const float PI = 3.14159265;
static const float EPSILON = 0.00001;

struct VertexOutput {
    float4 sv_position : SV_Position;
    float2 clip_dir;
};

float2 sample_env(float3 dir) {
    float2 uv;
    uv.x = atan2(dir.z, dir.x) / (2.0 * PI) + 0.5;
    uv.y = acos(dir.y) / PI;
    return uv;
}

[shader("vertex")]
VertexOutput vertex_main(uint vertexID : SV_VertexID) {
    VertexOutput output;

    const float2 positions[3] = {
        float2(3.0, -1.0),
        float2(-1.0, -1.0),
        float2(-1.0, 3.0)
    };

    const float2 tex_coords[3] = {
       float2(2.0, 0.0),
       float2(0.0, 0.0),
       float2(0.0, 2.0)
    };

    float2 position = positions[vertexID];

    output.clip_dir = position;
    output.sv_position = float4(position, 0.0, 1.0);

    return output;
}

[shader("fragment")]
void fragment_main(
    VertexOutput input,
    out float4 out_color : SV_Target0
) {
    float2 uv = input.sv_position.xy / push_constants.resolution; // [0; 1]
    uv = uv * 2.0 - 1.0;                                          // [-1; 1]

    float4 world_dir = mul(push_constants.inv_view_proj, float4(uv, 1.0, 1.0));
    float3 dir = normalize(world_dir.xyz / world_dir.w);

    float2 sph_uv = sample_env(dir);
    float3 env = env_texture.Sample(sph_uv).rgb;

    float exposure = 1.0;
    env = float3(1.0) - exp(-env * exposure);

    out_color = float4(env, 1.0);
}