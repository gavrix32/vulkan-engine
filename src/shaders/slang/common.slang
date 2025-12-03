static const float PI = 3.14159265;
static const float EPSILON = 0.00001;
//static const float MAX_REFLECTION_LOD = 4.0;
static const uint MAX_MIP_LEVELS = 5;
static const float EXPOSURE = 1.0;
static const uint SAMPLE_COUNT = 4096;

float3 aces_film(float3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

float3 neutral(float3 color) {
  const float start_compression = 0.8 - 0.04;
  const float desaturation = 0.15;

  float x = min(color.r, min(color.g, color.b));
  float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
  color -= offset;

  float peak = max(color.r, max(color.g, color.b));
  if (peak < start_compression) return color;

  const float d = 1.0 - start_compression;
  float new_peak = 1.0 - d * d / (peak + d - start_compression);
  color *= new_peak / peak;

  float g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0);
  return lerp(color, float3(new_peak), g);
}

//float distribution_GGX(float3 N, float3 H, float roughness) {
//    float a = roughness * roughness;
//    float a2 = a * a;
//    float NdotH = max(dot(N, H), 0.0);
//    float NdotH2 = NdotH * NdotH;
//
//    float nom = a2;
//    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
//    denom = PI * denom * denom;
//
//    return nom / denom;
//}

// GGX/Trowbridge-Reitz Normal Distribution Function
float distribution_GGX(float3 N, float NdotH, float alpha) {
    float alpha2 = pow(alpha, 2.0);
    float NdotH2 = pow(NdotH, 2.0);

    float numerator = alpha2;
    float denominator = PI * pow(NdotH2 * (alpha2 - 1.0) + 1.0, 2.0);

    return numerator / max(denominator, EPSILON);
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

// Schlick-Beckmann Geometry Shadowing Function
float geometry_schlick_GGX(float NdotX, float alpha) {
    float k = alpha / 2.0;

    float numerator = NdotX;
    float denominator = NdotX * (1.0 - k) + k;

    return numerator / max(denominator, EPSILON);
}

float2 sample_env(float3 dir) {
    float2 uv;
    uv.x = atan2(dir.z, dir.x) / (2.0 * PI) + 0.5;
    uv.y = acos(dir.y) / PI;
    return uv;
}

// Smith Geometry Shadowing Function (light and view components)
float geometry_smith(float NdotL, float NdotV, float alpha) {
    return geometry_schlick_GGX(NdotL, alpha) * geometry_schlick_GGX(NdotV, alpha);
}