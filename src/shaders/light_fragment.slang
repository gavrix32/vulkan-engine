[shader("fragment")]
void main(
    out float4 out_color : SV_Target0,
    float4 out_light_pos,
) {
    out_color = float4(float3(1.0), 1.0);
}