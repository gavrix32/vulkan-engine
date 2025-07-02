[shader("fragment")]
void main(
    out float4 out_color : SV_Target0,
    float3 frag_color : COLOR0,
) {
    out_color = float4(frag_color, 1.0);
}