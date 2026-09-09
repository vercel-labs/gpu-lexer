Shader "Custom/Example" {
  Properties { _Color ("Color", Color) = (1, 1, 1, 1) }
  SubShader { Pass {
    HLSLPROGRAM
    #pragma vertex vert
    // embedded comment
    float4 vert(float4 position : POSITION) : SV_POSITION {
      return position * 2.0;
    }
    ENDHLSL
  } }
}
