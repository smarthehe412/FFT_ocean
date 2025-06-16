#version 330

in vec3 normal_vector;
in vec3 light_vector;
in vec3 view_vector;
in vec2 tex_coord;
in float fog_factor;
in float vAlphaWeight;
in vec3 world_position;
in float steepness;
in vec3 vPosition;

uniform sampler2D u_foamTexture;
uniform sampler2D u_normalMap;
uniform float uTransparency;
uniform float u_time;
uniform float u_lightDistance;
uniform vec3 light_position;

out vec4 fragColor;

void main(void) {
    // 1. 基础参数
    vec3 lightColor = vec3(1.0, 0.95, 0.85);
    // 蓝绿色调
    vec3 waterColor = vec3(0.0, 0.15, 0.45);
    
    // 光照衰减
    // float attenuation = 1.0 / (1.0 + 0.02*u_lightDistance + 0.0002*u_lightDistance*u_lightDistance);
    
    // 2. 法线处理
    vec3 normal1 = normalize(normal_vector);
    
    // 3. 光照计算
    vec3 viewDir = normalize(view_vector);
    vec3 lightDir = normalize(light_vector);
    
    // 环境光
    vec3 ambient = lightColor * 0.8;
    
    // 漫反射
    float diff = max(dot(normal1, lightDir), 0.0);
    vec3 diffuse = diff * lightColor * 0.4;
    
    // 镜面反射
    vec3 reflectDir = reflect(-lightDir, normal1);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 128.0);
    vec3 specular = spec * lightColor * 0.8;
    
    // 基础颜色组合
    vec3 baseColor = waterColor * (ambient + diffuse) + specular;
    
    // 4. 次表面散射
    float thickness = 1.0 - dot(viewDir, normal1);
    float depth = -vPosition.z;
    vec3 sss = vec3(0.0, 0.15, 0.35) * thickness * exp(-depth * 0.1) * 0.2;
    baseColor += sss;
    
    // 5. 泡沫效果
    float foamIntensity = smoothstep(0.5, 0.8, -depth);
    if(foamIntensity > 0.8) {
        baseColor = mix(baseColor, vec3(0.9, 0.95, 1.0), foamIntensity * 0.04);
    }
    
    // 6. 透明度处理
    // float edgeAlpha = 1.0 - smoothstep(0.5, 0.8, abs(world_position.y) * 0.01);
    // float depthAlpha = 1.0 - exp(-depth * 0.01);
    // float finalAlpha = min(uTransparency, min(edgeAlpha, depthAlpha));
    // finalAlpha *= vAlphaWeight;
    
    // 7. 雾效混合
    vec3 fogColor = vec3(0.53, 0.81, 0.92); // 天蓝色雾
    vec3 finalColor = mix(baseColor, fogColor, fog_factor);
    
    // 8. 伽马校正
    finalColor = pow(finalColor, vec3(1.0/2.2));
    
    // 输出最终颜色
    fragColor = vec4(finalColor, uTransparency);
}