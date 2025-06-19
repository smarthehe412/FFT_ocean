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

// 新增输入变量
in vec3 surface_to_light;
in vec3 surface_to_camera;
in vec3 refracted_light;
in float underwater_depth;

uniform sampler2D u_foamTexture;
uniform sampler2D u_normalMap;
uniform sampler2D u_causticTexture;  // 焦散纹理
uniform float uTransparency;
uniform float u_time;
uniform float u_lightDistance;
uniform vec3 light_position;
uniform float u_fogDensity;
uniform vec3 u_fogColor;

// 新增 uniform
uniform float u_causticIntensity;    // 焦散强度
uniform float u_godrayIntensity;     // 光柱强度
uniform float u_scatteringCoeff;     // 散射系数 (新增)

out vec4 fragColor;

void main(void) {
    // 1. 基础参数
    vec3 lightColor = vec3(1.0, 0.95, 0.85);
    vec3 waterColor = vec3(0.0, 0.15, 0.45);
    
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
    
    // ====================== 焦散效果 ======================
    // 计算焦散纹理坐标
    vec2 causticUV = world_position.xz * 0.01 + refracted_light.xz * underwater_depth * 0.05;
    causticUV += vec2(u_time * 0.005, u_time * 0.0025);
    
    // 采样焦散纹理
    float caustic = texture(u_causticTexture, causticUV).r;
    
    // 应用深度衰减 (深度越大焦散越明显)
    float depthAtten = 1.0 - exp(-underwater_depth * 0.1);
    caustic *= depthAtten;
    
    // 应用通量守恒 (聚焦区域更亮)
    float focus_factor = pow(1.0 - abs(dot(normal1, normalize(refracted_light))), 2.0);
    caustic *= mix(1.0, 1.5, focus_factor);
    
    // 添加到基础颜色
    baseColor = mix(baseColor, lightColor, caustic * u_causticIntensity * 0.5);
    
    // ====================== 光柱效果 ======================
    if (underwater_depth > 0.1) {  // 仅在水下生效
        // 计算到光柱中心线的距离
        vec3 toLight = normalize(surface_to_light);
        vec3 toCamera = normalize(surface_to_camera);
        
        // 计算散射角度
        float scatterAngle = acos(dot(toLight, toCamera));
        float scatterFactor = exp(-scatterAngle * scatterAngle * 50.0);
        
        // 计算体积散射 (基于深度和角度)
        float godray = scatterFactor * u_godrayIntensity;
        
        // 应用深度衰减和散射系数 (使用新增的uniform)
        godray *= exp(-underwater_depth * 0.5) * u_scatteringCoeff;
        
        // 添加到基础颜色
        // baseColor += caustic * u_causticIntensity * lightColor;
        baseColor += min(godray * lightColor * 0.2, vec3(0.3));
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