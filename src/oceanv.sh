#version 330

in vec3 vertex;
in vec3 normal;
in vec3 texture;

uniform mat4 Projection;
uniform mat4 View;
uniform mat4 Model;
uniform vec3 light_position;
uniform float u_time;
uniform vec3 camera_position;  // 添加相机位置 uniform

out vec3 light_vector;
out vec3 normal_vector;
out vec3 view_vector;
out float fog_factor;
out vec2 tex_coord;
out vec3 world_position;
out float steepness;

in float aAlphaWeight;
out float vAlphaWeight;
out vec3 vPosition;

out vec3 surface_to_light;  // 水面到光源的向量
out vec3 surface_to_camera; // 水面到相机的向量
out vec3 refracted_light;   // 折射光线方向
out float underwater_depth; // 水下深度

void main() {
    // 位置计算
    vec4 worldPos = Model * vec4(vertex, 1.0);
    vec4 viewPos = View * worldPos;
    gl_Position = Projection * viewPos;
    
    world_position = worldPos.xyz;
    vPosition = viewPos.xyz;
    
    // 雾效因子
    fog_factor = clamp(-viewPos.z/5000.0, 0.0, 1.0);
    
    // 法线转换
    mat3 normalMatrix = mat3(transpose(inverse(View * Model)));
    vec3 normal1 = normalize(normalMatrix * normal);
    
    // 光照向量
    light_vector = normalize(light_position - worldPos.xyz); // 使用世界空间
    view_vector = normalize(-viewPos.xyz); // 视图方向
    normal_vector = normal1;
    
    // 纹理坐标
    tex_coord = texture.xy + vec2(u_time * 0.01, u_time * 0.005);
    
    // 透明度权重
    vAlphaWeight = aAlphaWeight;
    
    // 计算波浪陡峭度
    steepness = length(normal1.xz) * 3.0;

    // 计算水面到光源和相机的向量
    surface_to_light = light_position - worldPos.xyz;
    surface_to_camera = camera_position - worldPos.xyz;  // 使用声明的 uniform
    
    // 计算折射光线方向
    float air_ior = 1.0;      // 空气折射率
    float water_ior = 1.33;   // 水折射率
    vec3 incident = normalize(worldPos.xyz - camera_position);  // 使用声明的 uniform
    vec3 norm = normalize(normal1);
    refracted_light = refract(incident, norm, air_ior/water_ior);
    
    // 计算水下深度 (假设水面在 y=0)
    underwater_depth = clamp(-world_position.y, 0.0, 100.0);
}