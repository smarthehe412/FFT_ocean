#version 330 core
layout (location = 0) in vec3 vertex;

uniform mat4 Projection;
uniform mat4 View;
uniform mat4 Model;

void main() {
    gl_Position = Projection * View * Model * vec4(vertex, 1.0);
}