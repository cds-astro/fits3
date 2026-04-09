// shader.vert
#version 440
precision highp int;
precision highp float;

layout(location=0) in vec2 a_ndc;

layout(location=0) out vec2 ndc;

layout(set = 0, binding = 4)
uniform Scene {
    vec4 win_size;
    vec4 origin;
    vec4 perspective;
    float zoom_factor;
};

void main() {
    gl_Position = vec4(a_ndc.xy / zoom_factor, 0.0, 1.0);
    ndc = vec2(a_ndc.x, a_ndc.y * win_size.y / win_size.x) / zoom_factor;
}