// shader.vert
#version 440
precision highp int;
precision highp float;

layout(location=0) in vec2 ndc;
layout(location=0) out vec2 pos;

layout(set = 0, binding = 2)
uniform Scene {
    vec4 win_size;
    vec4 origin;
    vec4 perspective;
    float zoom_factor;
};

void main() {
    pos = ndc;
    gl_Position = vec4(
        pos / zoom_factor,
        0.0,
        1.0
    );
}