// shader.frag
#version 440

layout(set = 0, binding = 0) uniform texture2D t_map;
layout(set = 0, binding = 1) uniform sampler s_map;

layout(location=0) in vec2 pos;

layout(location=0) out vec4 f_color;

void main() {
    vec2 ndc = vec2(pos.x, -pos.y);
    f_color = vec4(texture(sampler2D(t_map, s_map), 0.5*ndc + 0.5).rgb, 1.0);
    //f_color = vec4(1.0, 0.0, 1.0, 1.0); // bright pink
}
 