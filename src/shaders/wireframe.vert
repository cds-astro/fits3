// shader.vert
#version 440
precision highp int;
precision highp float;

layout(location=0) in vec3 xyz;

layout(set = 0, binding = 0)
uniform Scene {
    vec4 win_size;
    vec4 origin;
    vec4 perspective;
};

vec3 lonlat2xyz(float lon, float lat) {
    float lat_s = sin(lat);
    float lat_c = cos(lat);
    float lon_s = sin(lon);
    float lon_c = cos(lon);

    return vec3(lat_c * lon_s, lat_s, lat_c * lon_c);
}

const float epsilon = 0.0001;
const float f = 1.0; // focal length (controls FOV)

void main() {
    vec3 cam_origin = lonlat2xyz(origin.x, origin.y);
    // vector from camera origin to the look
    vec3 cam_dir = normalize(-cam_origin);

    // origin of the screen in world space
    //float camera_near = 1.0f;
    vec3 o_cam = cam_origin;
    
    vec3 ox = normalize(vec3(cam_dir.z, 0.0, -cam_dir.x));
    vec3 oy = -cross(ox, cam_dir);

    vec3 p_rel = xyz - o_cam;

    float x = dot(p_rel, ox);
    float y = dot(p_rel, oy);
    float z = dot(p_rel, cam_dir);

    // Avoid division by zero or behind-camera artifacts
    z = max(z, epsilon);

    vec2 pp = vec2(x, y * (win_size.x / win_size.y));

    vec2 pos = mix(pp, pp * f / z, float(perspective.x == 1.0));

    gl_Position = vec4(
        pos,
        0.0,
        1.0
    );
}