// shader.vert
#version 440
precision highp int;
precision highp float;

layout(location=0) in vec3 xyz;

layout(set = 0, binding = 0)
uniform RotationMatrix {
    mat4 rotmat;
};
layout(set = 0, binding = 1)
uniform Window {
    vec4 size;
};
layout(set = 0, binding = 2)
uniform Time {
    vec4 time;
};
layout(set = 0, binding = 3)
uniform Origin {
    vec4 origin;
};
layout(set = 0, binding = 4)
uniform Perspective {
    vec4 perspective;
};

vec3 lonlat2xyz(float lon, float lat) {
    float lat_s = sin(lat);
    float lat_c = cos(lat);
    float lon_s = sin(lon);
    float lon_c = cos(lon);

    return vec3(lat_c * lon_s, lat_s, lat_c * lon_c);
}

void main() {    
    vec3 cam_origin = lonlat2xyz(origin.x, origin.y);
     // vector from camera origin to the look
    vec3 cam_dir = normalize(-cam_origin);
    // origin of the screen in world space
    //float camera_near = 1.0f;
    vec3 o_cam = cam_origin;
    
    vec3 ox = normalize(vec3(cam_dir.z, 0.0, -cam_dir.x));
    vec3 oy = -cross(ox, cam_dir);

    float x = dot(xyz - o_cam, ox);
    float y = dot(xyz - o_cam, oy);

    gl_Position = vec4(
        x,
        y * (size.x / size.y),
        0.0,
        1.0
    );
}