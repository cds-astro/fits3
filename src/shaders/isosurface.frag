// shader.frag
#version 440

layout(location=0) in vec2 ndc;
layout(location=0) out vec4 f_color;


layout(set = 0, binding = 0) uniform texture3D t_map;
layout(set = 0, binding = 1) uniform sampler s_map;
layout(set = 0, binding = 2) uniform texture3D td_map;
layout(set = 0, binding = 3) uniform sampler sd_map;

layout(set = 0, binding = 4)
uniform Scene {
    vec4 win_size;
    vec4 origin;
    vec4 perspective;
    float zoom_factor;
};

layout(set = 0, binding = 5)
uniform Volume {
    vec3 cube_size;
    float _pad1;   // padding!
    vec3 block_size;
    float _pad2;   // padding!
};

layout(set = 0, binding = 6)
uniform VolumetricRenderParams {
    vec2 cut;
    int transfer;
    int colormap;
    vec3 bg_color;
};

layout(set = 0, binding = 7)
uniform SurfaceRenderParams {
    vec4 diffuse_color;
    float isosurface;
};

layout(set = 0, binding = 8)
uniform Interaction {
    vec3 zoom_l;
    float _pad3; // padding!

    vec3 zoom_h;
    float _pad4; // padding!

    vec3 bbox_min;
    float _pad5;   // padding!

    vec3 bbox_max;
    float _pad6;   // padding!
};

vec3 lonlat2xyz(float lon, float lat) {
    float lat_s = sin(lat);
    float lat_c = cos(lat);
    float lon_s = sin(lon);
    float lon_c = cos(lon);

    return vec3(lat_c * lon_s, lat_s, lat_c * lon_c);
}

float to_l_endian(float x) {
    uint y = floatBitsToUint(x);

    uint a = y & 0xff;
    uint b = (y >> 8) & 0xff;
    uint c = (y >> 16) & 0xff;
    uint d = y >> 24;

    uint w = (a << 24) | (b << 16) | (c << 8) | d;

    return uintBitsToFloat(w);
}

// Parameters:
//   x - input intensity (usually normalized to [0,1])
//   scale - scaling factor to control the stretch strength
//   nonlinearity - controls how nonlinear the stretch is (typically > 0)
//
// Returns:
//   A value between 0 and 1 after applying the asinh stretch
float asinhStretch(float x, float scale, float nonlinearity) {
    return asinh(scale * x) / asinh(scale * nonlinearity);
}

bool is_finite_f32(float x) {
    return abs(x) <= 3.402823e38;
}

const float fov = 0.523333;
const float camera_near = 1.0;

float probe_cube(vec3 p) {
    float v = to_l_endian(texture(sampler3D(t_map, s_map), p).r);
    return mix(v, -1e30, isnan(v));
}
float probe_downsampled_cube(vec3 p) {
    // no need to handle NaNs because it is assumed there are none by construction.
    return texture(sampler3D(td_map, sd_map), p).r;
}

vec3 compute_normal(vec3 p) {
    vec3 dv = 2.0 / cube_size;

    vec3 n = vec3(
        probe_cube(p - vec3(dv.x, 0.0, 0.0)) - probe_cube(p + vec3(dv.x, 0.0, 0.0)),
        probe_cube(p - vec3(0.0, dv.y, 0.0)) - probe_cube(p + vec3(0.0, dv.y, 0.0)),
        probe_cube(p - vec3(0.0, 0.0, dv.z)) - probe_cube(p + vec3(0.0, 0.0, dv.z))
    );

    return normalize(n);
}

vec3 grad4(vec3 p) {
    vec3 e = 1.0 / cube_size;

    vec3 k1 = vec3( 1, -1, -1);
    vec3 k2 = vec3(-1, -1,  1);
    vec3 k3 = vec3(-1,  1, -1);
    vec3 k4 = vec3( 1,  1,  1);

    return normalize(
        k1 * probe_cube(p + k1 * e) +
        k2 * probe_cube(p + k2 * e) +
        k3 * probe_cube(p + k3 * e) +
        k4 * probe_cube(p + k4 * e)
    );
}

void main() {
    vec3 cam_origin = lonlat2xyz(origin.x, origin.y);

    // vector from camera origin to the look
    vec3 cam_dir = -cam_origin;
    // origin of the screen in world space
    vec3 o_cam = cam_origin + cam_dir * camera_near;

    // find a vector belonging to the plane of screen oriented with y
    vec3 ox = normalize(vec3(cam_dir.z, 0.0, -cam_dir.x));
    vec3 oy = -cross(ox, cam_dir);

    vec3 p_cam = o_cam + ox * ndc.x + oy * ndc.y;

    // vector director from the cam origin to the pixel on screen
    // traditional perspective director vector
    // orthographic perspective
    vec3 r = mix(normalize(p_cam - cam_origin), cam_dir, float(perspective.x == 0.0));

    vec3 t_low = (bbox_min - p_cam) / r;
    vec3 t_high = (bbox_max - p_cam) / r;

    vec3 t_close = min(t_low, t_high);
    vec3 t_far = max(t_low, t_high);

    float t_c = max(t_close.x, max(t_close.y, t_close.z));
    float t_f = min(t_far.x, min(t_far.y, t_far.z));

    if (t_f < t_c) {
        discard;
    }

    vec3 abs_r = abs(r);
    vec3 inv_r = 1.0 / r;
    vec3 padding = mod(block_size.xyz - mod(cube_size.xyz, block_size.xyz), block_size.xyz);

    vec3 f = (cube_size.xyz) / (cube_size.xyz + padding.xyz);

    vec3 inv_dir = abs_r * cube_size.xyz;
    float step = 1.0 / max(max(inv_dir.x, inv_dir.y), inv_dir.z);
    
    vec3 coarse_size = (cube_size.xyz + padding.xyz) / block_size.xyz;
    vec3 coarse_inv = 1.0 / coarse_size;

    vec3 dr = r * step;
    float random = fract(dot(gl_FragCoord.xy, vec2(0.75487766, 0.56984029)));
    // absolute sampling point
    // scaled to the origin of the cube
    float t = t_c + step * random;
    // p in [0; 1]
    vec3 p = p_cam + r * t + vec3(0.5);

    float intensity = cut.x;

    vec3 step_dir = sign(r);
    vec3 cell = floor(p * coarse_size * f);
    vec3 next_boundary = (cell + max(step_dir, 0.0)) * coarse_inv;

    vec3 tMax = vec3(t) + (next_boundary - p * f) * inv_r;
    vec3 tDelta = coarse_inv * abs(inv_r);

    bvec3 zero_dir = lessThan(abs(r), vec3(1e-8));
    tDelta = mix(tDelta, vec3(1e30), zero_dir);
    tMax   = mix(tMax,   vec3(1e30), zero_dir);

    float num_sampling = 0.0;

    vec3 pp = p;

    float v = -1e30;
    float vv = v;

    while (t < t_f && v < isosurface) {
        vec3 uv = (cell + 0.5) * coarse_inv;
        float max_v = probe_downsampled_cube(zoom_l + uv * (zoom_h - zoom_l));

        if (max_v > isosurface) {
            float boundary = min(tMax.x, min(tMax.y, tMax.z));
            float limit = min(boundary, t_f);
 
            while(v < isosurface && t < limit) {
                vv = v;
                v = probe_cube(zoom_l + p * f * (zoom_h - zoom_l));
                num_sampling += step;

                if (v > isosurface)
                    break;

                pp = p;
                p += dr;
                t += step;
            }

            if (v > isosurface)
                break;
        }

        float t_prev = t;

        bvec3 m = lessThanEqual(tMax, min(tMax.yzx, tMax.zxy));
        vec3 mask = vec3(m);

        t = dot(mask, tMax);

        cell += mask * step_dir;
        tMax += mask * tDelta;

        pp = p;
        p += r * (t - t_prev);
        num_sampling += t - t_prev;
    }

    vec3 N = grad4(zoom_l + p * f * (zoom_h - zoom_l));
    vec3 light_dir = -vec3(0.577350269, 0.577350269, 0.577350269);
    float diffuse = max(dot(N, light_dir), 0.0);
    vec4 color = vec4(diffuse_color.rgb * 0.05 + diffuse_color.rgb * diffuse, diffuse_color.a);

    f_color = mix(vec4(0.0, 0.0, 0.0, 1.0), color, float(v > isosurface));
}
 