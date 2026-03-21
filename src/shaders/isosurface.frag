// shader.frag
#version 440

layout(location=0) in vec2 ndc;
layout(location=0) out vec4 f_color;


layout(set = 0, binding = 0) uniform texture3D t_map;
layout(set = 0, binding = 1) uniform sampler s_map;
layout(set = 0, binding = 2) uniform texture3D td_map;
layout(set = 0, binding = 3) uniform sampler sd_map;
layout(set = 0, binding = 4)
uniform RotationMatrix {
    mat4 rot;
};
layout(set = 0, binding = 6)
uniform Time {
    vec4 time;
};
layout(set = 0, binding = 7)
uniform Origin {
    vec4 origin;
};
layout(set = 0, binding = 8)
uniform Cut {
    vec4 cut;
};
layout(set = 0, binding = 9)
uniform Perspective {
    vec4 perspective;
};
layout(set = 0, binding = 10)
uniform Isosurface {
    vec4 isosurface;
};
layout(set = 0, binding = 11)
uniform DiffuseColor {
    vec4 diffuse_color;
};
layout(set = 0, binding = 12)
uniform Size {
    vec4 cube_size;
};
layout(set = 0, binding = 13)
uniform Slices {
    vec2 sx;
    vec2 sy;
    vec2 sz;
    vec2 sw;
};
layout(set = 0, binding = 14)
uniform BlockSize {
    vec4 block_size;
};


vec3 lonlat2xyz(float lon, float lat) {
    float lat_s = sin(lat);
    float lat_c = cos(lat);
    float lon_s = sin(lon);
    float lon_c = cos(lon);

    return vec3(lat_c * lon_s, lat_s, lat_c * lon_c);
}
float colormap_red(float x) {
    if (x < 0.7) {
        return 4.0 * x - 1.5;
    } else {
        return -4.0 * x + 4.5;
    }
}

float colormap_green(float x) {
    if (x < 0.5) {
        return 4.0 * x - 0.5;
    } else {
        return -4.0 * x + 3.5;
    }
}

float colormap_blue(float x) {
    if (x < 0.3) {
       return 4.0 * x + 0.5;
    } else {
       return -4.0 * x + 2.5;
    }
}
/*
vec3 colormap_viridis(float t) {
    // Clamp input to [0,1]
    t = clamp(t, 0.0, 1.0);

    // Coefficients from the original viridis colormap (Matplotlib)
    const vec3 c0 = vec3(0.280, 0.165, 0.476);
    const vec3 c1 = vec3(0.110, 0.573, 0.664);
    const vec3 c2 = vec3(0.478, 0.821, 0.318);

    // Interpolation logic
    if (t < 0.5) {
        float f = smoothstep(0.0, 0.5, t);
        return mix(c0, c1, f);
    } else {
        float f = smoothstep(0.5, 1.0, t);
        return mix(c1, c2, f);
    }
}*/

vec3 colormap_viridis(float t) {
    vec3 c0 = vec3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
    vec3 c1 = vec3(0.1050930431085774, 1.404613529898575, 1.384590162594685);
    vec3 c2 = vec3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
    vec3 c3 = vec3(-4.634230498983486, -5.799100973351585, -19.33244095627987);
    vec3 c4 = vec3(6.228269936347081, 14.17993336680509, 56.69055260068105);
    vec3 c5 = vec3(4.776384997670288, -13.74514537774601, -65.35303263337234);
    vec3 c6 = vec3(-5.435455855934631, 4.645852612178535, 26.3124352495832);
    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));
}

vec3 colormap_turbo(in float x) {
    const vec4 kRedVec4 = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const vec4 kGreenVec4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    const vec4 kBlueVec4 = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const vec2 kRedVec2 = vec2(-152.94239396, 59.28637943);
    const vec2 kGreenVec2 = vec2(4.27729857, 2.82956604);
    const vec2 kBlueVec2 = vec2(-89.90310912, 27.34824973);
  
    x = clamp(x,0.0,1.0);
    vec4 v4 = vec4( 1.0, x, x * x, x * x * x);
    vec2 v2 = v4.zw * v4.z;
    return vec3(
        dot(v4, kRedVec4)   + dot(v2, kRedVec2),
        dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
        dot(v4, kBlueVec4)  + dot(v2, kBlueVec2)
    );
}

vec4 colormap(float x) {
    float r = clamp(colormap_red(x), 0.0, 1.0);
    float g = clamp(colormap_green(x), 0.0, 1.0);
    float b = clamp(colormap_blue(x), 0.0, 1.0);
    return vec4(r, g, b, 1.0);
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
    vec3 dv = 2.0 / cube_size.xyz;

    vec3 n = vec3(
        probe_cube(p - vec3(dv.x, 0.0, 0.0)) - probe_cube(p + vec3(dv.x, 0.0, 0.0)),
        probe_cube(p - vec3(0.0, dv.y, 0.0)) - probe_cube(p + vec3(0.0, dv.y, 0.0)),
        probe_cube(p - vec3(0.0, 0.0, dv.z)) - probe_cube(p + vec3(0.0, 0.0, dv.z))
    );

    return normalize(n);
}

vec3 grad4(vec3 p) {
    vec3 e = 1.0 / cube_size.xyz;

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
 // we define our cube as 2 bounds vertices, l and h
    vec3 l = max(vec3((sx.x / cube_size.x) - 0.5, (sy.x / cube_size.y) - 0.5, (sz.x / cube_size.z) - 0.5), vec3(-0.5));
    vec3 h = min(vec3((sx.y / cube_size.x) - 0.5, (sy.y / cube_size.y) - 0.5, (sz.y / cube_size.z) - 0.5), vec3(0.5));

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

    vec3 t_low = (l - p_cam) / r;
    vec3 t_high = (h - p_cam) / r;

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

    while (t < t_f && v < isosurface.x) {
        vec3 uv = (cell + 0.5) * coarse_inv;
        float max_v = probe_downsampled_cube(uv);

        if (max_v > isosurface.x) {
            float boundary = min(tMax.x, min(tMax.y, tMax.z));
            float limit = min(boundary, t_f);
 
            while(v < isosurface.x && t < limit) {
                vv = v;
                v = probe_cube(p * f);
                num_sampling += step;

                if (v > isosurface.x)
                    break;

                pp = p;
                p += dr;
                t += step;
            }

            if (v > isosurface.x)
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

    //vec3 ps = (pp + (p - pp) * (isosurface.x - vv) / (v - vv));

    //vec3 N = compute_normal(ps);
    //vec3 L = normalize(vec3(10.0, 10.0, 10.0) - ps);

    //float c = clamp((isosurface.x - cut.x) / (cut.y - cut.x), 0.0, 1.0);
    vec3 N = grad4(p * f);
    vec3 light_dir = normalize(-vec3(10.0, 10.0, 10.0));
    float diffuse = max(dot(N, light_dir), 0.0);
    vec4 color = vec4(diffuse_color.rgb * 0.05 + diffuse_color.rgb * diffuse, diffuse_color.a);
    //f_color = vec4(cc.rgb*0.05 + cc.rgb * max(dot(N, l), 0.0), 1.0);

    f_color = mix(vec4(0.0, 0.0, 0.0, 1.0), color, float(v > isosurface.x));
    //f_color = vec4(vec3(num_sampling / 1000.0), 1.0);
}
 