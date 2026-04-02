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
};

layout(set = 0, binding = 5)
uniform Volume {
    vec3 cube_size;
    float _pad1;   // padding!
    vec3 block_size;
    float _pad2;   // padding!
};

layout(set = 0, binding = 6)
uniform RenderParams {
    vec3 cut_iso;
    int colormap;
    // x,y = cut
    // z = isosurface
    // w = colormap_selected (cast to float)
    vec4 diffuse_color;
    int transfer;
    int reversed;
};

layout(set = 0, binding = 7)
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
layout(set = 0, binding = 16)
uniform Colormap_selected {
    ivec4 colormap_selected;
};
layout(set = 0, binding = 12)
uniform Function_selected {
    ivec4 function_selected;
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

vec3 colormap_viridis2(float t) {

    const vec3 c0 = vec3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
    const vec3 c1 = vec3(0.1050930431085774, 1.404613529898575, 1.384590162594685);
    const vec3 c2 = vec3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
    const vec3 c3 = vec3(-4.634230498983486, -5.799100973351585, -19.33244095627987);
    const vec3 c4 = vec3(6.228269936347081, 14.17993336680509, 56.69055260068105);
    const vec3 c5 = vec3(4.776384997670288, -13.74514537774601, -65.35303263337234);
    const vec3 c6 = vec3(-5.435455855934631, 4.645852612178535, 26.3124352495832);

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

vec3 colormap_inferno(float t) {

    const vec3 c0 = vec3(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184);
    const vec3 c1 = vec3(0.1065134194856116, 0.5639564367884091, 3.932712388889277);
    const vec3 c2 = vec3(11.60249308247187, -3.972853965665698, -15.9423941062914);
    const vec3 c3 = vec3(-41.70399613139459, 17.43639888205313, 44.35414519872813);
    const vec3 c4 = vec3(77.162935699427, -33.40235894210092, -81.80730925738993);
    const vec3 c5 = vec3(-71.31942824499214, 32.62606426397723, 73.20951985803202);
    const vec3 c6 = vec3(25.13112622477341, -12.24266895238567, -23.07032500287172);

    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));

}

vec3 colormap_plasma(float t) {

    const vec3 c0 = vec3(0.05873234392399702, 0.02333670892565664, 0.5433401826748754);
    const vec3 c1 = vec3(2.176514634195958, 0.2383834171260182, 0.7539604599784036);
    const vec3 c2 = vec3(-2.689460476458034, -7.455851135738909, 3.110799939717086);
    const vec3 c3 = vec3(6.130348345893603, 42.3461881477227, -28.51885465332158);
    const vec3 c4 = vec3(-11.10743619062271, -82.66631109428045, 60.13984767418263);
    const vec3 c5 = vec3(10.02306557647065, 71.41361770095349, -54.07218655560067);
    const vec3 c6 = vec3(-3.658713842777788, -22.93153465461149, 18.19190778539828);

    return c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6)))));

}

vec3 colormap_rainbow(float t) {
    t = clamp(t, 0.0, 1.0);
    return clamp(vec3((0.521926+t*(-5.081361+t*(90.667146+t*(-1024.570176+t*(5466.954668+t*(-15142.356204+t*(22823.994490+t*(-17834.657381+t*5669.456533)))))))),
                      (0.001238+t*(3.087993+t*(0.533349+t*(-7.242684+t*3.621342)))),
                      (1.040332+t*(-0.285029+t*-0.775501))), 0.0, 1.0);
}

vec3 colormap_cubehelix(float t) {
    t = clamp(t, 0.0, 1.0);
    return clamp(vec3((-0.013249+t*(2.275258+t*(-8.817343+t*(-53.364075+t*(404.975951+t*(-860.459961+t*(757.174851+t*-240.797852))))))),
                      (-0.005678+t*(0.984536+t*(-8.239465+t*(106.331710+t*(-417.363212+t*(713.067352+t*(-558.604911+t*164.851278))))))),
                      (0.031276+t*(-0.925911+t*(49.839030+t*(-323.015906+t*(847.188116+t*(-1048.463499+t*(606.532787+t*-130.108084)))))))), 0.0, 1.0);
}

vec4 colormap(float x) {
    switch(colormap) {
        case 1:
            return vec4(colormap_viridis2(x),1.0);
        case 2:
            return vec4(colormap_inferno(x),1.0);
        case 3:
            return vec4(colormap_plasma(x),1.0);
        case 4:
            return vec4(colormap_rainbow(x),1.0);
        case 5:
            return vec4(colormap_cubehelix(x),1.0);
        default:
            float r = clamp(colormap_red(x), 0.0, 1.0);
            float g = clamp(colormap_green(x), 0.0, 1.0);
            float b = clamp(colormap_blue(x), 0.0, 1.0);
            return vec4(r, g, b, 1.0);
    }
}


float transfer(float x) {
    switch(transfer) {
        case 1:
            return sqrt(x);
        case 2:
            return pow(x,2);
        case 3:
            return asinh(10.0*x)/3.0;
        case 4:
            return log(1000.0*x + 1.0)/log(1000.0);
        default:
            return x;
    }
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
//const float dmin = -2.451346722E-03;
//const float dmax = 1.179221552E-02;

float probe_cube(vec3 p) {
    float v = to_l_endian(texture(sampler3D(t_map, s_map), p).r);
    return mix(v, -1e30, isnan(v));
}
float probe_downsampled_cube(vec3 p) {
    // no need to handle NaNs because it is assumed there are none by construction.
    return texture(sampler3D(td_map, sd_map), p).r;
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

    float intensity = cut_iso.x;

    vec3 step_dir = sign(r);
    vec3 cell = floor(p * coarse_size * f);
    vec3 next_boundary = (cell + max(step_dir, 0.0)) * coarse_inv;

    vec3 tMax = vec3(t) + (next_boundary - p * f) * inv_r;
    vec3 tDelta = coarse_inv * abs(inv_r);

    bvec3 zero_dir = lessThan(abs(r), vec3(1e-8));
    tDelta = mix(tDelta, vec3(1e30), zero_dir);
    tMax   = mix(tMax,   vec3(1e30), zero_dir);

    //int num_sampling = 0;

    while (t < t_f && intensity < cut_iso.y) {
        vec3 uv = (cell + 0.5) * coarse_inv;
        float max_v = probe_downsampled_cube(zoom_l + uv * (zoom_h - zoom_l));

        if (max_v > intensity) {
            float boundary = min(tMax.x, min(tMax.y, tMax.z));
            float limit = min(boundary, t_f);

            while(t < limit && intensity < cut_iso.y) {
                float v = probe_cube(zoom_l + p * f * (zoom_h - zoom_l));
                intensity = max(intensity, v);

                //num_sampling += 1;

                p += dr;
                t += step;
            }
        }

        float t_prev = t;

        bvec3 m = lessThanEqual(tMax, min(tMax.yzx, tMax.zxy));
        vec3 mask = vec3(m);

        t = dot(mask, tMax);

        cell += mask * step_dir;
        tMax += mask * tDelta;

        p += r * (t - t_prev);
    }

    intensity = clamp((intensity - cut_iso.x) / (cut_iso.y - cut_iso.x), 0.0, 1.0);
    if(reversed == 1) {
        intensity = 1 - intensity;
    }
    f_color = colormap(transfer(intensity));
}
 