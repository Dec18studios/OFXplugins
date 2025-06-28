#import <Metal/Metal.h>
#include "OpenDRTParams.h"  // Include the shared header

#include <unordered_map>
#include <mutex>

// Constants and matrices at the top of the file
constexpr float SQRT3 = 1.73205080756887729353f;
constexpr float PI = 3.14159265358979323846f;

// Matrix structure for 3x3 matrices
struct float3x3 {
    float3 x, y, z;
};

// Helper function to create a float3x3
inline float3x3 make_float3x3(float3 a, float3 b, float3 c) {
    float3x3 d;
    d.x = a; d.y = b; d.z = c;
    return d;
}

// Matrix multiplication
inline float3 vdot(float3x3 m, float3 v) {
    return make_float3(
        m.x.x*v.x + m.x.y*v.y + m.x.z*v.z,
        m.y.x*v.x + m.y.y*v.y + m.y.z*v.z,
        m.z.x*v.x + m.z.y*v.y + m.z.z*v.z
    );
}

// Identity matrix
inline float3x3 identity() {
    return make_float3x3(
        make_float3(1.0f, 0.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f),
        make_float3(0.0f, 0.0f, 1.0f)
    );
}

// Gamut conversion matrices (from DCTL)
static const float3x3 matrix_ap0_to_xyz = {
    {0.93863094875f, -0.00574192055f, 0.017566898852f},
    {0.338093594922f, 0.727213902811f, -0.065307497733f},
    {0.000723121511f, 0.000818441849f, 1.0875161874f}
};

static const float3x3 matrix_ap1_to_xyz = {
    {0.652418717672f, 0.127179925538f, 0.170857283842f},
    {0.268064059194f, 0.672464478993f, 0.059471461813f},
    {-0.00546992851f, 0.005182799977f, 1.08934487929f}
};

static const float3x3 matrix_rec709_to_xyz = {
    {0.412390917540f, 0.357584357262f, 0.180480793118f},
    {0.212639078498f, 0.715168714523f, 0.072192311287f},
    {0.019330825657f, 0.119194783270f, 0.950532138348f}
};

static const float3x3 matrix_p3d65_to_xyz = {
    {0.486571133137f, 0.265667706728f, 0.198217317462f},
    {0.228974640369f, 0.691738605499f, 0.079286918044f},
    {0.0f, 0.045113388449f, 1.043944478035f}
};

static const float3x3 matrix_xyz_to_p3d65 = {
    {2.49349691194f, -0.931383617919f, -0.402710784451f},
    {-0.829488969562f, 1.76266406032f, 0.023624685842f},
    {0.035845830244f, -0.076172389268f, 0.956884524008f}
};

static const float3x3 matrix_rec2020_to_xyz = {
    {0.636958122253f, 0.144616916776f, 0.168880969286f},
    {0.262700229883f, 0.677998125553f, 0.059301715344f},
    {0.0f, 0.028072696179f, 1.060985088348f}
};

// Display gamut matrices
static const float3x3 matrix_p3_to_rec709_d65 = {
    {1.224940181f, -0.2249402404f, 0.0f},
    {-0.04205697775f, 1.042057037f, -1.4901e-08f},
    {-0.01963755488f, -0.07863604277f, 1.098273635f}
};

static const float3x3 matrix_p3_to_rec709_d60 = {
    {1.189986856f, -0.192168414f, 0.002185496045f},
    {-0.04168263635f, 0.9927757018f, -5.5660878e-05f},
    {-0.01937995127f, -0.07933006919f, 0.9734397041f}
};

static const float3x3 matrix_p3_to_rec709_d55 = {
    {1.149327514f, -0.1536910745f, 0.004366526746f},
    {-0.0412590771f, 0.9351717477f, -0.000116126221f},
    {-0.01900949528f, -0.07928282823f, 0.8437884317f}
};

static const float3x3 matrix_p3_to_rec709_d50 = {
    {1.103807322f, -0.1103425121f, 0.006531676079f},
    {-0.04079386701f, 0.8704694227f, -0.000180522628f},
    {-0.01854055914f, -0.07857582481f, 0.7105498861f}
};

static const float3x3 matrix_p3_to_p3_d60 = {
    {0.979832881f, 0.01836378979f, 0.001803284786f},
    {-0.000805359793f, 0.9618000331f, 1.8876121e-05f},
    {-0.000338382322f, -0.003671835795f, 0.894139105f}
};

static const float3x3 matrix_p3_to_p3_d55 = {
    {0.9559790976f, 0.0403850003f, 0.003639287409f},
    {-0.001771929896f, 0.9163058305f, 3.3300759e-05f},
    {-0.000674760809f, -0.0072466358f, 0.7831189153f}
};

static const float3x3 matrix_p3_to_p3_d50 = {
    {0.9287127388f, 0.06578032793f, 0.005506708345f},
    {-0.002887159176f, 0.8640709228f, 4.3593718e-05f},
    {-0.001009551548f, -0.01073503317f, 0.6672692039f}
};

static const float3x3 matrix_p3_to_rec2020 = {
    {0.7538330344f, 0.1985973691f, 0.04756959659f},
    {0.04574384897f, 0.9417772198f, 0.01247893122f},
    {-0.001210340355f, 0.0176017173f, 0.9836086231f}
};

// Math helper functions
inline float sdivf(float a, float b) {
    return (b == 0.0f) ? 0.0f : a/b;
}

inline float3 sdivf3f(float3 a, float b) {
    return make_float3(sdivf(a.x, b), sdivf(a.y, b), sdivf(a.z, b));
}

inline float3 sdivf3f3(float3 a, float3 b) {
    return make_float3(sdivf(a.x, b.x), sdivf(a.y, b.y), sdivf(a.z, b.z));
}

inline float spowf(float a, float b) {
    return (a <= 0.0f) ? a : pow(a, b);
}

inline float3 spowf3(float3 a, float b) {
    return make_float3(spowf(a.x, b), spowf(a.y, b), spowf(a.z, b));
}

inline float hypotf3(float3 a) {
    return sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
}

inline float fmaxf3(float3 a) {
    return max(a.x, max(a.y, a.z));
}

inline float fminf3(float3 a) {
    return min(a.x, min(a.y, a.z));
}

inline float3 clampmaxf3(float3 a, float mx) {
    return make_float3(min(a.x, mx), min(a.y, mx), min(a.z, mx));
}

inline float3 clampminf3(float3 a, float mn) {
    return make_float3(max(a.x, mn), max(a.y, mn), max(a.z, mn));
}

inline float clampf(float a, float mn, float mx) {
    return min(max(a, mn), mx);
}

inline float3 clampf3(float3 a, float mn, float mx) {
    return make_float3(clampf(a.x, mn, mx), clampf(a.y, mn, mx), clampf(a.z, mn, mx));
}

// OETF linearization functions
inline float oetf_davinci_intermediate(float x) {
    return x <= 0.02740668f ? x/10.44426855f : exp2(x/0.07329248f - 7.0f) - 0.0075f;
}

inline float oetf_filmlight_tlog(float x) {
    return x < 0.075f ? (x-0.075f)/16.184376489665897f : exp((x - 0.5520126568606655f)/0.09232902596577353f) - 0.0057048244042473785f;
}

inline float oetf_acescct(float x) {
    return x <= 0.155251141552511f ? (x - 0.0729055341958355f)/10.5402377416545f : exp2(x*17.52f - 9.72f);
}

inline float3 linearize(float3 rgb, int tf) {
    if (tf == 0) { // Linear
        return rgb;
    } else if (tf == 1) { // Davinci Intermediate
        rgb.x = oetf_davinci_intermediate(rgb.x);
        rgb.y = oetf_davinci_intermediate(rgb.y);
        rgb.z = oetf_davinci_intermediate(rgb.z);
    } else if (tf == 2) { // Filmlight T-Log
        rgb.x = oetf_filmlight_tlog(rgb.x);
        rgb.y = oetf_filmlight_tlog(rgb.y);
        rgb.z = oetf_filmlight_tlog(rgb.z);
    } else if (tf == 3) { // ACEScct
        rgb.x = oetf_acescct(rgb.x);
        rgb.y = oetf_acescct(rgb.y);
        rgb.z = oetf_acescct(rgb.z);
    }
    return rgb;
}

// EOTF functions
inline float3 eotf_pq(float3 rgb, int inverse) {
    const float m1 = 2610.0f/16384.0f;
    const float m2 = 2523.0f/32.0f;
    const float c1 = 107.0f/128.0f;
    const float c2 = 2413.0f/128.0f;
    const float c3 = 2392.0f/128.0f;

    if (inverse == 1) {
        rgb = spowf3(rgb, m1);
        rgb = spowf3((c1 + c2*rgb)/(1.0f + c3*rgb), m2);
    } else {
        rgb = spowf3(rgb, 1.0f/m2);
        rgb = spowf3((rgb - c1)/(c2 - c3*rgb), 1.0f/m1);
    }
    return rgb;
}

inline float3 eotf_hlg(float3 rgb, int inverse) {
    if (inverse == 1) {
        float Yd = 0.2627f*rgb.x + 0.6780f*rgb.y + 0.0593f*rgb.z;
        rgb = rgb*spowf(Yd, (1.0f - 1.2f)/1.2f);
        rgb.x = rgb.x <= 1.0f/12.0f ? sqrt(3.0f*rgb.x) : 0.17883277f*log(12.0f*rgb.x - 0.28466892f) + 0.55991073f;
        rgb.y = rgb.y <= 1.0f/12.0f ? sqrt(3.0f*rgb.y) : 0.17883277f*log(12.0f*rgb.y - 0.28466892f) + 0.55991073f;
        rgb.z = rgb.z <= 1.0f/12.0f ? sqrt(3.0f*rgb.z) : 0.17883277f*log(12.0f*rgb.z - 0.28466892f) + 0.55991073f;
    } else {
        rgb.x = rgb.x <= 0.5f ? rgb.x*rgb.x/3.0f : (exp((rgb.x - 0.55991073f)/0.17883277f) + 0.28466892f)/12.0f;
        rgb.y = rgb.y <= 0.5f ? rgb.y*rgb.y/3.0f : (exp((rgb.y - 0.55991073f)/0.17883277f) + 0.28466892f)/12.0f;
        rgb.z = rgb.z <= 0.5f ? rgb.z*rgb.z/3.0f : (exp((rgb.z - 0.55991073f)/0.17883277f) + 0.28466892f)/12.0f;
        float Ys = 0.2627f*rgb.x + 0.6780f*rgb.y + 0.0593f*rgb.z;
        rgb = rgb*spowf(Ys, 1.2f - 1.0f);
    }
    return rgb;
}

// OpenDRT core functions
inline float compress_hyperbolic_power(float x, float s, float p) {
    return spowf(x/(x + s), p);
}

inline float compress_toe_quadratic(float x, float toe, int inv) {
    if (toe == 0.0f) return x;
    if (inv == 0) {
        return spowf(x, 2.0f)/(x + toe);
    } else {
        return (x + sqrt(x*(4.0f*toe + x)))/2.0f;
    }
}

inline float compress_toe_cubic(float x, float m, float w, int inv) {
    if (m == 1.0f) return x;
    float x2 = x*x;
    if (inv == 0) {
        return x*(x2 + m*w)/(x2 + w);
    } else {
        float p0 = x2 - 3.0f*m*w;
        float p1 = 2.0f*x2 + 27.0f*w - 9.0f*m*w;
        float p2 = pow(sqrt(x2*p1*p1 - 4*p0*p0*p0)/2.0f + x*p1/2.0f, 1.0f/3.0f);
        return p0/(3.0f*p2) + p2/3.0f + x/3.0f;
    }
}

inline float complement_power(float x, float p) {
    return 1.0f - spowf(1.0f - x, 1.0f/p);
}

inline float sigmoid_cubic(float x, float s) {
    if (x < 0.0f || x > 1.0f) return 1.0f;
    return 1.0f + s*(1.0f - 3.0f*x*x + 2.0f*x*x*x);
}

inline float contrast_high(float x, float p, float pv, float pv_lx, int inv) {
    const float x0 = 0.18f*pow(2.0f, pv);
    if (x < x0 || p == 1.0f) return x;

    const float o = x0 - x0/p;
    const float s0 = pow(x0, 1.0f - p)/p;
    const float x1 = x0*pow(2.0f, pv_lx);
    const float k1 = p*s0*pow(x1, p)/x1;
    const float y1 = s0*pow(x1, p) + o;
    
    if (inv == 1)
        return x > y1 ? (x - y1)/k1 + x1 : pow((x - o)/s0, 1.0f/p);
    else
        return x > x1 ? k1*(x - x1) + y1 : s0*pow(x, p) + o;
}

inline float softplus(float x, float s, float x0, float y0) {
    if (x > 10.0f*s + y0 || s < 1e-3f) return x;
    float m = 1.0f;
    if (abs(y0) > 1e-6f) m = exp(y0/s);
    m -= exp(x0/s);
    return s*log(max(0.0f, m + exp(x/s)));
}

inline float gauss_window(float x, float w) {
    x /= w;
    return exp(-x*x);
}

inline float hue_offset(float h, float o) {
    return fmod(h - o + PI, 2.0f*PI) - PI;
}

// Main OpenDRT function
float4 applyOpenDRT(OpenDRTParams params, float4 inPixel) {
    float3 rgb = make_float3(inPixel.x, inPixel.y, inPixel.z);

    // Get input gamut matrix
    float3x3 in_to_xyz;
    if (params.inGamut == 0) in_to_xyz = identity();
    else if (params.inGamut == 1) in_to_xyz = matrix_ap0_to_xyz;
    else if (params.inGamut == 2) in_to_xyz = matrix_ap1_to_xyz;
    else if (params.inGamut == 3) in_to_xyz = matrix_p3d65_to_xyz;
    else if (params.inGamut == 4) in_to_xyz = matrix_rec2020_to_xyz;
    else if (params.inGamut == 5) in_to_xyz = matrix_rec709_to_xyz;
    else in_to_xyz = identity();

    // Linearize if needed
    rgb = linearize(rgb, params.inOetf);

    // Tonescale constraint calculations
    const float ts_x1 = pow(2.0f, 6.0f*params.tnSh + 4.0f);
    const float ts_y1 = params.tnLp/100.0f;
    const float ts_x0 = 0.18f + params.tnOff;
    const float ts_y0 = params.tnLg/100.0f*(1.0f + params.tnGb*log2(ts_y1));
    const float ts_s0 = compress_toe_quadratic(ts_y0, params.tnToe, 1);
    const float ts_s10 = ts_x0*(pow(ts_s0, -1.0f/params.tnCon) - 1.0f);
    const float ts_m1 = ts_y1/pow(ts_x1/(ts_x1 + ts_s10), params.tnCon);
    const float ts_m2 = compress_toe_quadratic(ts_m1, params.tnToe, 1);
    const float ts_s = ts_x0*(pow(ts_s0/ts_m2, -1.0f/params.tnCon) - 1.0f);
    const float ts_dsc = (params.eotf == 4) ? 0.01f : (params.eotf == 5) ? 0.1f : 100.0f/params.tnLp;

    // HDR purity compensation
    const float pt_cmp_Lf = params.ptHdr*min(1.0f, (params.tnLp - 100.0f)/900.0f);
    const float s_Lp100 = ts_x0*(pow((params.tnLg/100.0f), -1.0f/params.tnCon) - 1.0f);
    const float ts_s1 = ts_s*pt_cmp_Lf + s_Lp100*(1.0f - pt_cmp_Lf);

    // Convert from input gamut to P3-D65
    rgb = vdot(in_to_xyz, rgb);
    rgb = vdot(matrix_xyz_to_p3d65, rgb);

    // Rendering space desaturation
    float3 rs_w = make_float3(params.rsRw, 1.0f - params.rsRw - params.rsBw, params.rsBw);
    float sat_L = rgb.x*rs_w.x + rgb.y*rs_w.y + rgb.z*rs_w.z;
    rgb = sat_L*params.rsSa + rgb*(1.0f - params.rsSa);

    // Offset
    rgb += params.tnOff;

    // Contrast Low Module
    if (params.tnLcon > 0.0f) {
        float mcon_m = pow(2.0f, -params.tnLcon);
        float mcon_w = params.tnLconW/4.0f;
        mcon_w *= mcon_w;

        // Normalize for ts_x0 intersection constraint
        const float mcon_cnst_sc = compress_toe_cubic(ts_x0, mcon_m, mcon_w, 1)/ts_x0;
        rgb *= mcon_cnst_sc;

        // Scale for ratio-preserving midtone contrast
        float mcon_nm = hypotf3(clampminf3(rgb, 0.0f))/SQRT3;
        float mcon_sc = (mcon_nm*mcon_nm + mcon_m*mcon_w)/(mcon_nm*mcon_nm + mcon_w);

        if (params.tnLconPc > 0.0f) {
            // Apply per-channel midtone contrast
            float3 mcon_rgb = rgb;
            mcon_rgb.x = compress_toe_cubic(rgb.x, mcon_m, mcon_w, 0);
            mcon_rgb.y = compress_toe_cubic(rgb.y, mcon_m, mcon_w, 0);
            mcon_rgb.z = compress_toe_cubic(rgb.z, mcon_m, mcon_w, 0);

            // Mix based on distance from achromatic
            float mcon_mx = fmaxf3(rgb);
            float mcon_mn = fminf3(rgb);
            float mcon_ch = clampf(1.0f - sdivf(mcon_mn, mcon_mx), 0.0f, 1.0f);
            mcon_ch = pow(mcon_ch, 4.0f*params.tnLconPc);
            rgb = mcon_sc*rgb*mcon_ch + mcon_rgb*(1.0f - mcon_ch);
        } else {
            rgb = mcon_sc*rgb;
        }
    }

    // Tonescale and purity norms
    float tsn = hypotf3(clampminf3(rgb, 0.0f))/SQRT3;
    float ts_pt = sqrt(max(0.0f, rgb.x*rgb.x*params.ptR + rgb.y*rgb.y*params.ptG + rgb.z*rgb.z*params.ptB));

    // RGB ratios
    rgb = sdivf3f(clampminf3(rgb, -2.0f), tsn);

    // Apply high contrast
    if (params.tnHcon != 0.0f) {
        float hcon_p = pow(2.0f, params.tnHcon);
        tsn = contrast_high(tsn, hcon_p, params.tnHconPv, params.tnHconSt, 0);
        ts_pt = contrast_high(ts_pt, hcon_p, params.tnHconPv, params.tnHconSt, 0);
    }

    // Apply tonescale
    tsn = compress_hyperbolic_power(tsn, ts_s, params.tnCon);
    ts_pt = compress_hyperbolic_power(ts_pt, ts_s1, params.tnCon);

    // Simple Cyan-Yellow / Green-Magenta opponent space
    float opp_cy = rgb.x - rgb.z;
    float opp_gm = rgb.y - (rgb.x + rgb.z)/2.0f;
    float ach_d = sqrt(max(0.0f, opp_cy*opp_cy + opp_gm*opp_gm))/SQRT3;

    // Smooth ach_d
    ach_d = (1.25f)*compress_toe_quadratic(ach_d, 0.25f, 0);

    // Hue angle
    float hue = fmod(atan2(opp_cy, opp_gm) + PI + 1.10714931f, 2.0f*PI);

    // RGB and CMY hue angles
    float3 ha_rgb = make_float3(
        gauss_window(hue_offset(hue, 0.1f), 0.9f),
        gauss_window(hue_offset(hue, 4.3f), 0.9f),
        gauss_window(hue_offset(hue, 2.3f), 0.9f)
    );

    float3 ha_cmy = make_float3(
        gauss_window(hue_offset(hue, 3.3f), 0.6f),
        gauss_window(hue_offset(hue, 1.3f), 0.6f),
        gauss_window(hue_offset(hue, -1.2f), 0.6f)
    );

    // Purity compression range
    float ts_pt_cmp = 1.0f - pow(ts_pt, 1.0f/params.ptRngLow);

    float pt_rng_high_f = min(1.0f, ach_d/1.2f);
    pt_rng_high_f *= pt_rng_high_f;
    pt_rng_high_f = params.ptRngHigh < 1.0f ? 1.0f - pt_rng_high_f : pt_rng_high_f;
    ts_pt_cmp = pow(ts_pt_cmp, params.ptRngHigh)*(1.0f - pt_rng_high_f) + ts_pt_cmp*pt_rng_high_f;

    // Brilliance
    float brl_f = 1.0f;
    if (params.brlR != 0.0f || params.brlG != 0.0f || params.brlB != 0.0f || 
        params.brlC != 0.0f || params.brlM != 0.0f || params.brlY != 0.0f) {
        brl_f = -params.brlR*ha_rgb.x - params.brlG*ha_rgb.y - params.brlB*ha_rgb.z - 
                params.brlC*ha_cmy.x - params.brlM*ha_cmy.y - params.brlY*ha_cmy.z;
        brl_f = (1.0f - ach_d)*brl_f + 1.0f - brl_f;
        brl_f = softplus(brl_f, 0.25f, -100.0f, 0.0f);
        
        // Limit brilliance adjustment
        float brl_ts = brl_f > 1.0f ? 1.0f - ts_pt : ts_pt;
        float brl_lim = spowf(brl_ts, 1.0f - params.brlRng);
        brl_f = brl_f*brl_lim + 1.0f - brl_lim;
        brl_f = max(0.0f, min(2.0f, brl_f));
    }

    // Mid-range purity
    float ptm_sc = 1.0f;
    if (params.ptmLow != 0.0f || params.ptmHigh != 0.0f) {
        // Mid purity low
        float ptm_ach_d = complement_power(ach_d, params.ptmLowSt);
        ptm_sc = sigmoid_cubic(ptm_ach_d, params.ptmLow*(1.0f - ts_pt));

        // Mid purity high
        ptm_ach_d = complement_power(ach_d, params.ptmHighSt)*(1.0f - ts_pt) + ach_d*ach_d*ts_pt;
        ptm_sc *= sigmoid_cubic(ptm_ach_d, params.ptmHigh*ts_pt);
        ptm_sc = max(0.0f, ptm_sc);
    }

    // Premult hue angles
    ha_rgb *= ach_d;
    ha_cmy *= (1.5f)*compress_toe_quadratic(ach_d, 0.5f, 0);

    // Hue contrast R
    if (params.hcR != 0.0f) {
        float hc_ts = 1.0f - ts_pt;
        float hc_c = (1.0f - ach_d)*hc_ts + ach_d*(1.0f - hc_ts);
        hc_c *= ha_rgb.x;
        hc_ts *= hc_ts;
        float hc_f = params.hcR*(hc_c - 2.0f*hc_c*hc_ts) + 1.0f;
        rgb = make_float3(rgb.x, rgb.y*hc_f, rgb.z*hc_f);
    }

    // Hue shift RGB
    if (params.hsR != 0.0f || params.hsG != 0.0f || params.hsB != 0.0f) {
        float3 hs_rgb = ha_rgb*pow(ts_pt, 1.0f/params.hsRgbRng);
        float3 hsf = make_float3(hs_rgb.x*params.hsR, hs_rgb.y*-params.hsG, hs_rgb.z*-params.hsB);
        hsf = make_float3(hsf.z - hsf.y, hsf.x - hsf.z, hsf.y - hsf.x);
        rgb += hsf;
    }

    // Hue shift CMY
    if (params.hsC != 0.0f || params.hsM != 0.0f || params.hsY != 0.0f) {
        float3 hs_cmy = ha_cmy*(1.0f - ts_pt);
        float3 hsf = make_float3(hs_cmy.x*-params.hsC, hs_cmy.y*params.hsM, hs_cmy.z*params.hsY);
        hsf = make_float3(hsf.z - hsf.y, hsf.x - hsf.z, hsf.y - hsf.x);
        rgb += hsf;
    }

    // Apply brilliance
    rgb *= brl_f;

    // Apply purity compression and mid purity
    ts_pt_cmp *= ptm_sc;
    rgb = rgb*ts_pt_cmp + 1.0f - ts_pt_cmp;

    // Inverse rendering space
    sat_L = rgb.x*rs_w.x + rgb.y*rs_w.y + rgb.z*rs_w.z;
    rgb = (sat_L*params.rsSa - rgb)/(params.rsSa - 1.0f);

    // Convert to final display gamut
    float3 cwp_rgb = rgb;
    if (params.displayGamut == 0) { // Rec.709
        if (params.cwp == 1) cwp_rgb = vdot(matrix_p3_to_rec709_d60, rgb);
        else if (params.cwp == 2) cwp_rgb = vdot(matrix_p3_to_rec709_d55, rgb);
        else if (params.cwp == 3) cwp_rgb = vdot(matrix_p3_to_rec709_d50, rgb);
        rgb = vdot(matrix_p3_to_rec709_d65, rgb);
        if (params.cwp == 0) cwp_rgb = rgb;
    } else if (params.displayGamut >= 1) { // P3 or Rec.2020
        if (params.cwp == 1) cwp_rgb = vdot(matrix_p3_to_p3_d60, rgb);
        else if (params.cwp == 2) cwp_rgb = vdot(matrix_p3_to_p3_d55, rgb);
        else if (params.cwp == 3) cwp_rgb = vdot(matrix_p3_to_p3_d50, rgb);
    }

    // Creative whitepoint mix
    float cwp_f = pow(tsn, 1.0f - params.cwpRng);
    rgb = cwp_rgb*cwp_f + rgb*(1.0f - cwp_f);

    // Purity compress low (simplified)
    float sum0 = softplus(rgb.x, 0.2f, -100.0f, -0.3f) + rgb.y + softplus(rgb.z, 0.2f, -100.0f, -0.3f);
    rgb.x = softplus(rgb.x, 0.04f, -0.3f, 0.0f);
    rgb.y = softplus(rgb.y, 0.06f, -0.3f, 0.0f);
    rgb.z = softplus(rgb.z, 0.01f, -0.05f, 0.0f);

    float ptl_norm = min(1.0f, sdivf(sum0, rgb.x + rgb.y + rgb.z));
    rgb *= ptl_norm;

    // Final tonescale adjustments
    tsn *= ts_m2;
    tsn = compress_toe_quadratic(tsn, params.tnToe, 0);
    tsn *= ts_dsc;

    // Return from RGB ratios
    rgb *= tsn;

    // Clamp if enabled
    if (params.clamp != 0) {
        rgb = clampf3(rgb, 0.0f, 1.0f);
    }

    // Rec.2020 (P3 Limited)
    if (params.displayGamut == 2) {
        rgb = clampminf3(rgb, 0.0f);
        rgb = vdot(matrix_p3_to_rec2020, rgb);
    }

    // Apply inverse display EOTF
    float eotf_p = 2.0f + params.eotf * 0.2f;
    if ((params.eotf > 0) && (params.eotf < 4)) {
        rgb = spowf3(rgb, 1.0f/eotf_p);
    } else if (params.eotf == 4) {
        rgb = eotf_pq(rgb, 1);
    } else if (params.eotf == 5) {
        rgb = eotf_hlg(rgb, 1);
    }

    return make_float4(rgb.x, rgb.y, rgb.z, inPixel.w);
}

const char* kernelSource =  \
"#include <metal_stdlib>\n" \
"using namespace metal;\n" \
"\n" \
"constant float SQRT3 = 1.73205080756887729353f;\n" \
"constant float PI = 3.14159265358979323846f;\n" \
"\n" \
"struct OpenDRTParams {\n" \
"    int inGamut;\n" \
"    int inOetf;\n" \
"    float tnLp;\n" \
"    float tnGb;\n" \
"    float ptHdr;\n" \
"    int clamp;\n" \
"    float tnLg;\n" \
"    float tnCon;\n" \
"    float tnSh;\n" \
"    float tnToe;\n" \
"    float tnOff;\n" \
"    float tnHcon;\n" \
"    float tnHconPv;\n" \
"    float tnHconSt;\n" \
"    float tnLcon;\n" \
"    float tnLconW;\n" \
"    float tnLconPc;\n" \
"    int cwp;\n" \
"    float cwpRng;\n" \
"    float rsSa;\n" \
"    float rsRw;\n" \
"    float rsBw;\n" \
"    float ptR;\n" \
"    float ptG;\n" \
"    float ptB;\n" \
"    float ptRngLow;\n" \
"    float ptRngHigh;\n" \
"    float ptmLow;\n" \
"    float ptmLowSt;\n" \
"    float ptmHigh;\n" \
"    float ptmHighSt;\n" \
"    float brlR;\n" \
"    float brlG;\n" \
"    float brlB;\n" \
"    float brlC;\n" \
"    float brlM;\n" \
"    float brlY;\n" \
"    float brlRng;\n" \
"    float hsR;\n" \
"    float hsG;\n" \
"    float hsB;\n" \
"    float hsRgbRng;\n" \
"    float hsC;\n" \
"    float hsM;\n" \
"    float hsY;\n" \
"    float hcR;\n" \
"    int displayGamut;\n" \
"    int eotf;\n" \
"};\n" \
"\n" \
"// Gamut Conversion Matrices - Input to Working Space (AP1/ACEScg)\n" \
"constant float3x3 matrix_rec709_to_ap1 = float3x3(\n" \
"    float3(0.6130974292755127f, 0.3395230770111084f, 0.04737949371337891f),\n" \
"    float3(0.0701942518651485f, 0.9163357615470886f, 0.01347050815820694f),\n" \
"    float3(0.02061580494046211f, 0.1095697879791260f, 0.8698144555091858f)\n" \
");\n" \
"\n" \
"constant float3x3 matrix_p3d65_to_ap1 = float3x3(\n" \
"    float3(0.8224605321407081f, 0.1773348898887634f, 0.0f),\n" \
"    float3(0.0331941693425178f, 0.9668058395385742f, 0.0f),\n" \
"    float3(0.01706488430500030f, 0.07240105420351028f, 0.9105340838432312f)\n" \
");\n" \
"\n" \
"constant float3x3 matrix_rec2020_to_ap1 = float3x3(\n" \
"    float3(0.9503623843193054f, 0.04963761568069458f, 0.0f),\n" \
"    float3(0.0181993357092285f, 0.9818006634712219f, 0.0f),\n" \
"    float3(0.002163206413388252f, 0.1080925464630127f, 0.8897442817687988f)\n" \
");\n" \
"\n" \
"constant float3x3 matrix_ap0_to_ap1 = float3x3(\n" \
"    float3(1.4514393806457710f, -0.2365107834339237f, -0.2149285435676575f),\n" \
"    float3(-0.0765537768602371f, 1.1762296050786972f, -0.0996758341133595f),\n" \
"    float3(0.0083161509788513f, -0.0060324496007175f, 0.9977162480354309f)\n" \
");\n" \
"\n" \
"// Output Gamut Matrices - Working Space (AP1) to Output\n" \
"constant float3x3 matrix_ap1_to_rec709 = float3x3(\n" \
"    float3(1.7050515413284302f, -0.6217905283164978f, -0.0832610130100250f),\n" \
"    float3(-0.1302597671747208f, 1.1408044099807739f, -0.0105446428060532f),\n" \
"    float3(-0.0240032123029232f, -0.1289696395397186f, 1.1529728174209595f)\n" \
");\n" \
"\n" \
"constant float3x3 matrix_ap1_to_p3d65 = float3x3(\n" \
"    float3(1.2249401807785034f, -0.2249402403831482f, 0.0f),\n" \
"    float3(-0.0420569784939289f, 1.0420570373535156f, 0.0f),\n" \
"    float3(-0.0196375157684088f, -0.0786361545324326f, 1.0982736349105835f)\n" \
");\n" \
"\n" \
"constant float3x3 matrix_ap1_to_rec2020 = float3x3(\n" \
"    float3(1.0525742769241333f, -0.0525742918252945f, 0.0f),\n" \
"    float3(-0.0194155070185661f, 1.0194155275821686f, 0.0f),\n" \
"    float3(-0.0027389433141797f, -0.1256923824548721f, 1.1284313201904297f)\n" \
");\n" \
"\n" \
"// Identity matrix for AP1/ACEScg passthrough\n" \
"constant float3x3 matrix_identity = float3x3(\n" \
"    float3(1.0f, 0.0f, 0.0f),\n" \
"    float3(0.0f, 1.0f, 0.0f),\n" \
"    float3(0.0f, 0.0f, 1.0f)\n" \
");\n" \
"\n" \
"// Math Helper Functions\n" \
"float sdivf(float a, float b) {\n" \
"    return (b == 0.0f) ? 0.0f : a/b;\n" \
"}\n" \
"\n" \
"float3 sdivf3f(float3 a, float b) {\n" \
"    return float3(sdivf(a.x, b), sdivf(a.y, b), sdivf(a.z, b));\n" \
"}\n" \
"\n" \
"float spowf(float a, float b) {\n" \
"    return (a <= 0.0f) ? a : pow(a, b);\n" \
"}\n" \
"\n" \
"float3 spowf3(float3 a, float b) {\n" \
"    return float3(spowf(a.x, b), spowf(a.y, b), spowf(a.z, b));\n" \
"}\n" \
"\n" \
"float hypotf3(float3 a) {\n" \
"    return sqrt(a.x*a.x + a.y*a.y + a.z*a.z);\n" \
"}\n" \
"\n" \
"float fmaxf3(float3 a) {\n" \
"    return max(a.x, max(a.y, a.z));\n" \
"}\n" \
"\n" \
"float fminf3(float3 a) {\n" \
"    return min(a.x, min(a.y, a.z));\n" \
"}\n" \
"\n" \
"float3 clampmaxf3(float3 a, float mx) {\n" \
"    return make_float3(min(a.x, mx), min(a.y, mx), min(a.z, mx));\n" \
"}\n" \
"\n" \
"float3 clampminf3(float3 a, float mn) {\n" \
"    return make_float3(max(a.x, mn), max(a.y, mn), max(a.z, mn));\n" \
"}\n" \
"\n" \
"float clampf(float a, float mn, float mx) {\n" \
"    return min(max(a, mn), mx);\n" \
"}\n" \
"\n" \
"float3 clampf3(float3 a, float mn, float mx) {\n" \
"    return make_float3(clampf(a.x, mn, mx), clampf(a.y, mn, mx), clampf(a.z, mn, mx));\n" \
"}\n" \
"\n" \
"// OpenDRT Core Functions - ADD THESE MISSING FUNCTIONS\n" \
"float compress_hyperbolic_power(float x, float s, float p) {\n" \
"    if (s <= 0.0f || p <= 0.0f) return x;\n" \
"    return spowf(x/(x + s), p);\n" \
"}\n" \
"\n" \
"float compress_toe_quadratic(float x, float toe, bool inv) {\n" \
"    if (toe <= 0.0f) return x;\n" \
"    if (!inv) {\n" \
"        return x*x/(x + toe);\n" \
"    } else {\n" \
"        return (x + sqrt(x*(4.0f*toe + x)))/2.0f;\n" \
"    }\n" \
"}\n" \
"\n" \
"// Color Space Transform Functions\n" \
"float3x3 get_input_gamut_matrix(int gamut) {\n" \
"    switch(gamut) {\n" \
"        case 0: return matrix_identity;           // XYZ (assume already in working space)\n" \
"        case 1: return matrix_ap0_to_ap1;         // ACES 2065-1\n" \
"        case 2: return matrix_identity;           // ACEScg (AP1) - already working space\n" \
"        case 3: return matrix_p3d65_to_ap1;       // P3-D65\n" \
"        case 4: return matrix_rec2020_to_ap1;     // Rec.2020\n" \
"        case 5: return matrix_rec709_to_ap1;      // Rec.709\n" \
"        default: return matrix_identity;          // Default passthrough\n" \
"    }\n" \
"}\n" \
"\n" \
"float3x3 get_output_gamut_matrix(int gamut) {\n" \
"    switch(gamut) {\n" \
"        case 0: return matrix_identity;           // XYZ (assume working space output)\n" \
"        case 1: return matrix_identity;           // ACES 2065-1 (would need AP1 to AP0)\n" \
"        case 2: return matrix_identity;           // ACEScg (AP1) - working space\n" \
"        case 3: return matrix_ap1_to_p3d65;       // P3-D65\n" \
"        case 4: return matrix_ap1_to_rec2020;     // Rec.2020\n" \
"        case 5: return matrix_ap1_to_rec709;      // Rec.709\n" \
"        default: return matrix_ap1_to_rec709;     // Default to Rec.709\n" \
"    }\n" \
"}\n" \
"\n" \
"// OETF/EOTF Functions (simplified - can be expanded)\n" \
"float3 apply_input_oetf(float3 rgb, int oetf) {\n" \
"    switch(oetf) {\n" \
"        case 0: return rgb;                       // Linear - no transform\n" \
"        case 1: return rgb;                       // DaVinci Intermediate - assume linear for now\n" \
"        case 2: return rgb;                       // Filmlight T-Log - TODO: implement\n" \
"        case 3: return rgb;                       // ACEScct - TODO: implement\n" \
"        default: return rgb;                      // Default linear\n" \
"    }\n" \
"}\n" \
"\n" \
"float3 apply_output_eotf(float3 rgb, int eotf) {\n" \
"    switch(eotf) {\n" \
"        case 0: return rgb;                       // Linear - no transform\n" \
"        case 1: return spowf3(clampf3(rgb, 0.0f, 1.0f), 1.0f/2.4f); // sRGB gamma\n" \
"        case 2: return spowf3(clampf3(rgb, 0.0f, 1.0f), 1.0f/2.2f); // Rec.709 gamma\n" \
"        default: return rgb;                      // Default linear\n" \
"    }\n" \
"}\n" \
"\n" \
"// Simplified OpenDRT Transform - Basic Working Version\n" \
"float3 apply_opendrt(float3 rgb, constant OpenDRTParams& params) {\n" \
"    // INPUT COLOR SPACE TRANSFORM\n" \
"    // 1. Apply input OETF (linearize if needed)\n" \
"    rgb = apply_input_oetf(rgb, params.inOetf);\n" \
"    \n" \
"    // 2. Transform input gamut to working space (AP1/ACEScg)\n" \
"    float3x3 inputMatrix = get_input_gamut_matrix(params.inGamut);\n" \
"    rgb = inputMatrix * rgb;\n" \
"    \n" \
"    // SIMPLIFIED OPENDRT PROCESSING (safe implementation)\n" \
"    // Apply small offset to prevent division by zero\n" \
"    rgb += 1e-6f;\n" \
"    \n" \
"    // Basic tonescale parameters\n" \
"    float contrast = clampf(params.tnCon, 0.1f, 3.0f);\n" \
"    float brightness = params.tnLg / 100.0f;\n" \
"    \n" \
"    // Simple luminance-based tonescale\n" \
"    float luma = 0.2126f*rgb.x + 0.7152f*rgb.y + 0.0722f*rgb.z;\n" \
"    float scaled_luma = pow(max(0.0f, luma), 1.0f/contrast) * brightness;\n" \
"    \n" \
"    // Preserve color ratios\n" \
"    if (luma > 1e-6f) {\n" \
"        rgb *= scaled_luma / luma;\n" \
"    }\n" \
"    \n" \
"    // Apply tonescale offset\n" \
"    rgb += params.tnOff;\n" \
"    \n" \
"    // OUTPUT COLOR SPACE TRANSFORM\n" \
"    // 3. Transform from working space to output gamut\n" \
"    float3x3 outputMatrix = get_output_gamut_matrix(params.displayGamut);\n" \
"    rgb = outputMatrix * rgb;\n" \
"    \n" \
"    // Clamp if enabled\n" \
"    if (params.clamp != 0) {\n" \
"        rgb = clampf3(rgb, 0.0f, 1.0f);\n" \
"    }\n" \
"    \n" \
"    // 4. Apply output EOTF (encode for display)\n" \
"    rgb = apply_output_eotf(rgb, params.eotf);\n" \
"    \n" \
"    return rgb;\n" \
"}\n" \
"\n" \
"kernel void OpenDRTKernel(constant int& p_Width [[buffer(11)]], constant int& p_Height [[buffer(12)]],\n" \
"                          const device float* p_Input [[buffer(0)]], device float* p_Output [[buffer(8)]],\n" \
"                          constant OpenDRTParams& params [[buffer(9)]],\n" \
"                          uint2 id [[thread_position_in_grid]])\n" \
"{\n" \
"    if ((id.x < p_Width) && (id.y < p_Height))\n" \
"    {\n" \
"        const int index = ((id.y * p_Width) + id.x) * 4;\n" \
"        \n" \
"        // Extract RGBA values\n" \
"        float3 rgb = float3(p_Input[index + 0], p_Input[index + 1], p_Input[index + 2]);\n" \
"        float a = p_Input[index + 3];\n" \
"        \n" \
"        // Apply complete OpenDRT processing with CST\n" \
"        rgb = apply_opendrt(rgb, params);\n" \
"        \n" \
"        // Output to buffer\n" \
"        p_Output[index + 0] = rgb.x;\n" \
"        p_Output[index + 1] = rgb.y;\n" \
"        p_Output[index + 2] = rgb.z;\n" \
"        p_Output[index + 3] = a;\n" \
"    }\n" \
"}\n";

std::mutex s_PipelineQueueMutex;
typedef std::unordered_map<id<MTLCommandQueue>, id<MTLComputePipelineState>> PipelineQueueMap;
PipelineQueueMap s_PipelineQueueMap;

// Helper function to populate OpenDRT parameters struct
OpenDRTParams createOpenDRTParams(
    int p_InGamut, int p_InOetf,
    float p_TnLp, float p_TnGb, float p_PtHdr,
    bool p_Clamp, float p_TnLg, float p_TnCon, float p_TnSh, float p_TnToe, float p_TnOff,
    float p_TnHcon, float p_TnHconPv, float p_TnHconSt,
    float p_TnLcon, float p_TnLconW, float p_TnLconPc,
    int p_Cwp, float p_CwpRng,
    float p_RsSa, float p_RsRw, float p_RsBw,
    float p_PtR, float p_PtG, float p_PtB, float p_PtRngLow, float p_PtRngHigh,
    float p_PtmLow, float p_PtmLowSt, float p_PtmHigh, float p_PtmHighSt,
    float p_BrlR, float p_BrlG, float p_BrlB, 
    float p_BrlC, float p_BrlM, float p_BrlY, float p_BrlRng,
    float p_HsR, float p_HsG, float p_HsB, float p_HsRgbRng,
    float p_HsC, float p_HsM, float p_HsY,
    float p_HcR,
    int p_DisplayGamut, int p_Eotf)
{
    OpenDRTParams params = {};
    
    // Input Settings
    params.inGamut = p_InGamut;
    params.inOetf = p_InOetf;
    
    // Tonescale Parameters
    params.tnLp = p_TnLp;
    params.tnGb = p_TnGb;
    params.ptHdr = p_PtHdr;
    
    // Clamp Parameters (convert bool to int)
    params.clamp = p_Clamp ? 1 : 0;
    params.tnLg = p_TnLg;
    params.tnCon = p_TnCon;
    params.tnSh = p_TnSh;
    params.tnToe = p_TnToe;
    params.tnOff = p_TnOff;
    
    // High Contrast Parameters
    params.tnHcon = p_TnHcon;
    params.tnHconPv = p_TnHconPv;
    params.tnHconSt = p_TnHconSt;
    
    // Low Contrast Parameters
    params.tnLcon = p_TnLcon;
    params.tnLconW = p_TnLconW;
    params.tnLconPc = p_TnLconPc;
    
    // Creative White Parameters
    params.cwp = p_Cwp;
    params.cwpRng = p_CwpRng;
    
    // Render Space Parameters
    params.rsSa = p_RsSa;
    params.rsRw = p_RsRw;
    params.rsBw = p_RsBw;
    
    // Purity Compress Parameters
    params.ptR = p_PtR;
    params.ptG = p_PtG;
    params.ptB = p_PtB;
    params.ptRngLow = p_PtRngLow;
    params.ptRngHigh = p_PtRngHigh;
    
    // Mid Purity Parameters
    params.ptmLow = p_PtmLow;
    params.ptmLowSt = p_PtmLowSt;
    params.ptmHigh = p_PtmHigh;
    params.ptmHighSt = p_PtmHighSt;
    
    // Brilliance Parameters
    params.brlR = p_BrlR;
    params.brlG = p_BrlG;
    params.brlB = p_BrlB;
    params.brlC = p_BrlC;
    params.brlM = p_BrlM;
    params.brlY = p_BrlY;
    params.brlRng = p_BrlRng;
    
    // Hueshift RGB Parameters
    params.hsR = p_HsR;
    params.hsG = p_HsG;
    params.hsB = p_HsB;
    params.hsRgbRng = p_HsRgbRng;
    
    // Hueshift CMY Parameters
    params.hsC = p_HsC;
    params.hsM = p_HsM;
    params.hsY = p_HsY;
    
    // Hue Contrast Parameters
    params.hcR = p_HcR;
    
    // Display Parameters
    params.displayGamut = p_DisplayGamut;
    params.eotf = p_Eotf;
    
    return params;
}

void OpenDRTKernel(void* p_CmdQ, int p_Width, int p_Height,
                   const float* p_Input, float* p_Output,
                   int p_InGamut, int p_InOetf,
                   float p_TnLp, float p_TnGb, float p_PtHdr,
                   bool p_Clamp, float p_TnLg, float p_TnCon, float p_TnSh, float p_TnToe, float p_TnOff,
                   float p_TnHcon, float p_TnHconPv, float p_TnHconSt,
                   float p_TnLcon, float p_TnLconW, float p_TnLconPc,
                   int p_Cwp, float p_CwpRng,
                   float p_RsSa, float p_RsRw, float p_RsBw,
                   float p_PtR, float p_PtG, float p_PtB, float p_PtRngLow, float p_PtRngHigh,
                   float p_PtmLow, float p_PtmLowSt, float p_PtmHigh, float p_PtmHighSt,
                   float p_BrlR, float p_BrlG, float p_BrlB, 
                   float p_BrlC, float p_BrlM, float p_BrlY, float p_BrlRng,
                   float p_HsR, float p_HsG, float p_HsB, float p_HsRgbRng,
                   float p_HsC, float p_HsM, float p_HsY,
                   float p_HcR,
                   int p_DisplayGamut, int p_Eotf)
{
    const char* kernelName = "OpenDRTKernel";

    id<MTLCommandQueue> queue = static_cast<id<MTLCommandQueue>>(p_CmdQ);
    id<MTLDevice> device = queue.device;
    id<MTLLibrary> metalLibrary = nil;
    id<MTLFunction> kernelFunction = nil;
    id<MTLComputePipelineState> pipelineState = nil;
    NSError* err = nil;

    std::unique_lock<std::mutex> lock(s_PipelineQueueMutex);

    auto it = s_PipelineQueueMap.find(queue);
    if (it == s_PipelineQueueMap.end()) {
        MTLCompileOptions* options = [MTLCompileOptions new];
        #if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
            options.mathMode = MTLMathModeFast;
        #else
            options.fastMathEnabled = YES;
        #endif

        metalLibrary = [device newLibraryWithSource:@(kernelSource) options:options error:&err];
        [options release];
        if (!metalLibrary) {
            fprintf(stderr, "Failed to compile Metal library: %s\n", err.localizedDescription.UTF8String);
            return;
        }

        kernelFunction = [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:kernelName]];
        if (!kernelFunction) {
            fprintf(stderr, "Failed to get Metal kernel function.\n");
            [metalLibrary release];
            return;
        }

        pipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&err];
        if (!pipelineState) {
            fprintf(stderr, "Failed to create Metal pipeline: %s\n", err.localizedDescription.UTF8String);
            [metalLibrary release];
            [kernelFunction release];
            return;
        }

        s_PipelineQueueMap[queue] = pipelineState;
        [metalLibrary release];
        [kernelFunction release];
    } else {
        pipelineState = it->second;
    }

    lock.unlock();

    // Create and populate the parameter struct using helper function
    OpenDRTParams params = createOpenDRTParams(
        p_InGamut, p_InOetf, p_TnLp, p_TnGb, p_PtHdr,
        p_Clamp, p_TnLg, p_TnCon, p_TnSh, p_TnToe, p_TnOff,
        p_TnHcon, p_TnHconPv, p_TnHconSt,
        p_TnLcon, p_TnLconW, p_TnLconPc,
        p_Cwp, p_CwpRng,
        p_RsSa, p_RsRw, p_RsBw,
        p_PtR, p_PtG, p_PtB, p_PtRngLow, p_PtRngHigh,
        p_PtmLow, p_PtmLowSt, p_PtmHigh, p_PtmHighSt,
        p_BrlR, p_BrlG, p_BrlB, p_BrlC, p_BrlM, p_BrlY, p_BrlRng,
        p_HsR, p_HsG, p_HsB, p_HsRgbRng,
        p_HsC, p_HsM, p_HsY, p_HcR,
        p_DisplayGamut, p_Eotf
    );

    id<MTLBuffer> srcDeviceBuf = reinterpret_cast<id<MTLBuffer>>(const_cast<float*>(p_Input));
    id<MTLBuffer> dstDeviceBuf = reinterpret_cast<id<MTLBuffer>>(p_Output);

    // Create parameter buffer from struct
    id<MTLBuffer> paramBuffer = [device newBufferWithBytes:&params 
                                                    length:sizeof(OpenDRTParams) 
                                                   options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    commandBuffer.label = @"OpenDRTKernel";

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:pipelineState];

    int exeWidth = [pipelineState threadExecutionWidth];
    MTLSize threadGroupCount = MTLSizeMake(exeWidth, 1, 1);
    MTLSize threadGroups = MTLSizeMake((p_Width + exeWidth - 1)/exeWidth, p_Height, 1);
    
    // Set buffers - now much cleaner!
    [computeEncoder setBuffer:srcDeviceBuf offset:0 atIndex:0];      // Input buffer
    [computeEncoder setBuffer:dstDeviceBuf offset:0 atIndex:8];      // Output buffer
    [computeEncoder setBuffer:paramBuffer offset:0 atIndex:9];       // Parameters struct
    [computeEncoder setBytes:&p_Width length:sizeof(int) atIndex:11]; // Width
    [computeEncoder setBytes:&p_Height length:sizeof(int) atIndex:12]; // Height

    [computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup:threadGroupCount];
    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    [paramBuffer release];
}