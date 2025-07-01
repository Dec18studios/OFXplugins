// OpenCLKernel.cpp
#include <cmath>
#include <algorithm>
#include <cstring>
#include "OpenDRTParams.h"
#include "OpenDRTPresets.h"

// OpenCL constants
#define SQRT3 1.73205080756887729353f
#define PI 3.14159265358979323846f

// OpenCL float3 utilities
typedef struct {
    float x, y, z;
} float3;

typedef struct {
    float x, y;
} float2;

float3 make_float3(float x, float y, float z) {
    float3 result = {x, y, z};
    return result;
}

float2 make_float2(float x, float y) {
    float2 result = {x, y};
    return result;
}

float3 float3_add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

float3 float3_sub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

float3 float3_mul(float3 a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

float3 float3_mul3(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

float3 float3_div(float3 a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

// Matrix operations
float3 vdot(float matrix[9], float3 v) {
    // Matrix is stored in column-major order
    return make_float3(matrix[0]*v.x + matrix[3]*v.y + matrix[6]*v.z,
                      matrix[1]*v.x + matrix[4]*v.y + matrix[7]*v.z,
                      matrix[2]*v.x + matrix[5]*v.y + matrix[8]*v.z);
}

// Math Helper Functions
float sdivf(float a, float b) {
    return (b == 0.0f) ? 0.0f : a/b;
}

float3 sdivf3f(float3 a, float b) {
    return make_float3(sdivf(a.x, b), sdivf(a.y, b), sdivf(a.z, b));
}

float spowf(float a, float b) {
    return (a <= 0.0f) ? a : powf(a, b);
}

float3 spowf3(float3 a, float b) {
    return make_float3(spowf(a.x, b), spowf(a.y, b), spowf(a.z, b));
}

float clampf(float a, float mn, float mx) {
    return fminf(fmaxf(a, mn), mx);
}

float3 clampf3(float3 a, float mn, float mx) {
    return make_float3(clampf(a.x, mn, mx), clampf(a.y, mn, mx), clampf(a.z, mn, mx));
}

float3 clampminf3(float3 a, float mn) {
    return make_float3(fmaxf(a.x, mn), fmaxf(a.y, mn), fmaxf(a.z, mn));
}

float fmaxf3(float3 a) {
    return fmaxf(a.x, fmaxf(a.y, a.z));
}

float fminf3(float3 a) {
    return fminf(a.x, fminf(a.y, a.z));
}

float hypotf3(float3 a) {
    return sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
}

float compress_toe_cubic(float x, float m, float w, int inv) {
    if (m == 1.0f) return x;
    float x2 = x * x;
    if (inv == 0) {
        return x * (x2 + m * w) / (x2 + w);
    } else {
        float p0 = x2 - 3.0f * m * w;
        float p1 = 2.0f * x2 + 27.0f * w - 9.0f * m * w;
        float p2 = powf(sqrtf(x2 * p1 * p1 - 4 * p0 * p0 * p0) / 2.0f + x * p1 / 2.0f, 1.0f / 3.0f);
        return p0 / (3.0f * p2) + p2 / 3.0f + x / 3.0f;
    }
}

float compress_hyperbolic_power(float x, float s, float p) {
    return spowf(x / (x + s), p);
}

float contrast_high(float x, float p, float pv, float pv_lx, int inv) {
    const float x0 = 0.18f * powf(2.0f, pv);
    if (x < x0 || p == 1.0f) return x;

    const float o = x0 - x0 / p;
    const float s0 = powf(x0, 1.0f - p) / p;
    const float x1 = x0 * powf(2.0f, pv_lx);
    const float k1 = p * s0 * powf(x1, p) / x1;
    const float y1 = s0 * powf(x1, p) + o;
    if (inv == 1)
        return x > y1 ? (x - y1) / k1 + x1 : powf((x - o) / s0, 1.0f / p);
    else
        return x > x1 ? k1 * (x - x1) + y1 : s0 * powf(x, p) + o;
}

float gauss_window(float x, float w) {
    x /= w;
    return expf(-x * x);
}

float hue_offset(float h, float o) {
    return fmodf(h - o + PI, 2.0f * PI) - PI;
}

float compress_toe_quadratic(float x, float toe, int inv) {
    if (toe == 0.0f) return x;
    if (inv == 0) {
        return spowf(x, 2.0f) / (x + toe);
    } else {
        return (x + sqrtf(x * (4.0f * toe + x))) / 2.0f;
    }
}

float complement_power(float x, float p) {
    return 1.0f - spowf(1.0f - x, 1.0f/p);
}

float sigmoid_cubic(float x, float s) {
    if (x < 0.0f || x > 1.0f) return 1.0f;
    return 1.0f + s*(1.0f - 3.0f*x*x + 2.0f*x*x*x);
}

float softplus(float x, float s, float x0, float y0) {
    if (x > 10.0f*s + y0 || s < 1e-3f) return x;
    float m = 1.0f;
     if (fabsf(y0) > 1e-6f) m = expf(y0/s);
    m -= expf(x0/s);
    return s*logf(fmaxf(0.0f, m + expf(x/s)));
}

// HARDCODED OETF FUNCTIONS
float oetf_davinci_intermediate(float x) {
    return x <= 0.02740668f ? x/10.44426855f : exp2f(x/0.07329248f - 7.0f) - 0.0075f;
}

float oetf_filmlight_tlog(float x) {
    return x < 0.075f ? (x-0.075f)/16.184376489665897f : expf((x - 0.5520126568606655f)/0.09232902596577353f) - 0.0057048244042473785f;
}

float oetf_acescct(float x) {
    return x <= 0.155251141552511f ? (x - 0.0729055341958355f)/10.5402377416545f : exp2f(x*17.52f - 9.72f);
}

float oetf_arri_logc3(float x) {
    return x < 5.367655f*0.010591f + 0.092809f ? (x - 0.092809f)/5.367655f : (powf(10.0f, (x - 0.385537f)/0.247190f) - 0.052272f)/5.555556f;
}

float oetf_arri_logc4(float x) {
    return x < -0.7774983977293537f ? x*0.3033266726886969f - 0.7774983977293537f : (exp2f(14.0f*(x - 0.09286412512218964f)/0.9071358748778103f + 6.0f) - 64.0f)/2231.8263090676883f;
}

float oetf_red_log3g10(float x) {
    return x < 0.0f ? (x/15.1927f) - 0.01f : (powf(10.0f, x/0.224282f) - 1.0f)/155.975327f - 0.01f;
}

float oetf_panasonic_vlog(float x) {
    return x < 0.181f ? (x - 0.125f)/5.6f : powf(10.0f, (x - 0.598206f)/0.241514f) - 0.00873f;
}

float oetf_sony_slog3(float x) {
    return x < 171.2102946929f/1023.0f ? (x*1023.0f - 95.0f)*0.01125f/(171.2102946929f - 95.0f) : (powf(10.0f, ((x*1023.0f - 420.0f)/261.5f))*(0.18f + 0.01f) - 0.01f);
}

float oetf_fujifilm_flog2(float x) {
    return x < 0.100686685370811f ? (x - 0.092864f)/8.799461f : (powf(10.0f, ((x - 0.384316f)/0.245281f))/5.555556f - 0.064829f/5.555556f);
}

float apply_oetf(float x, int type) {
    switch(type) {
        case 0: return x; // Linear
        case 1: return oetf_davinci_intermediate(x);
        case 2: return oetf_filmlight_tlog(x);
        case 3: return oetf_acescct(x);
        case 4: return oetf_arri_logc3(x);
        case 5: return oetf_arri_logc4(x);
        case 6: return oetf_red_log3g10(x);
        case 7: return oetf_panasonic_vlog(x);
        case 8: return oetf_sony_slog3(x);
        case 9: return oetf_fujifilm_flog2(x);
        default: return x;
    }
}

// HARDCODED EOTF FUNCTIONS
float3 eotf_hlg(float3 rgb, int inverse) {
    if (inverse == 1) {
        float Yd = 0.2627f*rgb.x + 0.6780f*rgb.y + 0.0593f*rgb.z;
        rgb = float3_mul(rgb, powf(Yd, (1.0f - 1.2f)/1.2f));
        rgb.x = rgb.x <= 1.0f/12.0f ? sqrtf(3.0f*rgb.x) : 0.17883277f*logf(12.0f*rgb.x - 0.28466892f) + 0.55991073f;
        rgb.y = rgb.y <= 1.0f/12.0f ? sqrtf(3.0f*rgb.y) : 0.17883277f*logf(12.0f*rgb.y - 0.28466892f) + 0.55991073f;
        rgb.z = rgb.z <= 1.0f/12.0f ? sqrtf(3.0f*rgb.z) : 0.17883277f*logf(12.0f*rgb.z - 0.28466892f) + 0.55991073f;
    } else {
        rgb.x = rgb.x <= 0.5f ? rgb.x*rgb.x/3.0f : (expf((rgb.x - 0.55991073f)/0.17883277f) + 0.28466892f)/12.0f;
        rgb.y = rgb.y <= 0.5f ? rgb.y*rgb.y/3.0f : (expf((rgb.y - 0.55991073f)/0.17883277f) + 0.28466892f)/12.0f;
        rgb.z = rgb.z <= 0.5f ? rgb.z*rgb.z/3.0f : (expf((rgb.z - 0.55991073f)/0.17883277f) + 0.28466892f)/12.0f;
        float Ys = 0.2627f*rgb.x + 0.6780f*rgb.y + 0.0593f*rgb.z;
        rgb = float3_mul(rgb, powf(Ys, 1.2f - 1.0f));
    }
    return rgb;
}

float3 eotf_pq(float3 rgb, int inverse) {
    const float m1 = 2610.0f/16384.0f;
    const float m2 = 2523.0f/32.0f;
    const float c1 = 107.0f/128.0f;
    const float c2 = 2413.0f/128.0f;
    const float c3 = 2392.0f/128.0f;

    if (inverse == 1) {
        rgb = spowf3(rgb, m1);
        rgb = spowf3(make_float3((c1 + c2*rgb.x)/(1.0f + c3*rgb.x),
                                 (c1 + c2*rgb.y)/(1.0f + c3*rgb.y),
                                 (c1 + c2*rgb.z)/(1.0f + c3*rgb.z)), m2);
    } else {
        rgb = spowf3(rgb, 1.0f/m2);
        rgb = spowf3(make_float3((rgb.x - c1)/(c2 - c3*rgb.x),
                                 (rgb.y - c1)/(c2 - c3*rgb.y),
                                 (rgb.z - c1)/(c2 - c3*rgb.z)), 1.0f/m1);
    }
    return rgb;
}

float apply_eotf(float x, int type, int inverse) {
    switch(type) {
        case 0: return x; // Linear
        case 1: return inverse ? powf(fmaxf(0.0f, x), 1.0f/2.2f) : powf(fmaxf(0.0f, x), 2.2f);
        case 2: return inverse ? powf(fmaxf(0.0f, x), 1.0f/2.4f) : powf(fmaxf(0.0f, x), 2.4f);
        case 3: return inverse ? powf(fmaxf(0.0f, x), 1.0f/2.6f) : powf(fmaxf(0.0f, x), 2.6f);
        case 4: // PQ - handled separately
            return x;
        case 5: // HLG - handled separately
            return x;
        default: return x;
    }
}

float3 encode_for_display(float3 rgb, int eotfType) {
    switch(eotfType) {
        case 4: return eotf_pq(rgb, 1); // PQ inverse
        case 5: return eotf_hlg(rgb, 1); // HLG inverse
        default:
            rgb.x = apply_eotf(rgb.x, eotfType, 1);
            rgb.y = apply_eotf(rgb.y, eotfType, 1);
            rgb.z = apply_eotf(rgb.z, eotfType, 1);
            return rgb;
    }
}

float3 linearize(float3 rgb, int tf) {
    if (tf == 0) { // Linear
        return rgb;
    } else {
        rgb.x = apply_oetf(rgb.x, tf);
        rgb.y = apply_oetf(rgb.y, tf);
        rgb.z = apply_oetf(rgb.z, tf);
        return rgb;
    }
}

// HARDCODED INPUT MATRIX FUNCTIONS
void getInputMatrix(int gamut, float matrix[9]) {
    switch(gamut) {
        case 0: // XYZ (Identity)
            matrix[0] = 1.0f; matrix[3] = 0.0f; matrix[6] = 0.0f;
            matrix[1] = 0.0f; matrix[4] = 1.0f; matrix[7] = 0.0f;
            matrix[2] = 0.0f; matrix[5] = 0.0f; matrix[8] = 1.0f;
            break;
        case 1: // AP0 to XYZ
            matrix[0] = 0.93863094875f; matrix[3] = 0.338093594922f; matrix[6] = 0.000723121511f;
            matrix[1] = -0.00574192055f; matrix[4] = 0.727213902811f; matrix[7] = 0.000818441849f;
            matrix[2] = 0.017566898852f; matrix[5] = -0.065307497733f; matrix[8] = 1.0875161874f;
            break;
        case 2: // AP1 to XYZ
            matrix[0] = 0.652418717672f; matrix[3] = 0.268064059194f; matrix[6] = -0.00546992851f;
            matrix[1] = 0.127179925538f; matrix[4] = 0.672464478993f; matrix[7] = 0.005182799977f;
            matrix[2] = 0.170857283842f; matrix[5] = 0.059471461813f; matrix[8] = 1.08934487929f;
            break;
        case 3: // P3-D65 to XYZ
            matrix[0] = 0.486571133137f; matrix[3] = 0.228974640369f; matrix[6] = 0.0f;
            matrix[1] = 0.265667706728f; matrix[4] = 0.691738605499f; matrix[7] = 0.045113388449f;
            matrix[2] = 0.198217317462f; matrix[5] = 0.079286918044f; matrix[8] = 1.043944478035f;
            break;
        case 4: // Rec.2020 to XYZ
            matrix[0] = 0.636958122253f; matrix[3] = 0.262700229883f; matrix[6] = 0.0f;
            matrix[1] = 0.144616916776f; matrix[4] = 0.677998125553f; matrix[7] = 0.028072696179f;
            matrix[2] = 0.168880969286f; matrix[5] = 0.059301715344f; matrix[8] = 1.060985088348f;
            break;
        case 5: // Rec.709 to XYZ
            matrix[0] = 0.412390917540f; matrix[3] = 0.212639078498f; matrix[6] = 0.019330825657f;
            matrix[1] = 0.357584357262f; matrix[4] = 0.715168714523f; matrix[7] = 0.119194783270f;
            matrix[2] = 0.180480793118f; matrix[5] = 0.072192311287f; matrix[8] = 0.950532138348f;
            break;
        // ... add other cases as needed
        default:
            matrix[0] = 1.0f; matrix[3] = 0.0f; matrix[6] = 0.0f;
            matrix[1] = 0.0f; matrix[4] = 1.0f; matrix[7] = 0.0f;
            matrix[2] = 0.0f; matrix[5] = 0.0f; matrix[8] = 1.0f;
            break;
    }
}

void getOutputMatrix(int displayGamut, float matrix[9]) {
    switch(displayGamut) {
        case 0: // P3 to Rec.709
            matrix[0] = 1.224940181f; matrix[3] = -0.04205697775f; matrix[6] = -0.01963755488f;
            matrix[1] = -0.2249402404f; matrix[4] = 1.042057037f; matrix[7] = -0.07863604277f;
            matrix[2] = 0.0f; matrix[5] = -1.4901e-08f; matrix[8] = 1.098273635f;
            break;
        case 1: // P3 Identity
            matrix[0] = 1.0f; matrix[3] = 0.0f; matrix[6] = 0.0f;
            matrix[1] = 0.0f; matrix[4] = 1.0f; matrix[7] = 0.0f;
            matrix[2] = 0.0f; matrix[5] = 0.0f; matrix[8] = 1.0f;
            break;
        case 2: // P3 to Rec.2020
            matrix[0] = 0.7538330344f; matrix[3] = 0.04574384897f; matrix[6] = -0.001210340355f;
            matrix[1] = 0.1985973691f; matrix[4] = 0.9417772198f; matrix[7] = 0.0176017173f;
            matrix[2] = 0.04756959659f; matrix[5] = 0.01247893122f; matrix[8] = 0.9836086231f;
            break;
        default:
            matrix[0] = 1.0f; matrix[3] = 0.0f; matrix[6] = 0.0f;
            matrix[1] = 0.0f; matrix[4] = 1.0f; matrix[7] = 0.0f;
            matrix[2] = 0.0f; matrix[5] = 0.0f; matrix[8] = 1.0f;
            break;
    }
}

void getCreativeWhitepointMatrix(int displayGamut, int cwp, float matrix[9]) {
    if (displayGamut == 0) { // Rec.709
        switch(cwp) {
            case 1: // D60
                matrix[0] = 1.189986856f; matrix[3] = -0.04168263635f; matrix[6] = -0.01937995127f;
                matrix[1] = -0.192168414f; matrix[4] = 0.9927757018f; matrix[7] = -0.07933006919f;
                matrix[2] = 0.002185496045f; matrix[5] = -5.5660878e-05f; matrix[8] = 0.9734397041f;
                break;
            case 2: // D55
                matrix[0] = 1.149327514f; matrix[3] = -0.0412590771f; matrix[6] = -0.01900949528f;
                matrix[1] = -0.1536910745f; matrix[4] = 0.9351717477f; matrix[7] = -0.07928282823f;
                matrix[2] = 0.004366526746f; matrix[5] = -0.000116126221f; matrix[8] = 0.8437884317f;
                break;
            case 3: // D50
                matrix[0] = 1.103807322f; matrix[3] = -0.04079386701f; matrix[6] = -0.01854055914f;
                matrix[1] = -0.1103425121f; matrix[4] = 0.8704694227f; matrix[7] = -0.07857582481f;
                matrix[2] = 0.006531676079f; matrix[5] = -0.000180522628f; matrix[8] = 0.7105498861f;
                break;
            default: // D65 (Identity)
                matrix[0] = 1.0f; matrix[3] = 0.0f; matrix[6] = 0.0f;
                matrix[1] = 0.0f; matrix[4] = 1.0f; matrix[7] = 0.0f;
                matrix[2] = 0.0f; matrix[5] = 0.0f; matrix[8] = 1.0f;
                break;
        }
    } else { // P3 and Rec.2020
        switch(cwp) {
            case 1: // D60
                matrix[0] = 0.979832881f; matrix[3] = -0.000805359793f; matrix[6] = -0.000338382322f;
                matrix[1] = 0.01836378979f; matrix[4] = 0.9618000331f; matrix[7] = -0.003671835795f;
                matrix[2] = 0.001803284786f; matrix[5] = 1.8876121e-05f; matrix[8] = 0.894139105f;
                break;
            case 2: // D55
                matrix[0] = 0.9559790976f; matrix[3] = -0.001771929896f; matrix[6] = -0.000674760809f;
                matrix[1] = 0.0403850003f; matrix[4] = 0.9163058305f; matrix[7] = -0.0072466358f;
                matrix[2] = 0.003639287409f; matrix[5] = 3.3300759e-05f; matrix[8] = 0.7831189153f;
                break;
            case 3: // D50
                matrix[0] = 0.9287127388f; matrix[3] = -0.002887159176f; matrix[6] = -0.001009551548f;
                matrix[1] = 0.06578032793f; matrix[4] = 0.8640709228f; matrix[7] = -0.01073503317f;
                matrix[2] = 0.005506708345f; matrix[5] = 4.3593718e-05f; matrix[8] = 0.6672692039f;
                break;
            default: // D65 (Identity)
                matrix[0] = 1.0f; matrix[3] = 0.0f; matrix[6] = 0.0f;
                matrix[1] = 0.0f; matrix[4] = 1.0f; matrix[7] = 0.0f;
                matrix[2] = 0.0f; matrix[5] = 0.0f; matrix[8] = 1.0f;
                break;
        }
    }
}

// Main kernel function (equivalent to Metal/CUDA kernel)
void OpenDRTKernel_OpenCL(int p_Width, int p_Height,
                          const float* p_Input, float* p_Output,
                          OpenDRTParams params)
{
    // Process each pixel
    for (int y = 0; y < p_Height; y++) {
        for (int x = 0; x < p_Width; x++) {
            const int index = ((y * p_Width) + x) * 4;
            
            /***************************************************
             setup and extraction
            --------------------------------------------------*/
            // Extract RGBA values
            float3 rgb = make_float3(p_Input[index + 0], p_Input[index + 1], p_Input[index + 2]);
            float a = p_Input[index + 3];
            
            // If diagnostics mode is enabled and in ramp area, set input to ramp value
            if (params.diagnosticsMode == 1 && y < 100) {
                float ramp = (float)x / (float)(p_Width - 1);
                rgb = make_float3(ramp, ramp, ramp);
            }
            
            // If RGB chips mode is enabled, create RGB test pattern
            if (params.rgbChipsMode == 1) {
                float ramp = (float)x / (float)(p_Width - 1);
                int band = y * 7 / p_Height;
                
                switch (band) {
                    case 0: rgb = make_float3(ramp, 0.0f, 0.0f); break;     // Red
                    case 1: rgb = make_float3(ramp, ramp, 0.0f); break;     // Yellow
                    case 2: rgb = make_float3(0.0f, ramp, 0.0f); break;     // Green
                    case 3: rgb = make_float3(0.0f, ramp, ramp); break;     // Cyan
                    case 4: rgb = make_float3(0.0f, 0.0f, ramp); break;     // Blue
                    case 5: rgb = make_float3(ramp, 0.0f, ramp); break;     // Magenta
                    case 6: rgb = make_float3(ramp, ramp, ramp); break;    // White/Gray
                    default: rgb = make_float3(ramp, ramp, ramp); break;    // White/Gray
                }
            }
            
            rgb = linearize(rgb, params.inOetf);
            
            // Load dynamic matrices using switch functions
            float inputMatrix[9];
            float outputMatrix[9];
            float cwpMatrix[9];
            getInputMatrix(params.inGamut, inputMatrix);
            getOutputMatrix(params.displayGamut, outputMatrix);
            getCreativeWhitepointMatrix(params.displayGamut, params.cwp, cwpMatrix);
            
            // Apply input matrix transform
            rgb = vdot(inputMatrix, rgb);
            
            // XYZ to P3 transform (hardcoded)
            float hardcodedxyzToP3Matrix[9] = {
                 2.49349691194f, -0.931383617919f, -0.402710784451f,
                -0.829488694668f, 1.76266097069f, 0.0236246771724f,
                 0.0358458302915f, -0.0761723891287f, 0.956884503364f
            };
            rgb = vdot(hardcodedxyzToP3Matrix, rgb);
            
            /***************************************************
             Tonescale Overlay Initialization
            --------------------------------------------------*/
            float crv_val = 0.0f;
            float2 pos = make_float2(x, y);
            float2 res = make_float2(p_Width, p_Height);
            
            // x-position based input value for tonescale overlay
            if (params.tonescaleMap == 1) {
                crv_val = oetf_filmlight_tlog(pos.x/res.x);
            }
            
            // Rendering Space: "Desaturate" to control scale of the color volume in the rgb ratios.
            float3 rs_w = make_float3(params.rsRw, 1.0f - params.rsRw - params.rsBw, params.rsBw);
            float sat_L = rgb.x*rs_w.x + rgb.y*rs_w.y + rgb.z*rs_w.z;
            rgb = float3_add(float3_mul(make_float3(sat_L, sat_L, sat_L), params.rsSa), float3_mul(rgb, (1.0f - params.rsSa)));
            
            // Offset
            rgb = float3_add(rgb, make_float3(params.tnOff, params.tnOff, params.tnOff));
            if (params.tonescaleMap == 1) crv_val += params.tnOff;
            
            /***************************************************
              Contrast Low Module
            --------------------------------------------------*/
            if (params.tnLconPresetEnable || params.tnLconUIEnable) {
                float mcon_m = powf(2.0f, -params.tnLcon);
                float mcon_w = params.tnLconW/4.0f;
                mcon_w *= mcon_w;

                // Normalize for ts_x0 intersection constraint
                const float mcon_cnst_sc = compress_toe_cubic(params.ts_x0, mcon_m, mcon_w, 1)/params.ts_x0;
                rgb = float3_mul(rgb, mcon_cnst_sc);

                // Scale for ratio-preserving midtone contrast
                float mcon_nm = hypotf3(clampminf3(rgb, 0.0f))/SQRT3;
                float mcon_sc = (mcon_nm*mcon_nm + mcon_m*mcon_w)/(mcon_nm*mcon_nm + mcon_w);

                if (params.tnLconPc > 0.0f) {
                    // Mix between ratio-preserving and per-channel by blending based on distance from achromatic
                    // Apply per-channel midtone contrast
                    float3 mcon_rgb = rgb;
                    mcon_rgb.x = compress_toe_cubic(rgb.x, mcon_m, mcon_w, 0);
                    mcon_rgb.y = compress_toe_cubic(rgb.y, mcon_m, mcon_w, 0);
                    mcon_rgb.z = compress_toe_cubic(rgb.z, mcon_m, mcon_w, 0);

                    // Always use some amount of ratio-preserving method towards gamut boundary
                    float mcon_mx = fmaxf3(rgb);
                    float mcon_mn = fminf3(rgb);
                    float mcon_ch = clampf(1.0f - sdivf(mcon_mn, mcon_mx), 0.0, 1.0);
                    mcon_ch = powf(mcon_ch, 4.0f*params.tnLconPc);
                    rgb = float3_add(float3_mul(float3_mul(rgb, mcon_sc), mcon_ch), float3_mul(mcon_rgb, (1.0f - mcon_ch)));
                }
                else { // Just use ratio-preserving
                    rgb = float3_mul(rgb, mcon_sc);
                }
                
                // Overlay tracking for low contrast
                if (params.tonescaleMap == 1) {
                    crv_val *= mcon_cnst_sc;
                    crv_val = crv_val*(crv_val*crv_val + mcon_m*mcon_w)/(crv_val*crv_val + mcon_w);
                }
            }

            /***************************************************
              Filmic Dynamic Range Compression (BETA FEATURE)
            --------------------------------------------------*/
            if (params.filmicMode == 1 && params.betaFeaturesEnable == 1) {
                // Calculate maximum input based on original camera range
                float maxInput = powf(2.0f, params.filmicSourceStops);
                
                // Normalize RGB to 0-1 range based on original camera stops
                float3 normalizedRGB = float3_div(rgb, maxInput);
                
                // Use Film Dynamic Range to control highlight rolloff characteristics
                float rolloff_s = 0.05f + (params.filmicDynamicRange / 10.0f);
                float rolloff_p = 0.8f + (params.filmicDynamicRange / 25.0f);
                
                // Apply hyperbolic compression with dynamic rolloff
                float3 compressedRGB = make_float3(
                    compress_hyperbolic_power(normalizedRGB.x, rolloff_s, rolloff_p),
                    compress_hyperbolic_power(normalizedRGB.y, rolloff_s, rolloff_p),
                    compress_hyperbolic_power(normalizedRGB.z, rolloff_s, rolloff_p)
                );
                
                // Rescale to target film range
                float maxOutput = powf(2.0f, params.filmicTargetStops);
                float3 rescaledRGB = float3_mul(compressedRGB, maxOutput);
                
                // Mix between original and filmic compressed result
                rgb = float3_add(float3_mul(rgb, (1.0f - params.filmicStrength)), float3_mul(rescaledRGB, params.filmicStrength));
                
                // Apply same compression to overlay curve
                if (params.tonescaleMap == 1) {
                    float crv_normalized = crv_val / maxInput;
                    float crv_compressed = compress_hyperbolic_power(crv_normalized, rolloff_s, rolloff_p);
                    float crv_rescaled = crv_compressed * maxOutput;
                    crv_val = crv_val * (1.0f - params.filmicStrength) + crv_rescaled * params.filmicStrength;
                }
            }

            /***************************************************
             Tonescale and RGB Ratios
            --------------------------------------------------*/
            // Tonescale Norm
            float tsn = hypotf3(clampminf3(rgb, 0.0f)) / SQRT3;
            // Purity Compression Norm
            float ts_pt = sqrtf(fmaxf(0.0f, rgb.x * rgb.x * params.ptR + rgb.y * rgb.y * params.ptG + rgb.z * rgb.z * params.ptB));
            
            // RGB Ratios
            rgb = sdivf3f(clampminf3(rgb, -2.0f), tsn);
            
            /***************************************************
              Apply High Contrast
            --------------------------------------------------*/
            if (params.tnHconPresetEnable || params.tnHconUIEnable) {
                float hcon_p = powf(2.0f, params.tnHcon);
                tsn = contrast_high(tsn, hcon_p, params.tnHconPv, params.tnHconSt, 0);
                ts_pt = contrast_high(ts_pt, hcon_p, params.tnHconPv, params.tnHconSt, 0);
                if (params.tonescaleMap == 1) crv_val = contrast_high(crv_val, hcon_p, params.tnHconPv, params.tnHconSt, 0);
            }

            /***************************************************
              Apply Tonescale
            --------------------------------------------------*/
            tsn = compress_hyperbolic_power(tsn, params.ts_s, params.tnCon);
            ts_pt = compress_hyperbolic_power(ts_pt, params.ts_s1, params.tnCon);
            
            if (params.tonescaleMap == 1) crv_val = compress_hyperbolic_power(crv_val, params.ts_s, params.tnCon);
            
            /***************************************************
              Prerequisite color spaces
            --------------------------------------------------*/
            // Simple Cyan-Yellow / Green-Magenta opponent space
            float opp_cy = rgb.x - rgb.z;
            float opp_gm = rgb.y - (rgb.x + rgb.z)/2.0f;
            float ach_d = sqrtf(fmaxf(0.0f, opp_cy*opp_cy + opp_gm*opp_gm))/SQRT3;

            // Smooth ach_d, normalized so 1.0 doesn't change
            ach_d = (1.25f)*compress_toe_quadratic(ach_d, 0.25f, 0);

            // Hue angle, rotated so that red = 0.0
            float hue = fmodf(atan2f(opp_cy, opp_gm) + PI + 1.10714931f, 2.0f*PI);

            // RGB Hue Angles
            float3 ha_rgb = make_float3(
              gauss_window(hue_offset(hue, 0.1f), 0.9f),
              gauss_window(hue_offset(hue, 4.3f), 0.9f),
              gauss_window(hue_offset(hue, 2.3f), 0.9f));

            // CMY Hue Angles
            float3 ha_cmy = make_float3(
              gauss_window(hue_offset(hue, 3.3f), 0.6f),
              gauss_window(hue_offset(hue, 1.3f), 0.6f),
              gauss_window(hue_offset(hue, -1.2f), 0.6f));

            // Purity Compression Range
            float ts_pt_cmp = 1.0f - powf(ts_pt, 1.0f/params.ptRngLow);

            float pt_rng_high_f = fminf(1.0f, ach_d/1.2f);
            pt_rng_high_f *= pt_rng_high_f;
            pt_rng_high_f = params.ptRngHigh < 1.0f ? 1.0f - pt_rng_high_f : pt_rng_high_f;
            ts_pt_cmp = powf(ts_pt_cmp, params.ptRngHigh)*(1.0f - pt_rng_high_f) + ts_pt_cmp*pt_rng_high_f;

            /***************************************************
              Brilliance
            --------------------------------------------------*/
            float brl_f = 1.0f;
            if (params.brlPresetEnable || params.brlUIEnable) {
                brl_f = -params.brlR*ha_rgb.x - params.brlG*ha_rgb.y - params.brlB*ha_rgb.z - params.brlC*ha_cmy.x - params.brlM*ha_cmy.y - params.brlY*ha_cmy.z;
                brl_f = (1.0f - ach_d)*brl_f + 1.0f - brl_f;
                brl_f = softplus(brl_f, 0.25f, -100.0f, 0.0f);
                
                // Limit Brilliance adjustment by tonescale
                float brl_ts = brl_f > 1.0f ? 1.0f - ts_pt : ts_pt;
                float brl_lim = spowf(brl_ts, 1.0f - params.brlRng);
                brl_f = brl_f*brl_lim + 1.0f - brl_lim;
                brl_f = fmaxf(0.0f, fminf(2.0f, brl_f));
            }

            /***************************************************
              Mid-Range Purity
            --------------------------------------------------*/
            float ptm_sc = 1.0f;
            if (params.ptmPresetEnable || params.ptmUIEnable) {
                // Mid Purity Low
                float ptm_ach_d = complement_power(ach_d, params.ptmLowSt);
                ptm_sc = sigmoid_cubic(ptm_ach_d, params.ptmLow*(1.0f - ts_pt));
                
                // Mid Purity High
                ptm_ach_d = complement_power(ach_d, params.ptmHighSt)*(1.0f - ts_pt) + ach_d*ach_d*ts_pt;
                ptm_sc *= sigmoid_cubic(ptm_ach_d, params.ptmHigh*ts_pt);
                ptm_sc = fmaxf(0.0f, ptm_sc);
            }
            
            /***************************************************
              Hue Angle Premultiplication
            --------------------------------------------------*/
            ha_rgb = float3_mul(ha_rgb, ach_d);
            ha_cmy = float3_mul(ha_cmy, (1.5f)*compress_toe_quadratic(ach_d, 0.5f, 0));
            
            /***************************************************
              Hue Contrast R
            --------------------------------------------------*/
            if (params.hcPresetEnable || params.hcUIEnable) {
                float hc_ts = 1.0f - ts_pt;
                float hc_c = (1.0f - ach_d)*hc_ts + ach_d*(1.0f - hc_ts);
                hc_c *= ha_rgb.x;
                hc_ts *= hc_ts;
                float hc_f = params.hcR*(hc_c - 2.0f*hc_c*hc_ts) + 1.0f;
                rgb = make_float3(rgb.x, rgb.y*hc_f, rgb.z*hc_f);
            }

            /***************************************************
              Hue Shift
            --------------------------------------------------*/
            // Hue Shift RGB
            if (params.hsRgbPresetEnable || params.hsRgbUIEnable) {
                float3 hs_rgb = float3_mul(ha_rgb, powf(ts_pt, 1.0f/params.hsRgbRng));
                float3 hsf = make_float3(hs_rgb.x*params.hsR, hs_rgb.y*-params.hsG, hs_rgb.z*-params.hsB);
                hsf = make_float3(hsf.z - hsf.y, hsf.x - hsf.z, hsf.y - hsf.x);
                rgb = float3_add(rgb, hsf);
            }

            // Hue Shift CMY
            if (params.hsCmyPresetEnable || params.hsCmyUIEnable) {
                float3 hs_cmy = float3_mul(ha_cmy, (1.0f - ts_pt));
                float3 hsf = make_float3(hs_cmy.x*-params.hsC, hs_cmy.y*params.hsM, hs_cmy.z*params.hsY);
                hsf = make_float3(hsf.z - hsf.y, hsf.x - hsf.z, hsf.y - hsf.x);
                rgb = float3_add(rgb, hsf);
            }

            /***************************************************
              Module Application
            --------------------------------------------------*/
            // Apply brilliance
            rgb = float3_mul(rgb, brl_f);
            
            // Apply purity compression and mid purity
            ts_pt_cmp *= ptm_sc;
            rgb = float3_add(float3_mul(rgb, ts_pt_cmp), make_float3(1.0f - ts_pt_cmp, 1.0f - ts_pt_cmp, 1.0f - ts_pt_cmp));

            // Inverse Rendering Space
            sat_L = rgb.x*rs_w.x + rgb.y*rs_w.y + rgb.z*rs_w.z;
            rgb = float3_div(float3_sub(float3_mul(make_float3(sat_L, sat_L, sat_L), params.rsSa), rgb), (params.rsSa - 1.0f));

            /***************************************************
              Creative White Point and Output Transform
            --------------------------------------------------*/
            float3 cwp_rgb = rgb;

            // Apply creative whitepoint transform
            cwp_rgb = vdot(cwpMatrix, rgb);

            if (params.displayGamut == 0) { // Rec.709
                // Convert rgb to Rec.709 D65
                float p3ToRec709D65[9] = {
                    1.224940181f, -0.2249402404f, 0.0f,
                    -0.04205697775f, 1.042057037f, -1.4901e-08f,
                    -0.01963755488f, -0.07863604277f, 1.098273635f
                };
                rgb = vdot(p3ToRec709D65, rgb);
                
                if (params.cwp == 0) cwp_rgb = rgb;
            }

            // Mix between Creative Whitepoint and base by tsn
            float cwp_f = powf(tsn, 1.0f - params.cwpRng);
            rgb = float3_add(float3_mul(cwp_rgb, cwp_f), float3_mul(rgb, (1.0f - cwp_f)));

            // Process overlay curve for display gamut and creative whitepoint
            float3 crv_rgb = make_float3(crv_val, crv_val, crv_val);
            float3 crv_rgb_cwp = crv_rgb;
            if (params.tonescaleMap == 1) {
                if (params.displayGamut == 0) { // Rec.709
                    // Apply creative whitepoint transform for overlay
                    crv_rgb_cwp = vdot(cwpMatrix, crv_rgb);
                    // Apply P3 to Rec.709 transform
                    float p3ToRec709[9] = {
                        1.224940181f, -0.2249402404f, 0.0f,
                        -0.04205697775f, 1.042057037f, -1.4901e-08f,
                        -0.01963755488f, -0.07863604277f, 1.098273635f
                    };
                    crv_rgb = vdot(p3ToRec709, crv_rgb);
                    if (params.cwp == 0) crv_rgb_cwp = crv_rgb;
                }
                else if (params.displayGamut >= 1) {
                    // Apply creative whitepoint transform for P3 and Rec.2020
                    crv_rgb_cwp = vdot(cwpMatrix, crv_rgb);
                    if (params.cwp == 0) crv_rgb_cwp = crv_rgb;
                }
                
                // Mix creative whitepoint for overlay curve
                float crv_rgb_cwp_f = powf(crv_val, 1.0f - params.cwpRng);
                crv_rgb = float3_add(float3_mul(crv_rgb_cwp, crv_rgb_cwp_f), float3_mul(crv_rgb, (1.0f - crv_rgb_cwp_f)));
            }
            
            /***************************************************
              Purity Compress Low
            --------------------------------------------------*/
            if (params.ptlPresetEnable || params.ptlUIEnable) {
                float sum0 = softplus(rgb.x, 0.2f, -100.0f, -0.3f) + rgb.y + softplus(rgb.z, 0.2f, -100.0f, -0.3f);
                rgb.x = softplus(rgb.x, 0.04f, -0.3f, 0.0f);
                rgb.y = softplus(rgb.y, 0.06f, -0.3f, 0.0f);
                rgb.z = softplus(rgb.z, 0.01f, -0.05f, 0.0f);

                float ptl_norm = fminf(1.0f, sdivf(sum0, rgb.x + rgb.y + rgb.z));
                rgb = float3_mul(rgb, ptl_norm);
            }

            /***************************************************
              Final Tonescale and Display Transform
            --------------------------------------------------*/
            // Final tonescale adjustments
            tsn *= params.ts_m2;
            tsn = compress_toe_quadratic(tsn, params.tnToe, 0);
            tsn *= params.ts_dsc;
            
            // Track overlay value through final tonescale adjustments
            if (params.tonescaleMap == 1) {
                crv_rgb = float3_mul(crv_rgb, params.ts_m2);
                crv_rgb.x = compress_toe_quadratic(crv_rgb.x, params.tnToe, 0);
                crv_rgb.y = compress_toe_quadratic(crv_rgb.y, params.tnToe, 0);
                crv_rgb.z = compress_toe_quadratic(crv_rgb.z, params.tnToe, 0);
                crv_rgb = float3_mul(crv_rgb, params.ts_dsc);
                // scale to 1.0 = 1000 nits for st2084 PQ
                if (params.eotf == 4) crv_rgb = float3_mul(crv_rgb, 10.0f);
            }
            
            // Return from RGB ratios to final values
            rgb = float3_mul(rgb, tsn);
            
            // Clamp if enabled
            if (params.clamp != 0) {
                rgb = clampf3(rgb, 0.0f, 1.0f);
            }
            
            // Rec.2020 conversion
            if (params.displayGamut == 2) { // Rec.2020
                rgb = clampminf3(rgb, 0.0f);
                float p3ToRec2020[9] = {
                    0.7538330344f, 0.1985973691f, 0.04756959659f,
                    0.04574384897f, 0.9417772198f, 0.01247893122f,
                    -0.001210340355f, 0.0176017173f, 0.9836086231f
                };
                rgb = vdot(p3ToRec2020, rgb);
            }
            
            // ENCODE FOR DISPLAY
            rgb = encode_for_display(rgb, params.eotf);
            
            // Apply EOTF to overlay curve
            if (params.tonescaleMap == 1) {
                if ((params.eotf > 0) && (params.eotf < 4)) {
                    float eotf_p = 2.0f + params.eotf * 0.2f;
                    crv_rgb = spowf3(crv_rgb, 1.0f/eotf_p);
                }
                else if (params.eotf == 4) crv_rgb = eotf_pq(crv_rgb, 1);
                else if (params.eotf == 5) crv_rgb = eotf_hlg(crv_rgb, 1);
            }
            
            // Render overlay curve
            if (params.tonescaleMap == 1) {
                float3 crv_rgb_dst = make_float3(pos.y-crv_rgb.x*res.y, pos.y-crv_rgb.y*res.y, pos.y-crv_rgb.z*res.y);
                float crv_w0 = 0.05f;
                crv_rgb_dst.x = expf(-crv_rgb_dst.x*crv_rgb_dst.x*crv_w0);
                crv_rgb_dst.y = expf(-crv_rgb_dst.y*crv_rgb_dst.y*crv_w0);
                crv_rgb_dst.z = expf(-crv_rgb_dst.z*crv_rgb_dst.z*crv_w0);
                float crv_lm = params.eotf < 4 ? 1.0f : 0.5f;
                crv_rgb_dst = clampf3(crv_rgb_dst, 0.0f, 1.0f);
                rgb = float3_add(float3_mul(rgb, float3_sub(make_float3(1.0f, 1.0f, 1.0f), crv_rgb_dst)), 
                                float3_mul3(float3_mul(make_float3(crv_lm, crv_lm, crv_lm), crv_rgb_dst), crv_rgb_dst));
            }
            
            // Output to buffer
            p_Output[index + 0] = rgb.x;
            p_Output[index + 1] = rgb.y;
            p_Output[index + 2] = rgb.z;
            p_Output[index + 3] = a;
        }
    }
}

// Helper function to populate OpenDRT parameters struct (same as Metal version)
// Continue from where the file was cut off...

OpenDRTParams createOpenDRTParams_OpenCL(
    int p_InGamut, int p_InOetf,
    float p_TnLp, float p_TnGb, float p_PtHdr,
    bool p_Clamp, float p_TnLg, float p_TnCon, float p_TnSh, float p_TnToe, float p_TnOff,
    // UI ENABLE FLAGS (for UI visibility):
    bool p_TnHconUIEnable, bool p_TnLconUIEnable,
    bool p_PtlUIEnable, bool p_PtmUIEnable,
    bool p_BrlUIEnable, bool p_HsRgbUIEnable,
    bool p_HsCmyUIEnable, bool p_HcUIEnable,
    // PRESET ENABLE FLAGS (for execution):
    bool p_TnHconPresetEnable, bool p_TnLconPresetEnable,
    bool p_PtlPresetEnable, bool p_PtmPresetEnable,
    bool p_BrlPresetEnable, bool p_HsRgbPresetEnable,
    bool p_HsCmyPresetEnable, bool p_HcPresetEnable,
    // MODULE PARAMETERS:
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
    // NEW PARAMETERS - Filmic Mode and Advanced Controls
    bool p_FilmicMode, float p_FilmicDynamicRange, int p_FilmicProjectorSim,
    float p_FilmicSourceStops, float p_FilmicTargetStops, float p_FilmicStrength,
    bool p_AdvHueContrast, bool p_TonescaleMap, bool p_DiagnosticsMode, bool p_RgbChipsMode, bool p_BetaFeaturesEnable,
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
    
    // UI Enable Flags (convert bool to int)
    params.tnHconUIEnable = p_TnHconUIEnable ? 1 : 0;
    params.tnLconUIEnable = p_TnLconUIEnable ? 1 : 0;
    params.ptlUIEnable = p_PtlUIEnable ? 1 : 0;
    params.ptmUIEnable = p_PtmUIEnable ? 1 : 0;
    params.brlUIEnable = p_BrlUIEnable ? 1 : 0;
    params.hsRgbUIEnable = p_HsRgbUIEnable ? 1 : 0;
    params.hsCmyUIEnable = p_HsCmyUIEnable ? 1 : 0;
    params.hcUIEnable = p_HcUIEnable ? 1 : 0;
    
    // Preset Enable Flags (convert bool to int)
    params.tnHconPresetEnable = p_TnHconPresetEnable ? 1 : 0;
    params.tnLconPresetEnable = p_TnLconPresetEnable ? 1 : 0;
    params.ptlPresetEnable = p_PtlPresetEnable ? 1 : 0;
    params.ptmPresetEnable = p_PtmPresetEnable ? 1 : 0;
    params.brlPresetEnable = p_BrlPresetEnable ? 1 : 0;
    params.hsRgbPresetEnable = p_HsRgbPresetEnable ? 1 : 0;
    params.hsCmyPresetEnable = p_HsCmyPresetEnable ? 1 : 0;
    params.hcPresetEnable = p_HcPresetEnable ? 1 : 0;
    
    // Module Parameters
    params.tnHcon = p_TnHcon;
    params.tnHconPv = p_TnHconPv;
    params.tnHconSt = p_TnHconSt;
    params.tnLcon = p_TnLcon;
    params.tnLconW = p_TnLconW;
    params.tnLconPc = p_TnLconPc;
    params.cwp = p_Cwp;
    params.cwpRng = p_CwpRng;
    params.rsSa = p_RsSa;
    params.rsRw = p_RsRw;
    params.rsBw = p_RsBw;
    params.ptR = p_PtR;
    params.ptG = p_PtG;
    params.ptB = p_PtB;
    params.ptRngLow = p_PtRngLow;
    params.ptRngHigh = p_PtRngHigh;
    params.ptmLow = p_PtmLow;
    params.ptmLowSt = p_PtmLowSt;
    params.ptmHigh = p_PtmHigh;
    params.ptmHighSt = p_PtmHighSt;
    params.brlR = p_BrlR;
    params.brlG = p_BrlG;
    params.brlB = p_BrlB;
    params.brlC = p_BrlC;
    params.brlM = p_BrlM;
    params.brlY = p_BrlY;
    params.brlRng = p_BrlRng;
    params.hsR = p_HsR;
    params.hsG = p_HsG;
    params.hsB = p_HsB;
    params.hsRgbRng = p_HsRgbRng;
    params.hsC = p_HsC;
    params.hsM = p_HsM;
    params.hsY = p_HsY;
    params.hcR = p_HcR;
    
    // NEW PARAMETERS - Filmic Mode and Advanced Controls (convert bool to int)
    params.filmicMode = p_FilmicMode ? 1 : 0;
    params.filmicDynamicRange = p_FilmicDynamicRange;
    params.filmicProjectorSim = p_FilmicProjectorSim;
    params.filmicSourceStops = p_FilmicSourceStops;
    params.filmicTargetStops = p_FilmicTargetStops;
    params.filmicStrength = p_FilmicStrength;
    params.advHueContrast = p_AdvHueContrast ? 1 : 0;
    params.tonescaleMap = p_TonescaleMap ? 1 : 0;
    params.diagnosticsMode = p_DiagnosticsMode ? 1 : 0;
    params.rgbChipsMode = p_RgbChipsMode ? 1 : 0;
    params.betaFeaturesEnable = p_BetaFeaturesEnable ? 1 : 0;
    params.displayGamut = p_DisplayGamut;
    params.eotf = p_Eotf;
    
    // Calculate derived parameters (same as Metal version)
    // Tonescale Parameters
    float m1 = powf(2.0f, params.tnLp);
    float m2 = powf(2.0f, params.tnGb);
    float g = 1.12f;
    float c = params.tnLg / 100.0f;
    float n = 0.18f;
    float k = fmaxf(1e-10f, n * m2);
    float ip = 2.0f; // Intermediate peak
    float s0 = n / k;
    float s1 = m1 * (1.0f - s0);
    float cp = c * powf(ip * m1, -g);
    float w2 = m1 * (1.0f + c * powf(m1, g));
    float s = (w2 - m1) / w2;
    float t = (ip * m1 - m1) / (ip * m1 - ip * m1 * s);
    float nd = params.tnSh;
    
    // Store derived values in params
    params.ts_m1 = m1;
    params.ts_m2 = m2;
    params.ts_g = g;
    params.ts_c = c;
    params.ts_n = n;
    params.ts_k = k;
    params.ts_ip = ip;
    params.ts_s0 = s0;
    params.ts_s1 = s1;
    params.ts_cp = cp;
    params.ts_w2 = w2;
    params.ts_s = s;
    params.ts_t = t;
    params.ts_nd = nd;
    params.ts_x0 = 0.18f;
    params.ts_s1 = 1.25f;
    params.ts_dsc = 1.0f;
    
    return params;
}

// External interface function for OpenCL kernel execution
void RunOpenCLKernel(int p_Width, int p_Height, const float* p_Input, float* p_Output,
                     int p_InGamut, int p_InOetf,
                     float p_TnLp, float p_TnGb, float p_PtHdr,
                     bool p_Clamp, float p_TnLg, float p_TnCon, float p_TnSh, float p_TnToe, float p_TnOff,
                     // UI ENABLE FLAGS:
                     bool p_TnHconUIEnable, bool p_TnLconUIEnable,
                     bool p_PtlUIEnable, bool p_PtmUIEnable,
                     bool p_BrlUIEnable, bool p_HsRgbUIEnable,
                     bool p_HsCmyUIEnable, bool p_HcUIEnable,
                     // PRESET ENABLE FLAGS:
                     bool p_TnHconPresetEnable, bool p_TnLconPresetEnable,
                     bool p_PtlPresetEnable, bool p_PtmPresetEnable,
                     bool p_BrlPresetEnable, bool p_HsRgbPresetEnable,
                     bool p_HsCmyPresetEnable, bool p_HcPresetEnable,
                     // MODULE PARAMETERS:
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
                     // NEW PARAMETERS:
                     bool p_FilmicMode, float p_FilmicDynamicRange, int p_FilmicProjectorSim,
                     float p_FilmicSourceStops, float p_FilmicTargetStops, float p_FilmicStrength,
                     bool p_AdvHueContrast, bool p_TonescaleMap, bool p_DiagnosticsMode, bool p_RgbChipsMode, bool p_BetaFeaturesEnable,
                     int p_DisplayGamut, int p_Eotf)
{
    // Create parameters struct
    OpenDRTParams params = createOpenDRTParams_OpenCL(
        p_InGamut, p_InOetf,
        p_TnLp, p_TnGb, p_PtHdr,
        p_Clamp, p_TnLg, p_TnCon, p_TnSh, p_TnToe, p_TnOff,
        p_TnHconUIEnable, p_TnLconUIEnable,
        p_PtlUIEnable, p_PtmUIEnable,
        p_BrlUIEnable, p_HsRgbUIEnable,
        p_HsCmyUIEnable, p_HcUIEnable,
        p_TnHconPresetEnable, p_TnLconPresetEnable,
        p_PtlPresetEnable, p_PtmPresetEnable,
        p_BrlPresetEnable, p_HsRgbPresetEnable,
        p_HsCmyPresetEnable, p_HcPresetEnable,
        p_TnHcon, p_TnHconPv, p_TnHconSt,
        p_TnLcon, p_TnLconW, p_TnLconPc,
        p_Cwp, p_CwpRng,
        p_RsSa, p_RsRw, p_RsBw,
        p_PtR, p_PtG, p_PtB, p_PtRngLow, p_PtRngHigh,
        p_PtmLow, p_PtmLowSt, p_PtmHigh, p_PtmHighSt,
        p_BrlR, p_BrlG, p_BrlB,
        p_BrlC, p_BrlM, p_BrlY, p_BrlRng,
        p_HsR, p_HsG, p_HsB, p_HsRgbRng,
        p_HsC, p_HsM, p_HsY,
        p_HcR,
        p_FilmicMode, p_FilmicDynamicRange, p_FilmicProjectorSim,
        p_FilmicSourceStops, p_FilmicTargetStops, p_FilmicStrength,
        p_AdvHueContrast, p_TonescaleMap, p_DiagnosticsMode, p_RgbChipsMode, p_BetaFeaturesEnable,
        p_DisplayGamut, p_Eotf
    );
    
    // Execute the kernel (CPU-based implementation)
    OpenDRTKernel_OpenCL(p_Width, p_Height, p_Input, p_Output, params);
}