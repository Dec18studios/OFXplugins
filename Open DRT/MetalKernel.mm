#import <Metal/Metal.h>
#include "OpenDRTParams.h"
#include "OpenDRTPresets.h"
#include <unordered_map>
#include <mutex>

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
"    // Advanced Hue Contrast Parameters\n" \
"    float advHcR;\n" \
"    float advHcG;\n" \
"    float advHcB;\n" \
"    float advHcC;\n" \
"    float advHcM;\n" \
"    float advHcY;\n" \
"    float advHcPower;\n" \
"    // NEW PARAMETERS - Filmic Mode and Advanced Controls\n" \
"    int filmicMode;\n" \
"    float filmicDynamicRange;\n" \
"    int filmicProjectorSim;\n" \
"    float filmicSourceStops;\n" \
"    float filmicTargetStops;\n" \
"    float filmicStrength;\n" \
"    int advHueContrast;\n" \
"    int tonescaleMap;\n" \
"    int diagnosticsMode;\n" \
"    int rgbChipsMode;\n" \
"    int betaFeaturesEnable;\n" \
"    int displayGamut;\n" \
"    int eotf;\n" \
"    // Matrix data (3x3 = 9 floats each)\n" \
"    float inputMatrix[9];\n" \
"    float outputMatrix[9];\n" \
"    float cwpMatrix[9];\n" \
"    float xyzToP3Matrix[9];     // XYZ to P3-D65 working space\n" \
"    float p3ToRec709Matrix[9];  // P3 to Rec.709 D65 (for display_gamut==0)\n" \
"    // Transfer Function Parameters\n" \
"    float oetfParams[8];\n" \
"    float eotfParams[8];\n" \
"    int oetfType;\n" \
"    int eotfType;\n" \
"    // PRECALCULATED TONESCALE CONSTANTS\n" \
"    float ts_x1;\n" \
"    float ts_y1;\n" \
"    float ts_x0;\n" \
"    float ts_y0;\n" \
"    float ts_s0;\n" \
"    float ts_s10;\n" \
"    float ts_m1;\n" \
"    float ts_m2;\n" \
"    float ts_s;\n" \
"    float ts_dsc;\n" \
"    float pt_cmp_Lf;\n" \
"    float s_Lp100;\n" \
"    float ts_s1;\n" \
"    \n" \
"    // MODULE ENABLE FLAGS\n" \
"    int tnHconEnable;\n" \
"    int tnLconEnable;\n" \
"    int ptlEnable;\n" \
"    int ptmEnable;\n" \
"    int brlEnable;\n" \
"    int hsRgbEnable;\n" \
"    int hsCmyEnable;\n" \
"    int hcEnable;\n" \
"    \n" \
"    // UI ENABLE FLAGS (for UI visibility)\n" \
"    int tnHconUIEnable;\n" \
"    int tnLconUIEnable;\n" \
"    int ptlUIEnable;\n" \
"    int ptmUIEnable;\n" \
"    int brlUIEnable;\n" \
"    int hsRgbUIEnable;\n" \
"    int hsCmyUIEnable;\n" \
"    int hcUIEnable;\n" \
"    \n" \
"    // PRESET ENABLE FLAGS (for execution)\n" \
"    int tnHconPresetEnable;\n" \
"    int tnLconPresetEnable;\n" \
"    int ptlPresetEnable;\n" \
"    int ptmPresetEnable;\n" \
"    int brlPresetEnable;\n" \
"    int hsRgbPresetEnable;\n" \
"    int hsCmyPresetEnable;\n" \
"    int hcPresetEnable;\n" \
"};\n" \
"\n" \
"// Matrix multiplication helper function\n" \
"float3 vdot(float3x3 m, float3 v) {\n" \
"    return float3(m[0].x*v.x + m[1].x*v.y + m[2].x*v.z,\n" \
"                  m[0].y*v.x + m[1].y*v.y + m[2].y*v.z,\n" \
"                  m[0].z*v.x + m[1].z*v.y + m[2].z*v.z);\n" \
"}\n" \
"\n" \
"// Helper to convert array to matrix (CORRECTED for column-major)\n" \
"float3x3 arrayToMatrix(constant float* arr) {\n" \
"    return float3x3(\n" \
"        float3(arr[0], arr[3], arr[6]),\n" \
"        float3(arr[1], arr[4], arr[7]),\n" \
"        float3(arr[2], arr[5], arr[8])\n" \
"    );\n" \
"}\n" \
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
"float clampf(float a, float mn, float mx) {\n" \
"    return min(max(a, mn), mx);\n" \
"}\n" \
"\n" \
"float3 clampf3(float3 a, float mn, float mx) {\n" \
"    return float3(clampf(a.x, mn, mx), clampf(a.y, mn, mx), clampf(a.z, mn, mx));\n" \
"}\n" \
"\n" \
"float3 clampminf3(float3 a, float mn) {\n" \
"    return float3(max(a.x, mn), max(a.y, mn), max(a.z, mn));\n" \
"}\n" \
"\n" \
"// Additional math helper functions\n" \
"float fmaxf3(float3 a) {\n" \
"    return max(a.x, max(a.y, a.z));\n" \
"}\n" \
"\n" \
"float fminf3(float3 a) {\n" \
"    return min(a.x, min(a.y, a.z));\n" \
"}\n" \
"\n" \
"float hypotf3(float3 a) {\n" \
"    return sqrt(a.x*a.x + a.y*a.y + a.z*a.z);\n" \
"}\n" \
"\n" \
"float compress_toe_cubic(float x, float m, float w, int inv) {\n" \
"    // Cubic toe compression function\n" \
"    // https://www.desmos.com/calculator/ubgteikoke\n" \
"    if (m == 1.0f) return x;\n" \
"    float x2 = x * x;\n" \
"    if (inv == 0) {\n" \
"        return x * (x2 + m * w) / (x2 + w);\n" \
"    } else {\n" \
"        float p0 = x2 - 3.0f * m * w;\n" \
"        float p1 = 2.0f * x2 + 27.0f * w - 9.0f * m * w;\n" \
"        float p2 = pow(sqrt(x2 * p1 * p1 - 4 * p0 * p0 * p0) / 2.0f + x * p1 / 2.0f, 1.0f / 3.0f);\n" \
"        return p0 / (3.0f * p2) + p2 / 3.0f + x / 3.0f;\n" \
"    }\n" \
"}\n" \
"\n" \
"float compress_hyperbolic_power(float x, float s, float p) {\n" \
"    // Simple hyperbolic compression function\n" \
"    // https://www.desmos.com/calculator/ofwtcmzc3w\n" \
"    return spowf(x / (x + s), p);\n" \
"}\n" \
"\n" \
"float contrast_high(float x, float p, float pv, float pv_lx, int inv) {\n" \
"    // High exposure adjustment with linear extension\n" \
"    // https://www.desmos.com/calculator/etjgwyrgad\n" \
"    const float x0 = 0.18f * pow(2.0f, pv);\n" \
"    if (x < x0 || p == 1.0f) return x;\n" \
"\n" \
"    const float o = x0 - x0 / p;\n" \
"    const float s0 = pow(x0, 1.0f - p) / p;\n" \
"    const float x1 = x0 * pow(2.0f, pv_lx);\n" \
"    const float k1 = p * s0 * pow(x1, p) / x1;\n" \
"    const float y1 = s0 * pow(x1, p) + o;\n" \
"    if (inv == 1)\n" \
"        return x > y1 ? (x - y1) / k1 + x1 : pow((x - o) / s0, 1.0f / p);\n" \
"    else\n" \
"        return x > x1 ? k1 * (x - x1) + y1 : s0 * pow(x, p) + o;\n" \
"}\n" \
"\n" \
"float gamma_contrast(float x, float gamma, float mid_gray) {\n" \
"    return mid_gray * pow(max(0.0f, x / mid_gray), gamma);\n" \
"}\n" \
"\n" \
"float gauss_window(float x, float w) {\n" \
"    // Simple gaussian window\n" \
"    // https://www.desmos.com/calculator/vhr9hstlyk\n" \
"    x /= w;\n" \
"    return exp(-x * x);\n" \
"}\n" \
"\n" \
"float hue_offset(float h, float o) {\n" \
"    // Offset hue maintaining 0-2*pi range with modulo\n" \
"    return fmod(h - o + PI, 2.0f * PI) - PI;\n" \
"}\n" \
"\n" \
"float compress_toe_quadratic(float x, float toe, int inv) {\n" \
"    // Quadratic toe compress function\n" \
"    // https://www.desmos.com/calculator/skk8ahmnws\n" \
"    if (toe == 0.0f) return x;\n" \
"    if (inv == 0) {\n" \
"        return spowf(x, 2.0f) / (x + toe);\n" \
"    } else {\n" \
"        return (x + sqrt(x * (4.0f * toe + x))) / 2.0f;\n" \
"    }\n" \
"}\n" \
"\n" \
"float complement_power(float x, float p) {\n" \
"    return 1.0f - spowf(1.0f - x, 1.0f/p);\n" \
"}\n" \
"\n" \
"float sigmoid_cubic(float x, float s) {\n" \
"    // Simple cubic sigmoid: https://www.desmos.com/calculator/hzgib42en6\n" \
"    if (x < 0.0f || x > 1.0f) return 1.0f;\n" \
"    return 1.0f + s*(1.0f - 3.0f*x*x + 2.0f*x*x*x);\n" \
"}\n" \
"\n" \
"float softplus(float x, float s, float x0, float y0) {\n" \
"    if (x > 10.0f*s + y0 || s < 1e-3f) return x;\n" \
"    float m = 1.0f;\n" \
"     if (fabs(y0) > 1e-6f) m = exp(y0/s);\n" \
"    m -= exp(x0/s);\n" \
"    return s*log(max(0.0f, m + exp(x/s)));\n" \
"}\n" \
"\n" \
"// HARDCODED OETF FUNCTIONS (matching DCTL exactly)\n" \
"float oetf_davinci_intermediate(float x) {\n" \
"    return x <= 0.02740668f ? x/10.44426855f : exp2(x/0.07329248f - 7.0f) - 0.0075f;\n" \
"}\n" \
"\n" \
"float oetf_filmlight_tlog(float x) {\n" \
"    return x < 0.075f ? (x-0.075f)/16.184376489665897f : exp((x - 0.5520126568606655f)/0.09232902596577353f) - 0.0057048244042473785f;\n" \
"}\n" \
"\n" \
"float oetf_acescct(float x) {\n" \
"    return x <= 0.155251141552511f ? (x - 0.0729055341958355f)/10.5402377416545f : exp2(x*17.52f - 9.72f);\n" \
"}\n" \
"\n" \
"float oetf_arri_logc3(float x) {\n" \
"    return x < 5.367655f*0.010591f + 0.092809f ? (x - 0.092809f)/5.367655f : (pow(10.0f, (x - 0.385537f)/0.247190f) - 0.052272f)/5.555556f;\n" \
"}\n" \
"\n" \
"float oetf_arri_logc4(float x) {\n" \
"    return x < -0.7774983977293537f ? x*0.3033266726886969f - 0.7774983977293537f : (exp2(14.0f*(x - 0.09286412512218964f)/0.9071358748778103f + 6.0f) - 64.0f)/2231.8263090676883f;\n" \
"}\n" \
"\n" \
"float oetf_red_log3g10(float x) {\n" \
"    return x < 0.0f ? (x/15.1927f) - 0.01f : (pow(10.0f, x/0.224282f) - 1.0f)/155.975327f - 0.01f;\n" \
"}\n" \
"\n" \
"float oetf_panasonic_vlog(float x) {\n" \
"    return x < 0.181f ? (x - 0.125f)/5.6f : pow(10.0f, (x - 0.598206f)/0.241514f) - 0.00873f;\n" \
"}\n" \
"\n" \
"float oetf_sony_slog3(float x) {\n" \
"    return x < 171.2102946929f/1023.0f ? (x*1023.0f - 95.0f)*0.01125f/(171.2102946929f - 95.0f) : (pow(10.0f, ((x*1023.0f - 420.0f)/261.5f))*(0.18f + 0.01f) - 0.01f);\n" \
"}\n" \
"\n" \
"float oetf_fujifilm_flog2(float x) {\n" \
"    return x < 0.100686685370811f ? (x - 0.092864f)/8.799461f : (pow(10.0f, ((x - 0.384316f)/0.245281f))/5.555556f - 0.064829f/5.555556f);\n" \
"}\n" \
"\n" \
"// SIMPLIFIED OETF DISPATCHER\n" \
"float apply_oetf(float x, int type) {\n" \
"    switch(type) {\n"
"        case 0: return x; // Linear\n"
"        case 1: return oetf_davinci_intermediate(x);\n"
"        case 2: return oetf_filmlight_tlog(x);\n"
"        case 3: return oetf_acescct(x);\n"
"        case 4: return oetf_arri_logc3(x);\n"
"        case 5: return oetf_arri_logc4(x);\n"
"        case 6: return oetf_red_log3g10(x);\n"
"        case 7: return oetf_panasonic_vlog(x);\n"
"        case 8: return oetf_sony_slog3(x);\n"
"        case 9: return oetf_fujifilm_flog2(x);\n"
"        default: return x;\n"
"    }\n"
"}\n"
"// HARDCODED EOTF FUNCTIONS (matching DCTL exactly)\n" \
"float3 eotf_hlg(float3 rgb, int inverse) {\n" \
"    if (inverse == 1) {\n" \
"        float Yd = 0.2627f*rgb.x + 0.6780f*rgb.y + 0.0593f*rgb.z;\n" \
"        rgb = rgb*pow(Yd, (1.0f - 1.2f)/1.2f);\n" \
"        rgb.x = rgb.x <= 1.0f/12.0f ? sqrt(3.0f*rgb.x) : 0.17883277f*log(12.0f*rgb.x - 0.28466892f) + 0.55991073f;\n" \
"        rgb.y = rgb.y <= 1.0f/12.0f ? sqrt(3.0f*rgb.y) : 0.17883277f*log(12.0f*rgb.y - 0.28466892f) + 0.55991073f;\n" \
"        rgb.z = rgb.z <= 1.0f/12.0f ? sqrt(3.0f*rgb.z) : 0.17883277f*log(12.0f*rgb.z - 0.28466892f) + 0.55991073f;\n" \
"    } else {\n" \
"        rgb.x = rgb.x <= 0.5f ? rgb.x*rgb.x/3.0f : (exp((rgb.x - 0.55991073f)/0.17883277f) + 0.28466892f)/12.0f;\n" \
"        rgb.y = rgb.y <= 0.5f ? rgb.y*rgb.y/3.0f : (exp((rgb.y - 0.55991073f)/0.17883277f) + 0.28466892f)/12.0f;\n" \
"        rgb.z = rgb.z <= 0.5f ? rgb.z*rgb.z/3.0f : (exp((rgb.z - 0.55991073f)/0.17883277f) + 0.28466892f)/12.0f;\n" \
"        float Ys = 0.2627f*rgb.x + 0.6780f*rgb.y + 0.0593f*rgb.z;\n" \
"        rgb = rgb*pow(Ys, 1.2f - 1.0f);\n" \
"    }\n" \
"    return rgb;\n" \
"}\n" \
"\n" \
"float3 eotf_pq(float3 rgb, int inverse) {\n" \
"    const float m1 = 2610.0f/16384.0f;\n" \
"    const float m2 = 2523.0f/32.0f;\n" \
"    const float c1 = 107.0f/128.0f;\n" \
"    const float c2 = 2413.0f/128.0f;\n" \
"    const float c3 = 2392.0f/128.0f;\n" \
"\n" \
"    if (inverse == 1) {\n" \
"        rgb = pow(rgb, m1);\n" \
"        rgb = pow((c1 + c2*rgb)/(1.0f + c3*rgb), m2);\n" \
"    } else {\n" \
"        rgb = pow(rgb, 1.0f/m2);\n" \
"        rgb = pow((rgb - c1)/(c2 - c3*rgb), 1.0f/m1);\n" \
"    }\n" \
"    return rgb;\n" \
"}\n" \
"\n" \
"// SIMPLIFIED EOTF DISPATCHER\n" \
"float apply_eotf(float x, int type, int inverse) {\n" \
"    switch(type) {\n" \
"        case 0: return x; // Linear\n" \
"        case 1: return inverse ? pow(max(0.0f, x), 1.0f/2.2f) : pow(max(0.0f, x), 2.2f); // 2.2 Power\n" \
"        case 2: return inverse ? pow(max(0.0f, x), 1.0f/2.4f) : pow(max(0.0f, x), 2.4f); // 2.4 Power\n" \
"        case 3: return inverse ? pow(max(0.0f, x), 1.0f/2.6f) : pow(max(0.0f, x), 2.6f); // 2.6 Power\n" \
"        case 4: // PQ - handled separately\n" \
"            return x;\n" \
"        case 5: // HLG - handled separately\n" \
"            return x;\n" \
"        default: return x;\n" \
"    }\n" \
"}\n" \
"\n" \
"// Updated display encoding function\n" \
"float3 encode_for_display(float3 rgb, int eotfType) {\n"
"    switch(eotfType) {\n"
"        case 4: return eotf_pq(rgb, 1); // PQ inverse\n"
"        case 5: return eotf_hlg(rgb, 1); // HLG inverse\n"
"        default:\n"
"            rgb.x = apply_eotf(rgb.x, eotfType, 1);\n"  // Remove eotfParams parameter
"            rgb.y = apply_eotf(rgb.y, eotfType, 1);\n"
"            rgb.z = apply_eotf(rgb.z, eotfType, 1);\n"
"            return rgb;\n"
"    }\n"
"}\n"
"\n" \
"\n" \
"// LINEARIZE FUNCTION (matching DCTL)\n" \
"float3 linearize(float3 rgb, int tf) {\n"
"    if (tf == 0) { // Linear\n"
"        return rgb;\n"
"    } else {\n"
"        rgb.x = apply_oetf(rgb.x, tf);\n"  
"        rgb.y = apply_oetf(rgb.y, tf);\n"
"        rgb.z = apply_oetf(rgb.z, tf);\n"
"        return rgb;\n"
"    }\n"
"}\n"
"// HARDCODED INPUT MATRIX FUNCTIONS\n" \
"float3x3 getInputMatrix(int gamut) {\n" \
"    switch(gamut) {\n" \
"        case 0: // XYZ (Identity)\n" \
"            return float3x3(\n" \
"                float3(1.0f, 0.0f, 0.0f),\n" \
"                float3(0.0f, 1.0f, 0.0f),\n" \
"                float3(0.0f, 0.0f, 1.0f)\n" \
"            );\n" \
"        case 1: // AP0 to XYZ\n" \
"            return float3x3(\n" \
"                float3(0.93863094875f, 0.338093594922f, 0.000723121511f),\n" \
"                float3(-0.00574192055f, 0.727213902811f, 0.000818441849f),\n" \
"                float3(0.017566898852f, -0.065307497733f, 1.0875161874f)\n" \
"            );\n" \
"        case 2: // AP1 to XYZ\n" \
"            return float3x3(\n" \
"                float3(0.652418717672f, 0.268064059194f, -0.00546992851f),\n" \
"                float3(0.127179925538f, 0.672464478993f, 0.005182799977f),\n" \
"                float3(0.170857283842f, 0.059471461813f, 1.08934487929f)\n" \
"            );\n" \
"        case 3: // P3-D65 to XYZ\n" \
"            return float3x3(\n" \
"                float3(0.486571133137f, 0.228974640369f, 0.0f),\n" \
"                float3(0.265667706728f, 0.691738605499f, 0.045113388449f),\n" \
"                float3(0.198217317462f, 0.079286918044f, 1.043944478035f)\n" \
"            );\n" \
"        case 4: // Rec.2020 to XYZ\n" \
"            return float3x3(\n" \
"                float3(0.636958122253f, 0.262700229883f, 0.0f),\n" \
"                float3(0.144616916776f, 0.677998125553f, 0.028072696179f),\n" \
"                float3(0.168880969286f, 0.059301715344f, 1.060985088348f)\n" \
"            );\n" \
"        case 5: // Rec.709 to XYZ\n" \
"            return float3x3(\n" \
"                float3(0.412390917540f, 0.212639078498f, 0.019330825657f),\n" \
"                float3(0.357584357262f, 0.715168714523f, 0.119194783270f),\n" \
"                float3(0.180480793118f, 0.072192311287f, 0.950532138348f)\n" \
"            );\n" \
"        case 6: // Arri Wide Gamut 3 to XYZ\n" \
"            return float3x3(\n" \
"                float3(0.638007619284f, 0.291953779f, 0.002798279032f),\n" \
"                float3(0.214703856337f, 0.823841041511f, -0.067034235689f),\n" \
"                float3(0.097744451431f, -0.11579482051f, 1.15329370742f)\n" \
"            );\n" \
"        case 7: // Arri Wide Gamut 4 to XYZ\n" \
"            return float3x3(\n" \
"                float3(0.704858320407f, 0.254524176404f, 0.0f),\n" \
"                float3(0.12976029517f, 0.781477732712f, 0.0f),\n" \
"                float3(0.115837311474f, -0.036001909116f, 1.08905775076f)\n" \
"            );\n" \
"        case 8: // Red Wide Gamut RGB to XYZ\n" \
"            return float3x3(\n" \
"                float3(0.735275208950f, 0.286694079638f, -0.079680845141f),\n" \
"                float3(0.068609409034f, 0.842979073524f, -0.347343206406f),\n" \
"                float3(0.146571278572f, -0.129673242569f, 1.516081929207f)\n" \
"            );\n" \
"        case 9: // Sony S-Gamut3 to XYZ\n" \
"            return float3x3(\n" \
"                float3(0.706482713192f, 0.270979670813f, -0.009677845386f),\n" \
"                float3(0.128801049791f, 0.786606411221f, 0.004600037493f),\n" \
"                float3(0.115172164069f, -0.057586082034f, 1.09413555865f)\n" \
"            );\n" \
"        case 10: // Sony S-Gamut3.Cine to XYZ\n" \
"            return float3x3(\n" \
"                float3(0.599083920758f, 0.215075820116f, -0.032065849545f),\n" \
"                float3(0.248925516115f, 0.885068501744f, -0.027658390679f),\n" \
"                float3(0.102446490178f, -0.100144321859f, 1.14878199098f)\n" \
"            );\n" \
"        case 11: // Panasonic V-Gamut to XYZ\n" \
"            return float3x3(\n" \
"                float3(0.679644469878f, 0.26068555009f, -0.009310198218f),\n" \
"                float3(0.15221141244f, 0.77489446333f, -0.004612467044f),\n" \
"                float3(0.118600044733f, -0.03558001342f, 1.10298041602f)\n" \
"            );\n" \
"        case 12: // Blackmagic Wide Gamut to XYZ\n" \
"            return float3x3(\n" \
"                float3(0.606538414955f, 0.267992943525f, -0.029442556202f),\n" \
"                float3(0.220412746072f, 0.832748472691f, -0.086612440646f),\n" \
"                float3(0.123504832387f, -0.100741356611f, 1.205112814903f)\n" \
"            );\n" \
"        case 13: // Filmlight E-Gamut to XYZ\n" \
"            return float3x3(\n" \
"                float3(0.705396831036f, 0.280130714178f, -0.103781513870f),\n" \
"                float3(0.164041340351f, 0.820206701756f, -0.072907261550f),\n" \
"                float3(0.081017754972f, -0.100337378681f, 1.265746593475f)\n" \
"            );\n" \
"        case 14: // Filmlight E-Gamut2 to XYZ\n" \
"            return float3x3(\n" \
"                float3(0.736477700184f, 0.275069984406f, -0.124225154248f),\n" \
"                float3(0.130739651087f, 0.828017790216f, -0.087159767391f),\n" \
"                float3(0.083238575781f, -0.103087774621f, 1.3004426724f)\n" \
"            );\n" \
"        case 15: // DaVinci Wide Gamut to XYZ\n" \
"            return float3x3(\n" \
"                float3(0.700622320175f, 0.274118483067f, -0.098962903023f),\n" \
"                float3(0.148774802685f, 0.873631775379f, -0.137895315886f),\n" \
"                float3(0.101058728993f, -0.147750422359f, 1.325916051865f)\n" \
"            );\n" \
"        default:\n" \
"            return float3x3(\n" \
"                float3(1.0f, 0.0f, 0.0f),\n" \
"                float3(0.0f, 1.0f, 0.0f),\n" \
"                float3(0.0f, 0.0f, 1.0f)\n" \
"            );\n" \
"    }\n" \
"}\n" \
"\n" \
"float3x3 getOutputMatrix(int displayGamut) {\n" \
"    switch(displayGamut) {\n" \
"        case 0: // P3 to Rec.709\n" \
"            return float3x3(\n" \
"                float3(1.224940181f, -0.04205697775f, -0.01963755488f),\n" \
"                float3(-0.2249402404f, 1.042057037f, -0.07863604277f),\n" \
"                float3(0.0f, -1.4901e-08f, 1.098273635f)\n" \
"            );\n" \
"        case 1: // P3 Identity\n" \
"            return float3x3(\n" \
"                float3(1.0f, 0.0f, 0.0f),\n" \
"                float3(0.0f, 1.0f, 0.0f),\n" \
"                float3(0.0f, 0.0f, 1.0f)\n" \
"            );\n" \
"        case 2: // P3 to Rec.2020\n" \
"            return float3x3(\n" \
"                float3(0.7538330344f, 0.04574384897f, -0.001210340355f),\n" \
"                float3(0.1985973691f, 0.9417772198f, 0.0176017173f),\n" \
"                float3(0.04756959659f, 0.01247893122f, 0.9836086231f)\n" \
"            );\n" \
"        default:\n" \
"            return float3x3(\n" \
"                float3(1.0f, 0.0f, 0.0f),\n" \
"                float3(0.0f, 1.0f, 0.0f),\n" \
"                float3(0.0f, 0.0f, 1.0f)\n" \
"            );\n" \
"    }\n" \
"}\n" \
"\n" \
"float3x3 getCreativeWhitepointMatrix(int displayGamut, int cwp) {\n" \
"    if (displayGamut == 0) { // Rec.709\n" \
"        switch(cwp) {\n" \
"            case 1: // D60\n" \
"                return float3x3(\n" \
"                    float3(1.189986856f, -0.04168263635f, -0.01937995127f),\n" \
"                    float3(-0.192168414f, 0.9927757018f, -0.07933006919f),\n" \
"                    float3(0.002185496045f, -5.5660878e-05f, 0.9734397041f)\n" \
"                );\n" \
"            case 2: // D55\n" \
"                return float3x3(\n" \
"                    float3(1.149327514f, -0.0412590771f, -0.01900949528f),\n" \
"                    float3(-0.1536910745f, 0.9351717477f, -0.07928282823f),\n" \
"                    float3(0.004366526746f, -0.000116126221f, 0.8437884317f)\n" \
"                );\n" \
"            case 3: // D50\n" \
"                return float3x3(\n" \
"                    float3(1.103807322f, -0.04079386701f, -0.01854055914f),\n" \
"                    float3(-0.1103425121f, 0.8704694227f, -0.07857582481f),\n" \
"                    float3(0.006531676079f, -0.000180522628f, 0.7105498861f)\n" \
"                );\n" \
"            default: // D65 (Identity)\n" \
"                return float3x3(\n" \
"                    float3(1.0f, 0.0f, 0.0f),\n" \
"                    float3(0.0f, 1.0f, 0.0f),\n" \
"                    float3(0.0f, 0.0f, 1.0f)\n" \
"                );\n" \
"        }\n" \
"    } else { // P3 and Rec.2020\n" \
"        switch(cwp) {\n" \
"            case 1: // D60\n" \
"                return float3x3(\n" \
"                    float3(0.979832881f, -0.000805359793f, -0.000338382322f),\n" \
"                    float3(0.01836378979f, 0.9618000331f, -0.003671835795f),\n" \
"                    float3(0.001803284786f, 1.8876121e-05f, 0.894139105f)\n" \
"                );\n" \
"            case 2: // D55\n" \
"                return float3x3(\n" \
"                    float3(0.9559790976f, -0.001771929896f, -0.000674760809f),\n" \
"                    float3(0.0403850003f, 0.9163058305f, -0.0072466358f),\n" \
"                    float3(0.003639287409f, 3.3300759e-05f, 0.7831189153f)\n" \
"                );\n" \
"            case 3: // D50\n" \
"                return float3x3(\n" \
"                    float3(0.9287127388f, -0.002887159176f, -0.001009551548f),\n" \
"                    float3(0.06578032793f, 0.8640709228f, -0.01073503317f),\n" \
"                    float3(0.005506708345f, 4.3593718e-05f, 0.6672692039f)\n" \
"                );\n" \
"            default: // D65 (Identity)\n" \
"                return float3x3(\n" \
"                    float3(1.0f, 0.0f, 0.0f),\n" \
"                    float3(0.0f, 1.0f, 0.0f),\n" \
"                    float3(0.0f, 0.0f, 1.0f)\n" \
"                );\n" \
"        }\n" \
"    }\n" \
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
"        /***************************************************\n" \
"         setup and extraction\n" \
"        --------------------------------------------------*/\n" \
"        // Extract RGBA values\n" \
"        float3 rgb = float3(p_Input[index + 0], p_Input[index + 1], p_Input[index + 2]);\n" \
"        float a = p_Input[index + 3];\n" \
"        \n" \
"        // If diagnostics mode is enabled and in ramp area, set input to ramp value\n" \
"        if (params.diagnosticsMode == 1 && id.y < 100) {\n" \
"            float ramp = (float)id.x / (float)(p_Width - 1);\n" \
"            rgb = float3(ramp, ramp, ramp);\n" \
"        }\n" \
"        \n" \
"        // If RGB chips mode is enabled, create RGB test pattern\n" \
"        if (params.rgbChipsMode == 1) {\n" \
"            float ramp = (float)id.x / (float)(p_Width - 1);\n" \
"            int band = id.y * 7 / p_Height;\n" \
"            \n" \
"            switch (band) {\n" \
"                case 0: rgb = float3(ramp, 0.0f, 0.0f); break;     // Red\n" \
"                case 1: rgb = float3(ramp, ramp, 0.0f); break;     // Yellow\n" \
"                case 2: rgb = float3(0.0f, ramp, 0.0f); break;     // Green\n" \
"                case 3: rgb = float3(0.0f, ramp, ramp); break;     // Cyan\n" \
"                case 4: rgb = float3(0.0f, 0.0f, ramp); break;     // Blue\n" \
"                case 5: rgb = float3(ramp, 0.0f, ramp); break;     // Magenta\n" \
"                case 6: rgb = float3(ramp, ramp, ramp); break;    // White/Gray\n" \
"                default: rgb = float3(ramp, ramp, ramp); break;    // White/Gray\n" \
"            }\n" \
"        }\n" \
"        \n" \
"        \n" \
"       rgb = linearize(rgb, params.inOetf);\n" \
"        \n" \
"        // Load dynamic matrices using switch functions\n" \
"        float3x3 inputMatrix = getInputMatrix(params.inGamut);\n" \
"        float3x3 outputMatrix = getOutputMatrix(params.displayGamut);\n" \
"        float3x3 cwpMatrix = getCreativeWhitepointMatrix(params.displayGamut, params.cwp);\n" \
"        \n" \
"        // Apply input matrix transform\n" \
"        rgb = vdot(inputMatrix, rgb);\n" \
"        \n" \
"        // XYZ to P3 transform (hardcoded)\n" \
"        float3x3 hardcodedxyzToP3Matrix = float3x3(\n" \
"            float3( 2.49349691194f, -0.829488694668f,  0.0358458302915f),\n" \
"            float3(-0.931383617919f, 1.76266097069f,  -0.0761723891287f),\n" \
"            float3(-0.402710784451f, 0.0236246771724f, 0.956884503364f)\n" \
"        );\n" \
"        rgb = vdot(hardcodedxyzToP3Matrix, rgb);\n" \
"        \n" \
"        /***************************************************\n" \
"         Tonescale Overlay Initialization\n" \
"        --------------------------------------------------*/\n" \
"        float crv_val = 0.0f;\n" \
"        float2 pos = float2(id.x, id.y);\n" \
"        float2 res = float2(p_Width, p_Height);\n" \
"        \n" \
"        // x-position based input value for tonescale overlay\n" \
"        if (params.tonescaleMap == 1) {\n" \
"            crv_val = oetf_filmlight_tlog(pos.x/res.x);\n" \
"        }\n" \
"    // Rendering Space: \"Desaturate\" to control scale of the color volume in the rgb ratios.\n" \
"        // Controlled by params.rsSa (saturation) and red and blue weights (params.rsRw and params.rsBw)\n" \
"        float3 rs_w = float3(params.rsRw, 1.0f - params.rsRw - params.rsBw, params.rsBw);\n" \
"        float sat_L = rgb.x*rs_w.x + rgb.y*rs_w.y + rgb.z*rs_w.z;\n" \
"        rgb = sat_L*params.rsSa + rgb*(1.0f - params.rsSa);\n" \
"        \n" \
"        // Offset\n" \
"        rgb += params.tnOff;\n" \
"        if (params.tonescaleMap == 1) crv_val += params.tnOff;\n" \
"        \n" \
"        /***************************************************\n" \
"          Contrast Low Module\n" \
"        --------------------------------------------------*/\n" \
"        if (params.tnLconPresetEnable || params.tnLconUIEnable) {\n" \
"            float mcon_m = pow(2.0f, -params.tnLcon);\n" \
"            float mcon_w = params.tnLconW/4.0f;\n" \
"            mcon_w *= mcon_w;\n" \
"\n" \
"            // Normalize for ts_x0 intersection constraint: https://www.desmos.com/calculator/blyvi8t2b2\n" \
"            const float mcon_cnst_sc = compress_toe_cubic(params.ts_x0, mcon_m, mcon_w, 1)/params.ts_x0;\n" \
"            rgb *= mcon_cnst_sc;\n" \
"\n" \
"            // Scale for ratio-preserving midtone contrast\n" \
"            float mcon_nm = hypotf3(clampminf3(rgb, 0.0f))/SQRT3;\n" \
"            float mcon_sc = (mcon_nm*mcon_nm + mcon_m*mcon_w)/(mcon_nm*mcon_nm + mcon_w);\n" \
"\n" \
"            if (params.tnLconPc > 0.0f) {\n" \
"                // Mix between ratio-preserving and per-channel by blending based on distance from achromatic\n" \
"\n" \
"                // Apply per-channel midtone contrast\n" \
"                float3 mcon_rgb = rgb;\n" \
"                mcon_rgb.x = compress_toe_cubic(rgb.x, mcon_m, mcon_w, 0);\n" \
"                mcon_rgb.y = compress_toe_cubic(rgb.y, mcon_m, mcon_w, 0);\n" \
"                mcon_rgb.z = compress_toe_cubic(rgb.z, mcon_m, mcon_w, 0);\n" \
"\n" \
"                // Always use some amount of ratio-preserving method towards gamut boundary\n" \
"                float mcon_mx = fmaxf3(rgb);\n" \
"                float mcon_mn = fminf3(rgb);\n" \
"                float mcon_ch = clampf(1.0f - sdivf(mcon_mn, mcon_mx), 0.0, 1.0);\n" \
"                mcon_ch = pow(mcon_ch, 4.0f*params.tnLconPc);\n" \
"                rgb = mcon_sc*rgb*mcon_ch + mcon_rgb*(1.0f - mcon_ch);\n" \
"            }\n" \
"            else { // Just use ratio-preserving\n" \
"                rgb = mcon_sc*rgb;\n" \
"            }\n" \
"            \n" \
"            // Overlay tracking for low contrast\n" \
"            if (params.tonescaleMap == 1) {\n" \
"                crv_val *= mcon_cnst_sc;\n" \
"                crv_val = crv_val*(crv_val*crv_val + mcon_m*mcon_w)/(crv_val*crv_val + mcon_w);\n" \
"            }\n" \
"        }\n" \
"\n" \
"        /***************************************************\n" \
"          Filmic Dynamic Range Compression (BETA FEATURE)\n" \
"        --------------------------------------------------*/\n" \
"        if (params.filmicMode == 1 && params.betaFeaturesEnable == 1) {\n" \
"            // Calculate maximum input based on original camera range\n" \
"            float maxInput = pow(2.0f, params.filmicSourceStops);\n" \
"            \n" \
"            // Normalize RGB to 0-1 range based on original camera stops\n" \
"            float3 normalizedRGB = rgb / maxInput;\n" \
"            \n" \
"            // Use Film Dynamic Range to control highlight rolloff characteristics\n" \
"            // Higher values = gentler rolloff (more film-like latitude)\n" \
"            // Lower values = harder rolloff (more aggressive compression)\n" \
"            float rolloff_s = 0.05f + (params.filmicDynamicRange / 10.0f); // Range: 0.05 to 1.05\n" \
"            float rolloff_p = 0.8f + (params.filmicDynamicRange / 25.0f);  // Range: 0.8 to 1.2\n" \
"            \n" \
"            // Apply hyperbolic compression with dynamic rolloff\n" \
"            float3 compressedRGB = float3(\n" \
"                compress_hyperbolic_power(normalizedRGB.x, rolloff_s, rolloff_p),\n" \
"                compress_hyperbolic_power(normalizedRGB.y, rolloff_s, rolloff_p),\n" \
"                compress_hyperbolic_power(normalizedRGB.z, rolloff_s, rolloff_p)\n" \
"            );\n" \
"            \n" \
"            // Rescale to target film range\n" \
"            float maxOutput = pow(2.0f, params.filmicTargetStops);\n" \
"            float3 rescaledRGB = compressedRGB * maxOutput;\n" \
"            \n" \
"            // Mix between original and filmic compressed result\n" \
"            rgb = rgb * (1.0f - params.filmicStrength) + rescaledRGB * params.filmicStrength;\n" \
"            \n" \
"            // Apply same compression to overlay curve\n" \
"            if (params.tonescaleMap == 1) {\n" \
"                float crv_normalized = crv_val / maxInput;\n" \
"                float crv_compressed = compress_hyperbolic_power(crv_normalized, rolloff_s, rolloff_p);\n" \
"                float crv_rescaled = crv_compressed * maxOutput;\n" \
"                crv_val = crv_val * (1.0f - params.filmicStrength) + crv_rescaled * params.filmicStrength;\n" \
"            }\n" \
"        }\n" \
"     /***************************************************\n" \
"         Tonescale and RGB Ratios\n" \
"        --------------------------------------------------*/\n" \
"  // Tonescale Norm\n" \
"        float tsn = hypotf3(clampminf3(rgb, 0.0f)) / SQRT3;\n" \
"        // Purity Compression Norm\n" \
"        float ts_pt = sqrt(max(0.0f, rgb.x * rgb.x * params.ptR + rgb.y * rgb.y * params.ptG + rgb.z * rgb.z * params.ptB));\n" \
"        \n" \
"        // RGB Ratios\n" \
"        rgb = sdivf3f(clampminf3(rgb, -2.0f), tsn);\n" \
"        \n" \
"        /***************************************************\n" \
"          Apply High Contrast\n" \
"        --------------------------------------------------*/\n" \
"        if (params.tnHconPresetEnable || params.tnHconUIEnable) {\n" \
"            float hcon_p = pow(2.0f, params.tnHcon);\n" \
"            tsn = contrast_high(tsn, hcon_p, params.tnHconPv, params.tnHconSt, 0);\n" \
"            ts_pt = contrast_high(ts_pt, hcon_p, params.tnHconPv, params.tnHconSt, 0);\n" \
"            if (params.tonescaleMap == 1) crv_val = contrast_high(crv_val, hcon_p, params.tnHconPv, params.tnHconSt, 0);\n" \
"        }\n" \
"        /***************************************************\n" \
"          Apply Advanced Contrast\n" \
"        --------------------------------------------------*/\n" \
"        if (params.advHueContrast == 1) {\n" \
"            float3 cym_contrast = 1.0f - rgb;\n" \
"            float3 rgb_contrast = rgb;\n" \
"            float pivot = 1.0f;\n" \
"            rgb_contrast.x = gamma_contrast(rgb_contrast.x, params.advHcR, pivot);\n" \
"            cym_contrast.x = gamma_contrast(cym_contrast.x, params.advHcC, pivot);\n" \
"            rgb_contrast.y = gamma_contrast(rgb_contrast.y, params.advHcG, pivot);\n" \
"            cym_contrast.y = gamma_contrast(cym_contrast.y, params.advHcM, pivot);\n" \
"            rgb_contrast.z = gamma_contrast(rgb_contrast.z, params.advHcB, pivot);\n" \
"            cym_contrast.z = gamma_contrast(cym_contrast.z, params.advHcY, pivot);\n" \
"            \n" \
"            rgb_contrast.x = mix(rgb_contrast.x, 1.0f - cym_contrast.x, 0.5f);\n" \
"            rgb_contrast.y = mix(rgb_contrast.y, 1.0f - cym_contrast.y, 0.5f);\n" \
"            rgb_contrast.z = mix(rgb_contrast.z, 1.0f - cym_contrast.z, 0.5f);\n" \
"            \n" \
"            rgb = mix(rgb, rgb_contrast, params.advHcPower);\n" \
"        }\n" \
"        \n" \
"        /***************************************************\n" \
"          Apply Tonescale\n" \
"        --------------------------------------------------*/\n" \
"        tsn = compress_hyperbolic_power(tsn, params.ts_s, params.tnCon);\n" \
"        ts_pt = compress_hyperbolic_power(ts_pt, params.ts_s1, params.tnCon);\n" \
"        \n" \
"        if (params.tonescaleMap == 1) crv_val = compress_hyperbolic_power(crv_val, params.ts_s, params.tnCon);\n" \
"        \n" \
"        \n" \
" \n" \
"        /***************************************************\n" \
"          Prequiste color spaces\n" \
"        --------------------------------------------------*/\n" \
"        // Simple Cyan-Yellow / Green-Magenta opponent space for calculating smooth achromatic distance and hue angles\n" \
"        float opp_cy = rgb.x - rgb.z;\n" \
"        float opp_gm = rgb.y - (rgb.x + rgb.z)/2.0f;\n" \
"        float ach_d = sqrt(max(0.0f, opp_cy*opp_cy + opp_gm*opp_gm))/SQRT3;\n" \
"\n" \
"        // Smooth ach_d, normalized so 1.0 doesn't change https://www.desmos.com/calculator/ozjg09hzef\n" \
"        ach_d = (1.25f)*compress_toe_quadratic(ach_d, 0.25f, 0);\n" \
"\n" \
"        // Hue angle, rotated so that red = 0.0\n" \
"        float hue = fmod(atan2(opp_cy, opp_gm) + PI + 1.10714931f, 2.0f*PI);\n" \
"\n" \
"        // RGB Hue Angles\n" \
"        // Wider than CMY by default. R towards M, G towards Y, B towards C\n" \
"        float3 ha_rgb = float3(\n" \
"          gauss_window(hue_offset(hue, 0.1f), 0.9f),\n" \
"          gauss_window(hue_offset(hue, 4.3f), 0.9f),\n" \
"          gauss_window(hue_offset(hue, 2.3f), 0.9f));\n" \
"\n" \
"        // CMY Hue Angles\n" \
"        // Exact alignment to Cyan/Magenta/Yellow secondaries would be PI, PI/3 and -PI/3, but\n" \
"        // we customize these a bit for creative purposes: M towards B, Y towards G, C towards G\n" \
"        float3 ha_cmy = float3(\n" \
"          gauss_window(hue_offset(hue, 3.3f), 0.6f),\n" \
"          gauss_window(hue_offset(hue, 1.3f), 0.6f),\n" \
"          gauss_window(hue_offset(hue, -1.2f), 0.6f));\n" \
"\n" \
"        // Purity Compression Range: https://www.desmos.com/calculator/8ynarg1uxk\n" \
"        float ts_pt_cmp = 1.0f - pow(ts_pt, 1.0f/params.ptRngLow);\n" \
"\n" \
"        float pt_rng_high_f = min(1.0f, ach_d/1.2f);\n" \
"        pt_rng_high_f *= pt_rng_high_f;\n" \
"        pt_rng_high_f = params.ptRngHigh < 1.0f ? 1.0f - pt_rng_high_f : pt_rng_high_f;\n" \
"        ts_pt_cmp = pow(ts_pt_cmp, params.ptRngHigh)*(1.0f - pt_rng_high_f) + ts_pt_cmp*pt_rng_high_f;\n" \
"\n" \
"        /***************************************************\n" \
"          Brilliance\n" \
"        --------------------------------------------------*/\n" \
"        float brl_f = 1.0f;\n" \
"        if (params.brlPresetEnable || params.brlUIEnable) {\n" \
"            brl_f = -params.brlR*ha_rgb.x - params.brlG*ha_rgb.y - params.brlB*ha_rgb.z - params.brlC*ha_cmy.x - params.brlM*ha_cmy.y - params.brlY*ha_cmy.z;\n" \
"            brl_f = (1.0f - ach_d)*brl_f + 1.0f - brl_f;\n" \
"            brl_f = softplus(brl_f, 0.25f, -100.0f, 0.0f); // Protect against over-darkening\n" \
"            \n" \
"            // Limit Brilliance adjustment by tonescale\n" \
"            float brl_ts = brl_f > 1.0f ? 1.0f - ts_pt : ts_pt; // Limit by inverse tonescale if positive Brilliance adjustment\n" \
"            float brl_lim = spowf(brl_ts, 1.0f - params.brlRng);\n" \
"            brl_f = brl_f*brl_lim + 1.0f - brl_lim;\n" \
"            brl_f = max(0.0f, min(2.0f, brl_f)); // protect for shadow grain\n" \
"        }\n" \
"\n" \
"        /***************************************************\n" \
"          Mid-Range Purity\n" \
"            This boosts mid-range purity on the low end\n" \
"            and reduces mid-range purity on the high end\n" \
"        --------------------------------------------------*/\n" \
"        float ptm_sc = 1.0f;\n" \
"        if (params.ptmPresetEnable || params.ptmUIEnable) {\n" \
"            // Mid Purity Low\n" \
"            float ptm_ach_d = complement_power(ach_d, params.ptmLowSt);\n" \
"            ptm_sc = sigmoid_cubic(ptm_ach_d, params.ptmLow*(1.0f - ts_pt));\n" \
"            \n" \
"            // Mid Purity High\n" \
"            ptm_ach_d = complement_power(ach_d, params.ptmHighSt)*(1.0f - ts_pt) + ach_d*ach_d*ts_pt;\n" \
"            ptm_sc *= sigmoid_cubic(ptm_ach_d, params.ptmHigh*ts_pt);\n" \
"            ptm_sc = max(0.0f, ptm_sc); // Ensure no negative scale\n" \
"        }\n" \
"        \n" \
"        /***************************************************\n" \
"          Hue Angle Premultiplication (MISSING STEP ADDED)\n" \
"        --------------------------------------------------*/\n" \
"        // Premult hue angles for Hue Contrast and Hue Shift (matches DCTL implementation)\n" \
"        ha_rgb *= ach_d;\n" \
"        ha_cmy *= (1.5f)*compress_toe_quadratic(ach_d, 0.5f, 0); // Stronger smoothing for CMY hue shift\n" \
"        \n" \
"        /***************************************************\n" \
"          Hue Contrast R\n" \
"        --------------------------------------------------*/\n" \
"        if (params.hcPresetEnable || params.hcUIEnable) {\n" \
"            float hc_ts = 1.0f - ts_pt;\n" \
"            // Limit high purity on bottom end and low purity on top end by ach_d.\n" \
"            // This helps reduce artifacts and over-saturation.\n" \
"            float hc_c = (1.0f - ach_d)*hc_ts + ach_d*(1.0f - hc_ts);\n" \
"            hc_c *= ha_rgb.x;\n" \
"            hc_ts *= hc_ts;\n" \
"            // Bias contrast based on tonescale using Lift/Mult: https://www.desmos.com/calculator/gzbgov62hl\n" \
"            float hc_f = params.hcR*(hc_c - 2.0f*hc_c*hc_ts) + 1.0f;\n" \
"            rgb = float3(rgb.x, rgb.y*hc_f, rgb.z*hc_f);\n" \
"        }\n" \
"\n" \
"        /***************************************************\n" \
"          Hue Shift\n" \
"        --------------------------------------------------*/\n" \
"\n" \
"        // Hue Shift RGB by purity compress tonescale, shifting more as intensity increases\n" \
"        if (params.hsRgbPresetEnable || params.hsRgbUIEnable) {\n" \
"            float3 hs_rgb = ha_rgb*pow(ts_pt, 1.0f/params.hsRgbRng);\n" \
"            float3 hsf = float3(hs_rgb.x*params.hsR, hs_rgb.y*-params.hsG, hs_rgb.z*-params.hsB);\n" \
"            hsf = float3(hsf.z - hsf.y, hsf.x - hsf.z, hsf.y - hsf.x);\n" \
"            rgb += hsf;\n" \
"        }\n" \
"\n" \
"        // Hue Shift CMY by tonescale, shifting less as intensity increases\n" \
"        if (params.hsCmyPresetEnable || params.hsCmyUIEnable) {\n" \
"            float3 hs_cmy = ha_cmy*(1.0f - ts_pt);\n" \
"            float3 hsf = float3(hs_cmy.x*-params.hsC, hs_cmy.y*params.hsM, hs_cmy.z*params.hsY);\n" \
"            hsf = float3(hsf.z - hsf.y, hsf.x - hsf.z, hsf.y - hsf.x);\n" \
"            rgb += hsf;\n" \
"        }\n" \
"\n" \
"        /***************************************************\n" \
"          Module Application\n" \
"        --------------------------------------------------*/\n" \
"        \n" \
"        // Apply brilliance\n" \
"        rgb *= brl_f;\n" \
"        \n" \
"        // Apply purity compression and mid purity\n" \
"        ts_pt_cmp *= ptm_sc;\n" \
"        rgb = rgb*ts_pt_cmp + 1.0f - ts_pt_cmp;\n" \
"\n" \
"        // Inverse Rendering Space\n" \
"        sat_L = rgb.x*rs_w.x + rgb.y*rs_w.y + rgb.z*rs_w.z;\n" \
"        rgb = (sat_L*params.rsSa - rgb)/(params.rsSa - 1.0f);\n" \
"\n" \
"       /***************************************************\n" \
"          Creative White Point and Output Transform\n" \
"        --------------------------------------------------*/\n" \
"        // Convert to final display gamut with creative whitepoint\n" \
"        float3 cwp_rgb = rgb;\n" \
"\n" \
"        // Apply creative whitepoint transform (already handles D65 as identity)\n" \
"        cwp_rgb = vdot(cwpMatrix, rgb);\n" \
"\n" \
"        if (params.displayGamut == 0) { // Rec.709\n" \
"            // ALWAYS convert rgb to Rec.709 D65 (specific matrix)\n" \
"            float3x3 p3ToRec709D65 = float3x3(\n" \
"                float3(1.224940181f, -0.04205697775f, -0.01963755488f),\n" \
"                float3(-0.2249402404f, 1.042057037f, -0.07863604277f),\n" \
"                float3(0.0f, -1.4901e-08f, 1.098273635f)\n" \
"            );\n" \
"            rgb = vdot(p3ToRec709D65, rgb);\n" \
"            \n" \
"            // If D65, use the converted result\n" \
"            if (params.cwp == 0) cwp_rgb = rgb;\n" \
"        }\n" \
"        // For P3-D65 and Rec.2020 (displayGamut >= 1): no gamut conversion needed\n" \
"\n" \
"        // Mix between Creative Whitepoint and base by tsn\n" \
"        float cwp_f = pow(tsn, 1.0f - params.cwpRng);\n" \
"        rgb = cwp_rgb*cwp_f + rgb*(1.0f - cwp_f);\n" \
"// Process overlay curve for display gamut and creative whitepoint\n" \
"        float3 crv_rgb = float3(crv_val, crv_val, crv_val);\n" \
"        float3 crv_rgb_cwp = crv_rgb;\n" \
"       if (params.tonescaleMap == 1) {\n" \
"            if (params.displayGamut == 0) { // Rec.709\n" \
"                // Apply creative whitepoint transform for overlay\n" \
"                crv_rgb_cwp = vdot(cwpMatrix, crv_rgb);\n" \
"                // Apply P3 to Rec.709 transform\n" \
"                float3x3 p3ToRec709 = float3x3(\n" \
"                    float3(1.224940181f, -0.04205697775f, -0.01963755488f),\n" \
"                    float3(-0.2249402404f, 1.042057037f, -0.07863604277f),\n" \
"                    float3(0.0f, -1.4901e-08f, 1.098273635f)\n" \
"                );\n" \
"                crv_rgb = vdot(p3ToRec709, crv_rgb);\n" \
"                if (params.cwp == 0) crv_rgb_cwp = crv_rgb;\n" \
"            }\n" \
"            else if (params.displayGamut >= 1) {\n" \
"                // Apply creative whitepoint transform for P3 and Rec.2020\n" \
"                crv_rgb_cwp = vdot(cwpMatrix, crv_rgb);\n" \
"                if (params.cwp == 0) crv_rgb_cwp = crv_rgb;\n" \
"            }\n" \
"            \n" \
"            // Mix creative whitepoint for overlay curve\n" \
"            float crv_rgb_cwp_f = pow(crv_val, 1.0f - params.cwpRng);\n" \
"            crv_rgb = crv_rgb_cwp*crv_rgb_cwp_f + crv_rgb*(1.0f - crv_rgb_cwp_f);\n" \
"        }\n" \
"        \n" \
"        /***************************************************\n" \
"          Purity Compress Low\n" \
"        --------------------------------------------------*/\n" \
"        if (params.ptlPresetEnable || params.ptlUIEnable) {\n" \
"            float sum0 = softplus(rgb.x, 0.2f, -100.0f, -0.3f) + rgb.y + softplus(rgb.z, 0.2f, -100.0f, -0.3f);\n" \
"            rgb.x = softplus(rgb.x, 0.04f, -0.3f, 0.0f);\n" \
"            rgb.y = softplus(rgb.y, 0.06f, -0.3f, 0.0f);\n" \
"            rgb.z = softplus(rgb.z, 0.01f, -0.05f, 0.0f);\n" \
"\n" \
"            float ptl_norm = min(1.0f, sdivf(sum0, rgb.x + rgb.y + rgb.z));\n" \
"            rgb *= ptl_norm;\n" \
"        }\n" \
"\n" \
"        /***************************************************\n" \
"          Final Tonescale and Display Transform\n" \
"        --------------------------------------------------*/\n" \
"        // Final tonescale adjustments\n" \
"        tsn *= params.ts_m2; // scale for inverse toe\n" \
"        tsn = compress_toe_quadratic(tsn, params.tnToe, 0);\n" \
"        tsn *= params.ts_dsc; // scale for display encoding\n" \
"        \n" \
"        // Track overlay value through final tonescale adjustments\n" \
"        if (params.tonescaleMap == 1) {\n" \
"            crv_rgb *= params.ts_m2;\n" \
"            crv_rgb.x = compress_toe_quadratic(crv_rgb.x, params.tnToe, 0);\n" \
"            crv_rgb.y = compress_toe_quadratic(crv_rgb.y, params.tnToe, 0);\n" \
"            crv_rgb.z = compress_toe_quadratic(crv_rgb.z, params.tnToe, 0);\n" \
"            crv_rgb *= params.ts_dsc;\n" \
"            // scale to 1.0 = 1000 nits for st2084 PQ\n" \
"            if (params.eotf == 4) crv_rgb *= 10.0f;\n" \
"        }\n" \
"        \n" \
"        // Return from RGB ratios to final values\n" \
"        rgb *= tsn;\n" \
"        \n" \
"        \n" \
"        // Clamp if enabled\n" \
"        if (params.clamp != 0) {\n" \
"            rgb = clampf3(rgb, 0.0f, 1.0f);\n" \
"        }\n" \
"        \n" \
"        // Rec.2020 (P3 Limited) conversion - MOVED TO CORRECT LOCATION\n" \
"        if (params.displayGamut == 2) { // Rec.2020\n" \
"            rgb = clampminf3(rgb, 0.0f);\n" \
"            // Use specific P3→Rec.2020 matrix (hardcoded)\n" \
"            float3x3 p3ToRec2020 = float3x3(\n" \
"                float3(0.7538330344f, 0.04574384897f, -0.001210340355f),\n" \
"                float3(0.1985973691f, 0.9417772198f, 0.0176017173f),\n" \
"                float3(0.04756959659f, 0.01247893122f, 0.9836086231f)\n" \
"            );\n" \
"            rgb = vdot(p3ToRec2020, rgb);\n" \
"        }\n" \
"        \n" \
"        // ENCODE FOR DISPLAY using dynamic EOTF parameters\n" \
"        rgb = encode_for_display(rgb, params.eotf);\n" \
"        \n" \
"        // Apply EOTF to overlay curve\n" \
"        if (params.tonescaleMap == 1) {\n" \
"            if ((params.eotf > 0) && (params.eotf < 4)) {\n" \
"                float eotf_p = 2.0f + params.eotf * 0.2f;\n" \
"                crv_rgb = spowf3(crv_rgb, 1.0f/eotf_p);\n" \
"            }\n" \
"            else if (params.eotf == 4) crv_rgb = eotf_pq(crv_rgb, 1);\n" \
"            else if (params.eotf == 5) crv_rgb = eotf_hlg(crv_rgb, 1);\n" \
"        }\n" \
"        \n" \
"        // Render overlay curve\n" \
"        if (params.tonescaleMap == 1) {\n" \
"            float3 crv_rgb_dst = float3(pos.y-crv_rgb.x*res.y, pos.y-crv_rgb.y*res.y, pos.y-crv_rgb.z*res.y);\n" \
"            float crv_w0 = 0.05f; // width of tonescale overlay\n" \
"            crv_rgb_dst.x = exp(-crv_rgb_dst.x*crv_rgb_dst.x*crv_w0);\n" \
"            crv_rgb_dst.y = exp(-crv_rgb_dst.y*crv_rgb_dst.y*crv_w0);\n" \
"            crv_rgb_dst.z = exp(-crv_rgb_dst.z*crv_rgb_dst.z*crv_w0);\n" \
"            float crv_lm = params.eotf < 4 ? 1.0f : 0.5f; // reduced luminance in hdr\n" \
"            crv_rgb_dst = clampf3(crv_rgb_dst, 0.0f, 1.0f);\n" \
"            rgb = rgb * (1.0f - crv_rgb_dst) + crv_lm*crv_rgb_dst*crv_rgb_dst;\n" \
"        }\n" \
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
    // Advanced Hue Contrast Parameters
    float p_AdvHcR, float p_AdvHcG, float p_AdvHcB, float p_AdvHcC, float p_AdvHcM, float p_AdvHcY, float p_AdvHcPower,
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
    
    // Advanced Hue Contrast Parameters
    params.advHcR = p_AdvHcR;
    params.advHcG = p_AdvHcG;
    params.advHcB = p_AdvHcB;
    params.advHcC = p_AdvHcC;
    params.advHcM = p_AdvHcM;
    params.advHcY = p_AdvHcY;
    params.advHcPower = p_AdvHcPower;
    
    // NEW PARAMETERS - Filmic Mode and Advanced Controls
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
    
    // Display Parameters
    params.displayGamut = p_DisplayGamut;
    params.eotf = p_Eotf;
    
    // Map your split parameters to DCTL display_encoding_preset logic
    int display_encoding_preset;
    if (p_DisplayGamut == 0 && p_Eotf == 2) {        // Rec.709 + 2.4 Power
        display_encoding_preset = 0; // Rec.1886
    } else if (p_DisplayGamut == 0 && p_Eotf == 1) {  // Rec.709 + 2.2 Power
        display_encoding_preset = 1; // sRGB Display
    } else if (p_DisplayGamut == 1 && p_Eotf == 1) {  // P3-D65 + 2.2 Power
        display_encoding_preset = 2; // Display P3
    } else if (p_DisplayGamut == 2 && p_Eotf == 4) {  // Rec.2020 + PQ
        display_encoding_preset = 3; // Rec.2100 PQ
    } else if (p_DisplayGamut == 2 && p_Eotf == 5) {  // Rec.2020 + HLG
        display_encoding_preset = 4; // Rec.2100 HLG
    } else if (p_DisplayGamut == 1 && p_Eotf == 4) {  // P3-D65 + PQ
        display_encoding_preset = 5; // Dolby PQ
    } else {
        display_encoding_preset = 0; // Default fallback to Rec.1886
    }

    // Now map display_encoding_preset to the correct internal values
    int final_display_gamut, final_eotf;
    if (display_encoding_preset == 0) { // Rec.1886
        final_display_gamut = 0; // Rec.709
        final_eotf = 2;         // 2.4 Power
    } else if (display_encoding_preset == 1) { // sRGB Display
        final_display_gamut = 0; // Rec.709
        final_eotf = 1;         // 2.2 Power
    } else if (display_encoding_preset == 2) { // Display P3
        final_display_gamut = 1; // P3-D65
        final_eotf = 1;         // 2.2 Power
    } else if (display_encoding_preset == 3) { // Rec.2100 PQ
        final_display_gamut = 2; // Rec.2020
        final_eotf = 4;         // PQ
    } else if (display_encoding_preset == 4) { // Rec.2100 HLG
        final_display_gamut = 2; // Rec.2020
        final_eotf = 5;         // HLG
    } else if (display_encoding_preset == 5) { // Dolby PQ
        final_display_gamut = 1; // P3-D65
        final_eotf = 4;         // PQ
    }

    // Store the corrected values
    params.displayGamut = final_display_gamut;
    params.eotf = final_eotf;


    // Load transfer function parameters using corrected EOTF
    const auto& oetfPreset = OpenDRTPresets::getOETFParams(p_InOetf);
    const auto& eotfPreset = OpenDRTPresets::getEOTFParams(final_eotf);
    
    // DEBUG: Print transfer function info to Resolve console
    fprintf(stderr, "OpenDRT DEBUG: Input OETF=%d, type=%d | Output EOTF=%d, type=%d\n", 
            p_InOetf, oetfPreset.oetf_type, final_eotf, eotfPreset.eotf_type);
    
    // PRECALCULATE TONESCALE CONSTANTS
    const auto& tonescaleConstants = OpenDRTPresets::calculateTonescaleConstants(
        p_TnLp, p_TnGb, p_PtHdr, p_TnLg, p_TnCon, p_TnSh, p_TnToe, p_TnOff, final_eotf);


    // Copy transfer function data to params
    memcpy(params.oetfParams, oetfPreset.oetf_params, sizeof(float) * 8);
    memcpy(params.eotfParams, eotfPreset.eotf_params, sizeof(float) * 8);
    params.oetfType = oetfPreset.oetf_type;
    params.eotfType = eotfPreset.eotf_type;
    
    // Copy precalculated tonescale constants
    params.ts_x1 = tonescaleConstants.ts_x1;
    params.ts_y1 = tonescaleConstants.ts_y1;
    params.ts_x0 = tonescaleConstants.ts_x0;
    params.ts_y0 = tonescaleConstants.ts_y0;
    params.ts_s0 = tonescaleConstants.ts_s0;
    params.ts_s10 = tonescaleConstants.ts_s10;
    params.ts_m1 = tonescaleConstants.ts_m1;
    params.ts_m2 = tonescaleConstants.ts_m2;
    params.ts_s = tonescaleConstants.ts_s;
    params.ts_dsc = tonescaleConstants.ts_dsc;
    params.pt_cmp_Lf = tonescaleConstants.pt_cmp_Lf;
    params.s_Lp100 = tonescaleConstants.s_Lp100;
    params.ts_s1 = tonescaleConstants.ts_s1;
    
// ASSIGN UI ENABLE FLAGS:
params.tnHconUIEnable = p_TnHconUIEnable ? 1 : 0;
params.tnLconUIEnable = p_TnLconUIEnable ? 1 : 0;
params.ptlUIEnable = p_PtlUIEnable ? 1 : 0;
params.ptmUIEnable = p_PtmUIEnable ? 1 : 0;
params.brlUIEnable = p_BrlUIEnable ? 1 : 0;
params.hsRgbUIEnable = p_HsRgbUIEnable ? 1 : 0;
params.hsCmyUIEnable = p_HsCmyUIEnable ? 1 : 0;
params.hcUIEnable = p_HcUIEnable ? 1 : 0;

// ASSIGN PRESET ENABLE FLAGS:
params.tnHconPresetEnable = p_TnHconPresetEnable ? 1 : 0;
params.tnLconPresetEnable = p_TnLconPresetEnable ? 1 : 0;
params.ptlPresetEnable = p_PtlPresetEnable ? 1 : 0;
params.ptmPresetEnable = p_PtmPresetEnable ? 1 : 0;
params.brlPresetEnable = p_BrlPresetEnable ? 1 : 0;
params.hsRgbPresetEnable = p_HsRgbPresetEnable ? 1 : 0;
params.hsCmyPresetEnable = p_HsCmyPresetEnable ? 1 : 0;
params.hcPresetEnable = p_HcPresetEnable ? 1 : 0;
    
    return params;
}

void OpenDRTKernel(void* p_CmdQ, int p_Width, int p_Height,
                   const float* p_Input, float* p_Output,
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
                   // Advanced Hue Contrast Parameters
                   float p_AdvHcR, float p_AdvHcG, float p_AdvHcB, float p_AdvHcC, float p_AdvHcM, float p_AdvHcY, float p_AdvHcPower,
                   // NEW PARAMETERS - Filmic Mode and Advanced Controls
                   bool p_FilmicMode, float p_FilmicDynamicRange, int p_FilmicProjectorSim,
                   float p_FilmicSourceStops, float p_FilmicTargetStops, float p_FilmicStrength,
                   bool p_AdvHueContrast, bool p_TonescaleMap, bool p_DiagnosticsMode, bool p_RgbChipsMode, bool p_BetaFeaturesEnable,
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
        p_TnHconUIEnable, p_TnLconUIEnable,  // ADD THESE
        p_PtlUIEnable, p_PtmUIEnable,
        p_BrlUIEnable, p_HsRgbUIEnable,
        p_HsCmyUIEnable, p_HcUIEnable,
        p_TnHconPresetEnable, p_TnLconPresetEnable,  // ADD THESE
        p_PtlPresetEnable, p_PtmPresetEnable,
        p_BrlPresetEnable, p_HsRgbPresetEnable,
        p_HsCmyPresetEnable, p_HcPresetEnable,
        p_TnHcon, p_TnHconPv, p_TnHconSt,  // ADD THESE
        p_TnLcon, p_TnLconW, p_TnLconPc,   // ADD THESE
        p_Cwp, p_CwpRng,
        p_RsSa, p_RsRw, p_RsBw,
        p_PtR, p_PtG, p_PtB, p_PtRngLow, p_PtRngHigh,
        p_PtmLow, p_PtmLowSt, p_PtmHigh, p_PtmHighSt,
        p_BrlR, p_BrlG, p_BrlB,  // ADD ENABLE
        p_BrlC, p_BrlM, p_BrlY, p_BrlRng,
        p_HsR, p_HsG, p_HsB, p_HsRgbRng,  // ADD ENABLE
        p_HsC, p_HsM, p_HsY,  // ADD ENABLE
        p_HcR,  // ADD ENABLE
        // Advanced Hue Contrast Parameters
        p_AdvHcR, p_AdvHcG, p_AdvHcB, p_AdvHcC, p_AdvHcM, p_AdvHcY, p_AdvHcPower,
        // NEW PARAMETERS - Filmic Mode and Advanced Controls
        p_FilmicMode, p_FilmicDynamicRange, p_FilmicProjectorSim,
        p_FilmicSourceStops, p_FilmicTargetStops, p_FilmicStrength,
        p_AdvHueContrast, p_TonescaleMap, p_DiagnosticsMode, p_RgbChipsMode, p_BetaFeaturesEnable,
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

