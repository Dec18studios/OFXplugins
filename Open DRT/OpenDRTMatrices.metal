#include <metal_stdlib>
using namespace metal;

// Example: Add any constants you need
constant float PI = 3.14159265358979323846f;

// Dynamic Input Matrix Selection
float3x3 getInputMatrix(int gamut) {
    switch(gamut) {
        case 0: // DaVinci Wide Gamut to XYZ
            return float3x3(
                float3(0.700622320175f, 0.274118483067f, -0.098962903023f),
                float3(0.148774802685f, 0.873631775379f, -0.137895315886f),
                float3(0.101058728993f, -0.147750422359f, 1.325916051865f));
        // Add more cases as needed...
        default:
            return float3x3(1.0f);
    }
}

// Dynamic Output Matrix Selection
float3x3 getOutputMatrix(int gamut) {
    switch(gamut) {
        case 0: // P3 to Rec.709 D65
            return float3x3(
                float3( 1.224940181f,   -0.04205697775f, -0.01963755488f),
                float3(-0.2249402404f,   1.042057037f,   -0.07863604277f),
                float3( 0.0f,           -1.4901e-08f,     1.098273635f));
        case 1: // P3 Identity
            return float3x3(1.0f);
        case 2: // P3 to Rec.2020
            return float3x3(
                float3(0.627403914928f, 0.069097289441f, 0.016391203574f),
                float3(0.329283038616f, 0.919540429115f, 0.088013307851f),
                float3(0.043313046456f, 0.011362281442f, 0.895595488575f));
        default:
            return float3x3(1.0f);
    }
}

// Creative Whitepoint Matrix (example)
float3x3 getCreativeWhitepointMatrix(int displayGamut, int cwp) {
    // Add your logic here, or just return identity for now
    return float3x3(1.0f);
}

// XYZ to P3 working matrix (example)
float3x3 getXYZToWorkingMatrix() {
    return float3x3(
        float3( 2.49349691194f, -0.829488694668f,  0.0358458302915f),
        float3(-0.931383617919f, 1.76266097069f,  -0.0761723891287f),
        float3(-0.402710784451f, 0.0236246771724f, 0.956884503364f));
}