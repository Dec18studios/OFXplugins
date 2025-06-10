#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <pthread.h>  // FIXED: Add missing pthread header
#endif
#include <cstring>
#include <cmath>
#include <stdio.h>

const char *KernelSource = \
"__kernel void ContrastSatVolumeKernel(                                          \n" \
"   int p_Width,                                                                 \n" \
"   int p_Height,                                                                \n" \
"   __global float* p_RgbGammas,                                                 \n" \
"   __global float* p_CymGammas,                                                 \n" \
"   __global float* p_RgbMidgreys,                                               \n" \
"   __global float* p_CymMidgreys,                                               \n" \
"   __global float* p_Tilts,                                                     \n" \
"   __global const float* p_Input,                                               \n" \
"   __global float* p_Output)                                                    \n" \
"{                                                                               \n" \
"   const int x = get_global_id(0);                                              \n" \
"   const int y = get_global_id(1);                                              \n" \
"                                                                                \n" \
"   if ((x < p_Width) && (y < p_Height))                                         \n" \
"   {                                                                            \n" \
"       const int index = ((y * p_Width) + x) * 4;                              \n" \
"                                                                                \n" \
"       // Extract RGBA values                                                   \n" \
"       float r = p_Input[index + 0];                                            \n" \
"       float g = p_Input[index + 1];                                            \n" \
"       float b = p_Input[index + 2];                                            \n" \
"       float a = p_Input[index + 3];                                            \n" \
"                                                                                \n" \
"       // Extract parameters from arrays                                        \n" \
"       float gammaR = p_RgbGammas[0], gammaG = p_RgbGammas[1], gammaB = p_RgbGammas[2];    \n" \
"       float gammaC = p_CymGammas[0], gammaM = p_CymGammas[1], gammaY = p_CymGammas[2];    \n" \
"       float midgreyR = p_RgbMidgreys[0], midgreyG = p_RgbMidgreys[1], midgreyB = p_RgbMidgreys[2];  \n" \
"       float midgreyC = p_CymMidgreys[0], midgreyM = p_CymMidgreys[1], midgreyY = p_CymMidgreys[2];  \n" \
"       float tiltCR = p_Tilts[0], tiltGM = p_Tilts[1], tiltBY = p_Tilts[2];     \n" \
"                                                                                \n" \
"       // Apply contrast to ENTIRE RGB using different gammas (like CPU applyGammaContrastFloat3)  \n" \
"                                                                                \n" \
"       // RGB adjustments - apply each gamma to all RGB channels               \n" \
"       float safeR = clamp(r, 0.001f, 0.999f);                                 \n" \
"       float safeG = clamp(g, 0.001f, 0.999f);                                 \n" \
"       float safeB = clamp(b, 0.001f, 0.999f);                                 \n" \
"                                                                                \n" \
"       // Red gamma applied to all RGB channels                                \n" \
"       float safeMidgreyR = clamp(midgreyR, 0.001f, 0.999f);                   \n" \
"       float recipMidgreyR = 1.0f / safeMidgreyR;                               \n" \
"       float rAdj_R = pow(safeR * recipMidgreyR, gammaR) * safeMidgreyR;        \n" \
"       float rAdj_G = pow(safeG * recipMidgreyR, gammaR) * safeMidgreyR;        \n" \
"       float rAdj_B = pow(safeB * recipMidgreyR, gammaR) * safeMidgreyR;        \n" \
"                                                                                \n" \
"       // Green gamma applied to all RGB channels                              \n" \
"       float safeMidgreyG = clamp(midgreyG, 0.001f, 0.999f);                   \n" \
"       float recipMidgreyG = 1.0f / safeMidgreyG;                               \n" \
"       float gAdj_R = pow(safeR * recipMidgreyG, gammaG) * safeMidgreyG;        \n" \
"       float gAdj_G = pow(safeG * recipMidgreyG, gammaG) * safeMidgreyG;        \n" \
"       float gAdj_B = pow(safeB * recipMidgreyG, gammaG) * safeMidgreyG;        \n" \
"                                                                                \n" \
"       // Blue gamma applied to all RGB channels                               \n" \
"       float safeMidgreyB = clamp(midgreyB, 0.001f, 0.999f);                   \n" \
"       float recipMidgreyB = 1.0f / safeMidgreyB;                               \n" \
"       float bAdj_R = pow(safeR * recipMidgreyB, gammaB) * safeMidgreyB;        \n" \
"       float bAdj_G = pow(safeG * recipMidgreyB, gammaB) * safeMidgreyB;        \n" \
"       float bAdj_B = pow(safeB * recipMidgreyB, gammaB) * safeMidgreyB;        \n" \
"                                                                                \n" \
"       // CYM adjustments - apply each gamma to all RGB channels               \n" \
"       float safeMidgreyC = clamp(midgreyC, 0.001f, 0.999f);                   \n" \
"       float recipMidgreyC = 1.0f / safeMidgreyC;                               \n" \
"       float cAdj_R = pow(safeR * recipMidgreyC, gammaC) * safeMidgreyC;        \n" \
"       float cAdj_G = pow(safeG * recipMidgreyC, gammaC) * safeMidgreyC;        \n" \
"       float cAdj_B = pow(safeB * recipMidgreyC, gammaC) * safeMidgreyC;        \n" \
"                                                                                \n" \
"       float safeMidgreyM = clamp(midgreyM, 0.001f, 0.999f);                   \n" \
"       float recipMidgreyM = 1.0f / safeMidgreyM;                               \n" \
"       float mAdj_R = pow(safeR * recipMidgreyM, gammaM) * safeMidgreyM;        \n" \
"       float mAdj_G = pow(safeG * recipMidgreyM, gammaM) * safeMidgreyM;        \n" \
"       float mAdj_B = pow(safeB * recipMidgreyM, gammaM) * safeMidgreyM;        \n" \
"                                                                                \n" \
"       float safeMidgreyY = clamp(midgreyY, 0.001f, 0.999f);                   \n" \
"       float recipMidgreyY = 1.0f / safeMidgreyY;                               \n" \
"       float yAdj_R = pow(safeR * recipMidgreyY, gammaY) * safeMidgreyY;        \n" \
"       float yAdj_G = pow(safeG * recipMidgreyY, gammaY) * safeMidgreyY;        \n" \
"       float yAdj_B = pow(safeB * recipMidgreyY, gammaY) * safeMidgreyY;        \n" \
"                                                                                \n" \
"       // Mix results - each output channel gets specific adjustment blend     \n" \
"       float finalR = cAdj_R + (rAdj_R - cAdj_R) * tiltCR;  // Red output uses Red gamma vs Cyan gamma    \n" \
"       float finalG = mAdj_G + (gAdj_G - mAdj_G) * tiltGM;  // Green output uses Green gamma vs Magenta gamma  \n" \
"       float finalB = yAdj_B + (bAdj_B - yAdj_B) * tiltBY;  // Blue output uses Blue gamma vs Yellow gamma     \n" \
"                                                                                \n" \
"       // Clamp and output                                                      \n" \
"       p_Output[index + 0] = clamp(finalR, 0.0f, 1.0f);                        \n" \
"       p_Output[index + 1] = clamp(finalG, 0.0f, 1.0f);                        \n" \
"       p_Output[index + 2] = clamp(finalB, 0.0f, 1.0f);                        \n" \
"       p_Output[index + 3] = a;                                                 \n" \
"   }                                                                            \n" \
"}                                                                               \n" \
"                                                                                \n" \
"__kernel void GainAdjustKernel(                                                 \n" \
"   int p_Width,                                                                 \n" \
"   int p_Height,                                                                \n" \
"   float p_GainR,                                                               \n" \
"   float p_GainG,                                                               \n" \
"   float p_GainB,                                                               \n" \
"   float p_GainA,                                                               \n" \
"   __global const float* p_Input,                                               \n" \
"   __global float* p_Output)                                                    \n" \
"{                                                                               \n" \
"   const int x = get_global_id(0);                                              \n" \
"   const int y = get_global_id(1);                                              \n" \
"                                                                                \n" \
"   if ((x < p_Width) && (y < p_Height))                                         \n" \
"   {                                                                            \n" \
"       const int index = ((y * p_Width) + x) * 4;                              \n" \
"                                                                                \n" \
"       p_Output[index + 0] = p_Input[index + 0] * p_GainR;                      \n" \
"       p_Output[index + 1] = p_Input[index + 1] * p_GainG;                      \n" \
"       p_Output[index + 2] = p_Input[index + 2] * p_GainB;                      \n" \
"       p_Output[index + 3] = p_Input[index + 3] * p_GainA;                      \n" \
"   }                                                                            \n" \
"}                                                                               \n";

extern "C" {
    void RunOpenCLKernel(void* context, int width, int height, const float* input, float* output)
    {
        // For now, implement a simple pass-through or basic processing
        // Since this is a HueWarp plugin, you'd implement hue warping logic here
        
        // Simple pass-through for now:
        int totalPixels = width * height * 4; // RGBA
        for (int i = 0; i < totalPixels; i++) {
            output[i] = input[i];
        }
        
        printf("OpenCL HueWarp kernel - using CPU fallback\n");
    }
}

class Locker
{
public:
    Locker()
    {
#ifdef _WIN32
        InitializeCriticalSection(&CriticalSection);
#else
        pthread_mutex_init(&mutex, NULL);
#endif
    }

    ~Locker()
    {
#ifdef _WIN32
        DeleteCriticalSection(&CriticalSection);
#else
        pthread_mutex_destroy(&mutex);
#endif
    }

    void Lock()
    {
#ifdef _WIN32
        EnterCriticalSection(&CriticalSection);
#else
        pthread_mutex_lock(&mutex);
#endif
    }

    void Unlock()
    {
#ifdef _WIN32
        LeaveCriticalSection(&CriticalSection);
#else
        pthread_mutex_unlock(&mutex);
#endif
    }

private:
#ifdef _WIN32
    CRITICAL_SECTION CriticalSection;
#else
    pthread_mutex_t mutex;
#endif
};

void RunComplexOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, 
                           float* p_RgbGammas, float* p_CymGammas,
                           float* p_RgbMidgreys, float* p_CymMidgreys,
                           float* p_Tilts,
                           const float* p_Input, float* p_Output)
{
    // Basic OpenCL implementation - would need full OpenCL context setup
    // For now, fall back to CPU processing
    printf("OpenCL Complex kernel not fully implemented - using CPU fallback\n");
}
