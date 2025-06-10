#ifndef __APPLE__
#include <cmath>
#include <algorithm>
#endif

#ifdef __APPLE__
// On macOS, avoid CUDA entirely and use standard C++ math
#include <cmath>
#include <algorithm>
#define __global__
#define __host__
#define __device__
#define blockIdx (dim3{0,0,0})
#define blockDim (dim3{1,1,1})
#define threadIdx (dim3{0,0,0})
struct dim3 { int x, y, z; };
#else
// Real CUDA for Windows/Linux
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#endif

__global__ void ContrastSatVolumeKernel(int p_Width, int p_Height,
                                        float* p_RgbGammas, float* p_CymGammas,
                                        float* p_RgbMidgreys, float* p_CymMidgreys,
                                        float* p_Tilts,
                                        const float* p_Input, float* p_Output)
{
#ifndef __APPLE__
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < p_Width) && (y < p_Height))
    {
        const int index = ((y * p_Width) + x) * 4;

        // Extract RGBA values
        float r = p_Input[index + 0];
        float g = p_Input[index + 1];
        float b = p_Input[index + 2];
        float a = p_Input[index + 3];
        
        // Extract parameters from arrays
        float gammaR = p_RgbGammas[0], gammaG = p_RgbGammas[1], gammaB = p_RgbGammas[2];
        float gammaC = p_CymGammas[0], gammaM = p_CymGammas[1], gammaY = p_CymGammas[2];
        float midgreyR = p_RgbMidgreys[0], midgreyG = p_RgbMidgreys[1], midgreyB = p_RgbMidgreys[2];
        float midgreyC = p_CymMidgreys[0], midgreyM = p_CymMidgreys[1], midgreyY = p_CymMidgreys[2];
        float tiltCR = p_Tilts[0], tiltGM = p_Tilts[1], tiltBY = p_Tilts[2];
        
        // Apply contrast to ENTIRE RGB using different gammas (like CPU applyGammaContrastFloat3)
        
        // RGB adjustments - apply each gamma to all RGB channels
        float safeR = fmaxf(0.001f, fminf(0.999f, r));
        float safeG = fmaxf(0.001f, fminf(0.999f, g));
        float safeB = fmaxf(0.001f, fminf(0.999f, b));
        
        // Red gamma applied to all RGB channels
        float safeMidgreyR = fmaxf(0.001f, fminf(0.999f, midgreyR));
        float recipMidgreyR = 1.0f / safeMidgreyR;
        float rAdj_R = powf(safeR * recipMidgreyR, gammaR) * safeMidgreyR;
        float rAdj_G = powf(safeG * recipMidgreyR, gammaR) * safeMidgreyR;
        float rAdj_B = powf(safeB * recipMidgreyR, gammaR) * safeMidgreyR;
        
        // Green gamma applied to all RGB channels
        float safeMidgreyG = fmaxf(0.001f, fminf(0.999f, midgreyG));
        float recipMidgreyG = 1.0f / safeMidgreyG;
        float gAdj_R = powf(safeR * recipMidgreyG, gammaG) * safeMidgreyG;
        float gAdj_G = powf(safeG * recipMidgreyG, gammaG) * safeMidgreyG;
        float gAdj_B = powf(safeB * recipMidgreyG, gammaG) * safeMidgreyG;
        
        // Blue gamma applied to all RGB channels
        float safeMidgreyB = fmaxf(0.001f, fminf(0.999f, midgreyB));
        float recipMidgreyB = 1.0f / safeMidgreyB;
        float bAdj_R = powf(safeR * recipMidgreyB, gammaB) * safeMidgreyB;
        float bAdj_G = powf(safeG * recipMidgreyB, gammaB) * safeMidgreyB;
        float bAdj_B = powf(safeB * recipMidgreyB, gammaB) * safeMidgreyB;
        
        // CYM adjustments - apply each gamma to all RGB channels
        float safeMidgreyC = fmaxf(0.001f, fminf(0.999f, midgreyC));
        float recipMidgreyC = 1.0f / safeMidgreyC;
        float cAdj_R = powf(safeR * recipMidgreyC, gammaC) * safeMidgreyC;
        float cAdj_G = powf(safeG * recipMidgreyC, gammaC) * safeMidgreyC;
        float cAdj_B = powf(safeB * recipMidgreyC, gammaC) * safeMidgreyC;
        
        float safeMidgreyM = fmaxf(0.001f, fminf(0.999f, midgreyM));
        float recipMidgreyM = 1.0f / safeMidgreyM;
        float mAdj_R = powf(safeR * recipMidgreyM, gammaM) * safeMidgreyM;
        float mAdj_G = powf(safeG * recipMidgreyM, gammaM) * safeMidgreyM;
        float mAdj_B = powf(safeB * recipMidgreyM, gammaM) * safeMidgreyM;
        
        float safeMidgreyY = fmaxf(0.001f, fminf(0.999f, midgreyY));
        float recipMidgreyY = 1.0f / safeMidgreyY;
        float yAdj_R = powf(safeR * recipMidgreyY, gammaY) * safeMidgreyY;
        float yAdj_G = powf(safeG * recipMidgreyY, gammaY) * safeMidgreyY;
        float yAdj_B = powf(safeB * recipMidgreyY, gammaY) * safeMidgreyY;
        
        // Mix results - each output channel gets specific adjustment blend
        float finalR = cAdj_R + (rAdj_R - cAdj_R) * tiltCR;  // Red output uses Red gamma vs Cyan gamma
        float finalG = mAdj_G + (gAdj_G - mAdj_G) * tiltGM;  // Green output uses Green gamma vs Magenta gamma  
        float finalB = yAdj_B + (bAdj_B - yAdj_B) * tiltBY;  // Blue output uses Blue gamma vs Yellow gamma

        // Clamp and output
        p_Output[index + 0] = fmaxf(0.0f, fminf(1.0f, finalR));
        p_Output[index + 1] = fmaxf(0.0f, fminf(1.0f, finalG));
        p_Output[index + 2] = fmaxf(0.0f, fminf(1.0f, finalB));
        p_Output[index + 3] = a;
    }
#endif
}

__global__ void GainAdjustKernel(int p_Width, int p_Height, float p_GainR, float p_GainG, float p_GainB, float p_GainA, const float* p_Input, float* p_Output)
{
#ifndef __APPLE__
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < p_Width) && (y < p_Height))
    {
        const int index = ((y * p_Width) + x) * 4;

        p_Output[index + 0] = p_Input[index + 0] * p_GainR;
        p_Output[index + 1] = p_Input[index + 1] * p_GainG;
        p_Output[index + 2] = p_Input[index + 2] * p_GainB;
        p_Output[index + 3] = p_Input[index + 3] * p_GainA;
    }
#endif
}

#ifdef __APPLE__
// Stub implementations for macOS (does nothing)
void RunComplexCudaKernel(void* p_Stream, int p_Width, int p_Height,
                         float* p_RgbGammas, float* p_CymGammas,
                         float* p_RgbMidgreys, float* p_CymMidgreys,
                         float* p_Tilts,
                         const float* p_Input, float* p_Output)
{
    // Do nothing on macOS - Metal will be used instead
}

void RunCudaKernel(void* p_Stream, int p_Width, int p_Height, float* p_Gain, const float* p_Input, float* p_Output)
{
    // Do nothing on macOS - Metal will be used instead
}

#else
// Real CUDA implementations for Windows/Linux

void RunComplexCudaKernel(void* p_Stream, int p_Width, int p_Height,
                         float* p_RgbGammas, float* p_CymGammas,
                         float* p_RgbMidgreys, float* p_CymMidgreys,
                         float* p_Tilts,
                         const float* p_Input, float* p_Output)
{
    // SAFETY CHECK: Verify inputs are valid
    if (!p_Input || !p_Output || p_Width <= 0 || p_Height <= 0) {
        return;
    }
    
    if (!p_RgbGammas || !p_CymGammas || !p_RgbMidgreys || !p_CymMidgreys || !p_Tilts) {
        return;
    }

    dim3 threads(16, 16, 1);
    dim3 blocks(((p_Width + threads.x - 1) / threads.x), ((p_Height + threads.y - 1) / threads.y), 1);
    cudaStream_t stream = static_cast<cudaStream_t>(p_Stream);

    // Use simpler approach - pass arrays directly to kernel
    ContrastSatVolumeKernel<<<blocks, threads, 0, stream>>>(p_Width, p_Height,
                                                            p_RgbGammas, p_CymGammas,
                                                            p_RgbMidgreys, p_CymMidgreys,
                                                            p_Tilts,
                                                            p_Input, p_Output);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Synchronize to ensure completion
    if (stream) {
        cudaStreamSynchronize(stream);
    } else {
        cudaDeviceSynchronize();
    }
}

void RunCudaKernel(void* p_Stream, int p_Width, int p_Height, float* p_Gain, const float* p_Input, float* p_Output)
{
    // Convert simple gain to complex parameters for backward compatibility
    float rgbGammas[3] = {p_Gain[0], p_Gain[1], p_Gain[2]};
    float cymGammas[3] = {1.0f, 1.0f, 1.0f};
    float rgbMidgreys[3] = {0.18f, 0.18f, 0.18f};
    float cymMidgreys[3] = {0.18f, 0.18f, 0.18f};
    float tilts[3] = {1.0f, 1.0f, 1.0f}; // Pure RGB mode
    
    RunComplexCudaKernel(p_Stream, p_Width, p_Height, 
                        rgbGammas, cymGammas, rgbMidgreys, cymMidgreys, tilts,
                        p_Input, p_Output);
}

#endif
