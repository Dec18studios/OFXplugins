#ifndef FILM_PROJECTOR_METAL_H
#define FILM_PROJECTOR_METAL_H

#ifdef __APPLE__

#include <Metal/Metal.h>
#include <Foundation/Foundation.h>

// Metal kernel function declaration
void RunFilmProjectorMetalKernel(void* p_CmdQ, int p_Width, int p_Height,
                                float* p_NegativePreset, float* p_PrintPreset,
                                int p_Observer, int p_Mode, int p_LayerMode,
                                bool p_AlphaPassThru, float p_LinearAdjustment,
                                float p_GrainProbability, float p_GrainStrength, float p_GrainSeeth,
                                float p_HaloCutoff, float p_HaloRange, float p_HaloPressure,
                                const float* p_Input, float* p_Output);

#endif // __APPLE__

#endif // FILM_PROJECTOR_METAL_H