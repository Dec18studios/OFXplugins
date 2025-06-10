// Mac implementation using CPU fallback (no CUDA support on Mac)
#include <algorithm>
#include <cmath>

void RunComplexCudaKernel(void* p_Stream, int p_Width, int p_Height,
                         float* p_RgbGammas, float* p_CymGammas,
                         float* p_RgbMidgreys, float* p_CymMidgreys,
                         float* p_Tilts,
                         const float* p_Input, float* p_Output)
{
    // Mac fallback - run on CPU since CUDA is not available
    // This provides compatibility but won't be GPU accelerated
    
    for (int y = 0; y < p_Height; ++y) {
        for (int x = 0; x < p_Width; ++x) {
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
            
            // RGB adjustments - apply each gamma to all RGB channels
            float safeR = std::max(0.001f, std::min(0.999f, r));
            float safeG = std::max(0.001f, std::min(0.999f, g));
            float safeB = std::max(0.001f, std::min(0.999f, b));
            
            // Red gamma applied to all RGB channels
            float safeMidgreyR = std::max(0.001f, std::min(0.999f, midgreyR));
            float recipMidgreyR = 1.0f / safeMidgreyR;
            float rAdj_R = std::pow(safeR * recipMidgreyR, gammaR) * safeMidgreyR;
            float rAdj_G = std::pow(safeG * recipMidgreyR, gammaR) * safeMidgreyR;
            float rAdj_B = std::pow(safeB * recipMidgreyR, gammaR) * safeMidgreyR;
            
            // Green gamma applied to all RGB channels
            float safeMidgreyG = std::max(0.001f, std::min(0.999f, midgreyG));
            float recipMidgreyG = 1.0f / safeMidgreyG;
            float gAdj_R = std::pow(safeR * recipMidgreyG, gammaG) * safeMidgreyG;
            float gAdj_G = std::pow(safeG * recipMidgreyG, gammaG) * safeMidgreyG;
            float gAdj_B = std::pow(safeB * recipMidgreyG, gammaG) * safeMidgreyG;
            
            // Blue gamma applied to all RGB channels
            float safeMidgreyB = std::max(0.001f, std::min(0.999f, midgreyB));
            float recipMidgreyB = 1.0f / safeMidgreyB;
            float bAdj_R = std::pow(safeR * recipMidgreyB, gammaB) * safeMidgreyB;
            float bAdj_G = std::pow(safeG * recipMidgreyB, gammaB) * safeMidgreyB;
            float bAdj_B = std::pow(safeB * recipMidgreyB, gammaB) * safeMidgreyB;
            
            // CYM adjustments - apply each gamma to all RGB channels
            float safeMidgreyC = std::max(0.001f, std::min(0.999f, midgreyC));
            float recipMidgreyC = 1.0f / safeMidgreyC;
            float cAdj_R = std::pow(safeR * recipMidgreyC, gammaC) * safeMidgreyC;
            float cAdj_G = std::pow(safeG * recipMidgreyC, gammaC) * safeMidgreyC;
            float cAdj_B = std::pow(safeB * recipMidgreyC, gammaC) * safeMidgreyC;
            
            float safeMidgreyM = std::max(0.001f, std::min(0.999f, midgreyM));
            float recipMidgreyM = 1.0f / safeMidgreyM;
            float mAdj_R = std::pow(safeR * recipMidgreyM, gammaM) * safeMidgreyM;
            float mAdj_G = std::pow(safeG * recipMidgreyM, gammaM) * safeMidgreyM;
            float mAdj_B = std::pow(safeB * recipMidgreyM, gammaM) * safeMidgreyM;
            
            float safeMidgreyY = std::max(0.001f, std::min(0.999f, midgreyY));
            float recipMidgreyY = 1.0f / safeMidgreyY;
            float yAdj_R = std::pow(safeR * recipMidgreyY, gammaY) * safeMidgreyY;
            float yAdj_G = std::pow(safeG * recipMidgreyY, gammaY) * safeMidgreyY;
            float yAdj_B = std::pow(safeB * recipMidgreyY, gammaY) * safeMidgreyY;
            
            // Mix results - each output channel gets specific adjustment blend
            float finalR = cAdj_R + (rAdj_R - cAdj_R) * tiltCR;
            float finalG = mAdj_G + (gAdj_G - mAdj_G) * tiltGM;
            float finalB = yAdj_B + (bAdj_B - yAdj_B) * tiltBY;

            // Clamp and output
            p_Output[index + 0] = std::max(0.0f, std::min(1.0f, finalR));
            p_Output[index + 1] = std::max(0.0f, std::min(1.0f, finalG));
            p_Output[index + 2] = std::max(0.0f, std::min(1.0f, finalB));
            p_Output[index + 3] = a;
        }
    }
}