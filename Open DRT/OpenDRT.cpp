/////// Step 1: Match your Header file name
#include "OpenDRT.h"
#include "OpenDRTPresets.h"
#include "OpenDRTParams.h"  // Include the shared header
#include "MatrixManager.h"   // Include matrix manager

#include <stdio.h>

// Add using directive to avoid namespace qualifiers
using namespace OpenDRTPresets;

// BOILERPLATE: Keep these includes for all OFX plugins
#include "ofxsImageEffect.h"
#include "ofxsInteract.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"
#include "ofxsLog.h"
#include "ofxDrawSuite.h"
#include "ofxsSupportPrivate.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
////////////// Step 2: Update these defines to give your plugin an identity //////////////////
#define kPluginName "Open DRT GE OFX 1.1"        // <-- UPDATE: Change this
#define kPluginGrouping "create@Dec18Studios.com"        // <-- UPDATE: Change this
#define kPluginDescription "Picture Forming Of The Highest Caliber"  // <-- UPDATE: Change this
//// Don't change the openfx part of this just the name...///////
#define kPluginIdentifier "com.OpenFXSample.OpenDRT"  // <-- UPDATE: Change this
#define kPluginVersionMajor 1
#define kPluginVersionMinor 0

// BOILERPLATE: Keep these unless you need different capabilities
#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

// Step 3: ADD CONSTANTS for any parameters that need to be defined
// Example constants 
// #define RED_ONLY 0
// #define GREEN_ONLY 1
// #define BLUE_ONLY 2

////////////////////////////////////////////////////////////////////////////////
// IMAGE PROCESSOR CLASS - WHERE THE ACTUAL IMAGE PROCESSING HAPPENS
////////////////////////////////////////////////////////////////////////////////

// BOILERPLATE: Main processor class - rename if needed
class ImageProcessor : public OFX::ImageProcessor 
{
public:
    explicit ImageProcessor(OFX::ImageEffect& p_Instance);

    // BOILERPLATE: Keep these processing methods
    virtual void processImagesCUDA();
    virtual void processImagesOpenCL();
    virtual void processImagesMetal();
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

    // BOILERPLATE: Basic setters
    void setSrcImg(OFX::Image* p_SrcImg);

    
    // OpenDRT parameter setter
    void setOpenDRTParams(
        // Input Settings
        int p_InGamut, int p_InOetf,
        // Tonescale Parameters
        float p_TnLp, float p_TnGb, float p_PtHdr,
        // Clamp Parameters
        bool p_Clamp, float p_TnLg, float p_TnCon, float p_TnSh, float p_TnToe, float p_TnOff,
        // High Contrast Parameters
        bool p_TnHconEnable, float p_TnHcon, float p_TnHconPv, float p_TnHconSt,
        // Low Contrast Parameters  
        bool p_TnLconEnable, float p_TnLcon, float p_TnLconW, float p_TnLconPc,
        // Creative White Parameters
        int p_Cwp, float p_CwpRng,
        // Render Space Parameters
        float p_RsSa, float p_RsRw, float p_RsBw,
        // Purity Compress Parameters
        float p_PtR, float p_PtG, float p_PtB, float p_PtRngLow, float p_PtRngHigh,
        // Purity Enable/Disable
        bool p_PtlEnable, bool p_PtmEnable,
        // Mid Purity Parameters
        float p_PtmLow, float p_PtmLowSt, float p_PtmHigh, float p_PtmHighSt,
        // Brilliance Parameters
        bool p_BrlEnable, float p_BrlR, float p_BrlG, float p_BrlB, 
        float p_BrlC, float p_BrlM, float p_BrlY, float p_BrlRng,
        // Hueshift RGB Parameters
        bool p_HsRgbEnable, float p_HsR, float p_HsG, float p_HsB, float p_HsRgbRng,
        // Hueshift CMY Parameters
        bool p_HsCmyEnable, float p_HsC, float p_HsM, float p_HsY,
        // Hue Contrast Parameters
        bool p_HcEnable, float p_HcR,
        // NEW PARAMETERS - Filmic Mode and Advanced Controls
        bool p_FilmicMode, float p_FilmicDynamicRange, int p_FilmicProjectorSim,
        float p_FilmicSourceStops, float p_FilmicTargetStops, float p_FilmicStrength,
        bool p_AdvHueContrast, bool p_TonescaleMap, bool p_DiagnosticsMode, bool p_RgbChipsMode, bool p_BetaFeaturesEnable,
        // Display Parameters
        int p_DisplayGamut, int p_Eotf,
        // Current preset index to look up enables
        int p_LookPreset
    );
    
    // Step 4: ADD YOUR PARAMETER SETTERS HERE
    // void setProjectorParams(...);  // <-- Replace with your own parameter setter
/* void setContrastParams(float p_GammaR, float p_GammaG, float p_GammaB, 
                          float p_GammaC, float p_GammaM, float p_GammaY,
                          float p_MidgreyR, float p_MidgreyG, float p_MidgreyB,
                          float p_MidgreyC, float p_MidgreyM, float p_MidgreyY,
                          float p_TiltCR, float p_TiltGM, float p_TiltBY);

*/
private:
    // BOILERPLATE: Keep these basic members
    OFX::Image* _srcImg;
    
    // Replace all individual parameter variables with a single struct
    OpenDRTParams _params;  // Single struct instead of individual variables
    
    // Remove all the individual parameter variables that are currently declared
};

////////////////////////////////////////////////////////////////////////////////
// IMAGE PROCESSOR IMPLEMENTATION
////////////////////////////////////////////////////////////////////////////////
// BOILERPLATE: Constructor
ImageProcessor::ImageProcessor(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance)
{
}
////////////////////////////////////////////////////////////////////////////////
// GPU PROCESSING METHODS (CUDA/OPENCL/METAL)
// These call external kernel functions for GPU acceleration
////////////////////////////////////////////////////////////////////////////////
// Step 6: ADD EXTERNAL KERNEL DECLARATIONS HERE
// Make sure to pass the correct parameters to your kernels See Metal Example below
#ifndef __APPLE__
extern void RunCudaKernel(void* p_Stream, int p_Width, int p_Height, 
                         const float* p_Input, float* p_Output);
#endif

// Example Metal kernel (remove if not using Metal)
#ifdef __APPLE__
extern void OpenDRTKernel(void* p_CmdQ, int p_Width, int p_Height,
                          const float* p_Input, float* p_Output,
                          // Basic Parameters
                          int p_InGamut, int p_InOetf,
                          float p_TnLp, float p_TnGb, float p_PtHdr,
                          bool p_Clamp, float p_TnLg, float p_TnCon, float p_TnSh, float p_TnToe, float p_TnOff,
                          // UI ENABLE FLAGS (first group):
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
                          // NEW PARAMETERS - Filmic Mode and Advanced Controls
                          bool p_FilmicMode, float p_FilmicDynamicRange, int p_FilmicProjectorSim,
                          float p_FilmicSourceStops, float p_FilmicTargetStops, float p_FilmicStrength,
                          bool p_AdvHueContrast, bool p_TonescaleMap, bool p_DiagnosticsMode, bool p_RgbChipsMode, bool p_BetaFeaturesEnable,
                          int p_DisplayGamut, int p_Eotf);
#endif

// Example OpenCL kernel
extern void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, 
                           const float* p_Input, float* p_Output);

////////////////////////////////////////////////////////////////////////////////
// CORE  FUNCTIONS
////////////////////////////////////////////////////////////////////////////////// 
// Step 7: IMPLEMENT YOUR PARAMETER SETTER
// Replace with your own:
/*
void ImageProcessor::setYourParams(float param1, bool param2, int param3)
{
    _yourParameter1 = param1;
    _yourParameter2 = param2; 
    _yourParameter3 = param3;
}
*/
////////////////////////////////////////////////////////////////////////////////
// MAIN CPU PROCESSING FUNCTION
// This is where the actual pixel-by-pixel image processing happens
////////////////////////////////////////////////////////////////////////////////
// Step 8: IMPLEMENT PROCESSING METHODS
// Make sure to pass the correct parameters to your kernels See Metal Example below
void ImageProcessor::processImagesCUDA()
{
#ifndef __APPLE__
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunCudaKernel(_pCudaStream, width, height, input, output);
#endif
}

void ImageProcessor::processImagesMetal()
{
#ifdef __APPLE__
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    OpenDRTKernel(_pMetalCmdQ, width, height, input, output,
                  // Basic Parameters
                  _params.inGamut, _params.inOetf,
                  _params.tnLp, _params.tnGb, _params.ptHdr,
                  (_params.clamp != 0), _params.tnLg, _params.tnCon, _params.tnSh, _params.tnToe, _params.tnOff,
                  // UI ENABLE FLAGS (first group):
                  (_params.tnHconUIEnable != 0), (_params.tnLconUIEnable != 0),
                  (_params.ptlUIEnable != 0), (_params.ptmUIEnable != 0),
                  (_params.brlUIEnable != 0), (_params.hsRgbUIEnable != 0),
                  (_params.hsCmyUIEnable != 0), (_params.hcUIEnable != 0),
                  // PRESET ENABLE FLAGS:
                  (_params.tnHconPresetEnable != 0), (_params.tnLconPresetEnable != 0),
                  (_params.ptlPresetEnable != 0), (_params.ptmPresetEnable != 0),
                  (_params.brlPresetEnable != 0), (_params.hsRgbPresetEnable != 0),
                  (_params.hsCmyPresetEnable != 0), (_params.hcPresetEnable != 0),
                  // MODULE PARAMETERS:
                  _params.tnHcon, _params.tnHconPv, _params.tnHconSt,
                  _params.tnLcon, _params.tnLconW, _params.tnLconPc,
                  _params.cwp, _params.cwpRng,
                  _params.rsSa, _params.rsRw, _params.rsBw,
                  _params.ptR, _params.ptG, _params.ptB, _params.ptRngLow, _params.ptRngHigh,
                  _params.ptmLow, _params.ptmLowSt, _params.ptmHigh, _params.ptmHighSt,
                  _params.brlR, _params.brlG, _params.brlB,
                  _params.brlC, _params.brlM, _params.brlY, _params.brlRng,
                  _params.hsR, _params.hsG, _params.hsB, _params.hsRgbRng,
                  _params.hsC, _params.hsM, _params.hsY,
                  _params.hcR,
                  // NEW PARAMETERS - Filmic Mode and Advanced Controls
                  (_params.filmicMode != 0), _params.filmicDynamicRange, _params.filmicProjectorSim,
                  _params.filmicSourceStops, _params.filmicTargetStops, _params.filmicStrength,
                  (_params.advHueContrast != 0), (_params.tonescaleMap != 0), (_params.diagnosticsMode != 0), (_params.rgbChipsMode != 0), (_params.betaFeaturesEnable != 0),
                  _params.displayGamut, _params.eotf);
#endif
}

void ImageProcessor::processImagesOpenCL()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    // Replace with your own kernel call:
    RunOpenCLKernel(_pOpenCLCmdQ, width, height, input, output);
}

// BOILERPLATE: CPU fallback processing
void ImageProcessor::multiThreadProcessImages(OfxRectI p_ProcWindow)
{
    // Step 9: IMPLEMENT YOUR CPU PROCESSING HERE What happens when GPU is not available
    // This is a simple passthrough example - replace with your effect
    for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y)
    {
        if (_effect.abort()) break;

        float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));

        for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x)
        {
            float* srcPix = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(x, y) : 0);

            if (srcPix)
            {
                // EXAMPLE: Simple effect - replace with your processing
                dstPix[0] = srcPix[0]; // Red
                dstPix[1] = srcPix[1]; // Green
                dstPix[2] = srcPix[2]; // Blue
                dstPix[3] = srcPix[3]; // Alpha
            }
            else
            {
                // No src pixel, make it black and transparent
                for (int c = 0; c < 4; ++c)
                {
                    dstPix[c] = 0;
                }
            }
            dstPix += 4;
        }
    }
}
////////////////////////////////////////////////////////////////////////////////
// PARAMETER SETTER FUNCTION
// This receives all the UI parameter values and stores them for processing
////////////////////////////////////////////////////////////////////////////////
// BOILERPLATE: Basic setters
void ImageProcessor::setSrcImg(OFX::Image* p_SrcImg)
{
    _srcImg = p_SrcImg;
}



void ImageProcessor::setOpenDRTParams(
    // Input Settings
    int p_InGamut, int p_InOetf,
    // Tonescale Parameters
    float p_TnLp, float p_TnGb, float p_PtHdr,
    // Clamp Parameters
    bool p_Clamp, float p_TnLg, float p_TnCon, float p_TnSh, float p_TnToe, float p_TnOff,
    // High Contrast Parameters
    bool p_TnHconEnable, float p_TnHcon, float p_TnHconPv, float p_TnHconSt,
    // Low Contrast Parameters  
    bool p_TnLconEnable, float p_TnLcon, float p_TnLconW, float p_TnLconPc,
    // Creative White Parameters
    int p_Cwp, float p_CwpRng,
    // Render Space Parameters
    float p_RsSa, float p_RsRw, float p_RsBw,
    // Purity Compress Parameters
    float p_PtR, float p_PtG, float p_PtB, float p_PtRngLow, float p_PtRngHigh,
    // Purity Enable/Disable
    bool p_PtlEnable, bool p_PtmEnable,
    // Mid Purity Parameters
    float p_PtmLow, float p_PtmLowSt, float p_PtmHigh, float p_PtmHighSt,
    // Brilliance Parameters
    bool p_BrlEnable, float p_BrlR, float p_BrlG, float p_BrlB, 
    float p_BrlC, float p_BrlM, float p_BrlY, float p_BrlRng,
    // Hueshift RGB Parameters
    bool p_HsRgbEnable, float p_HsR, float p_HsG, float p_HsB, float p_HsRgbRng,
    // Hueshift CMY Parameters
    bool p_HsCmyEnable, float p_HsC, float p_HsM, float p_HsY,
    // Hue Contrast Parameters
    bool p_HcEnable, float p_HcR,
    // NEW PARAMETERS - Filmic Mode and Advanced Controls
    bool p_FilmicMode, float p_FilmicDynamicRange, int p_FilmicProjectorSim,
    float p_FilmicSourceStops, float p_FilmicTargetStops, float p_FilmicStrength,
    bool p_AdvHueContrast, bool p_TonescaleMap, bool p_DiagnosticsMode, bool p_RgbChipsMode, bool p_BetaFeaturesEnable,
    // Display Parameters
    int p_DisplayGamut, int p_Eotf,
    // Current preset index to look up enables
    int p_LookPreset
)
{
    // Store UI enable flags (from user checkboxes)
    _params.tnHconUIEnable = p_TnHconEnable ? 1 : 0;
    _params.tnLconUIEnable = p_TnLconEnable ? 1 : 0;
    _params.ptlUIEnable = p_PtlEnable ? 1 : 0;
    _params.ptmUIEnable = p_PtmEnable ? 1 : 0;
    _params.brlUIEnable = p_BrlEnable ? 1 : 0;
    _params.hsRgbUIEnable = p_HsRgbEnable ? 1 : 0;
    _params.hsCmyUIEnable = p_HsCmyEnable ? 1 : 0;
    _params.hcUIEnable = p_HcEnable ? 1 : 0;
    
    // Store preset enable flags (from current preset)
    if (p_LookPreset >= 0 && p_LookPreset <= 3) {
        const OpenDRTLookPreset& preset = LOOK_PRESETS[p_LookPreset];  // Direct mapping
        _params.tnHconPresetEnable = preset.tn_hcon_enable ? 1 : 0;
        _params.tnLconPresetEnable = preset.tn_lcon_enable ? 1 : 0;
        _params.ptlPresetEnable = preset.ptl_enable ? 1 : 0;
        _params.ptmPresetEnable = preset.ptm_enable ? 1 : 0;
        _params.brlPresetEnable = preset.brl_enable ? 1 : 0;
        _params.hsRgbPresetEnable = preset.hs_rgb_enable ? 1 : 0;
        _params.hsCmyPresetEnable = preset.hs_cmy_enable ? 1 : 0;
        _params.hcPresetEnable = preset.hc_enable ? 1 : 0;
    } else {
        // Invalid preset - all disabled
        _params.tnHconPresetEnable = 0;
        _params.tnLconPresetEnable = 0;
        _params.ptlPresetEnable = 0;
        _params.ptmPresetEnable = 0;
        _params.brlPresetEnable = 0;
        _params.hsRgbPresetEnable = 0;
        _params.hsCmyPresetEnable = 0;
        _params.hcPresetEnable = 0;
    }
    
    // Store other parameters (modify existing logic)
    _params.inGamut = p_InGamut;
    _params.inOetf = p_InOetf;
    _params.tnHcon = p_TnHcon;  // Don't zero out here - let Metal kernel decide
    _params.tnLcon = p_TnLcon;  // Don't zero out here - let Metal kernel decide
    _params.tnLp = p_TnLp;
    _params.tnGb = p_TnGb;
    _params.ptHdr = p_PtHdr;
    _params.clamp = p_Clamp ? 1 : 0;  // Convert bool to int
    _params.tnLg = p_TnLg;
    _params.tnCon = p_TnCon;
    _params.tnSh = p_TnSh;
    _params.tnToe = p_TnToe;
    _params.tnOff = p_TnOff;
    _params.tnHcon = p_TnHcon;
    _params.tnHconPv = p_TnHconPv;
    _params.tnHconSt = p_TnHconSt;
    _params.tnLcon = p_TnLcon;
    _params.tnLconW = p_TnLconW;
    _params.tnLconPc = p_TnLconPc;
int final_cwp = p_Cwp;
if (p_Cwp == 4) { // "Use Look Preset" selected
    // Get the current look preset's CWP value
    if (p_LookPreset >= 0 && p_LookPreset <= 3) {
        const OpenDRTLookPreset& preset = LOOK_PRESETS[p_LookPreset];
        final_cwp = preset.cwp; // Use the preset's CWP value (0-3)
    } else {
        final_cwp = 0; // Fallback to D65 if invalid preset
    }
}
_params.cwp = final_cwp; // Now guaranteed to be 0-3
    _params.cwpRng = p_CwpRng;
    _params.rsSa = p_RsSa;
    _params.rsRw = p_RsRw;
    _params.rsBw = p_RsBw;
    _params.ptR = p_PtR;
    _params.ptG = p_PtG;
    _params.ptB = p_PtB;
    _params.ptRngLow = p_PtRngLow;
    _params.ptRngHigh = p_PtRngHigh;
    _params.ptmLow = p_PtmLow;
    _params.ptmLowSt = p_PtmLowSt;
    _params.ptmHigh = p_PtmHigh;
    _params.ptmHighSt = p_PtmHighSt;
    _params.brlR = p_BrlR;
    _params.brlG = p_BrlG;
    _params.brlB = p_BrlB;
    _params.brlC = p_BrlC;
    _params.brlM = p_BrlM;
    _params.brlY = p_BrlY;
    _params.brlRng = p_BrlRng;
    _params.hsR = p_HsR;
    _params.hsG = p_HsG;
    _params.hsB = p_HsB;
    _params.hsRgbRng = p_HsRgbRng;
    _params.hsC = p_HsC;
    _params.hsM = p_HsM;
    _params.hsY = p_HsY;
    _params.hcR = p_HcR;
    
    // NEW PARAMETERS - Filmic Mode and Advanced Controls
    _params.filmicMode = p_FilmicMode ? 1 : 0;
    _params.filmicDynamicRange = p_FilmicDynamicRange;
    _params.filmicProjectorSim = p_FilmicProjectorSim;
    _params.filmicSourceStops = p_FilmicSourceStops;
    _params.filmicTargetStops = p_FilmicTargetStops;
    _params.filmicStrength = p_FilmicStrength;
    _params.advHueContrast = p_AdvHueContrast ? 1 : 0;
    _params.tonescaleMap = p_TonescaleMap ? 1 : 0;
    _params.diagnosticsMode = p_DiagnosticsMode ? 1 : 0;
    _params.rgbChipsMode = p_RgbChipsMode ? 1 : 0;
    _params.betaFeaturesEnable = p_BetaFeaturesEnable ? 1 : 0;
    
    _params.displayGamut = p_DisplayGamut;
    _params.eotf = p_Eotf;
}
/*
void ImageProcessor::setContrastParams(float p_GammaR, float p_GammaG, float p_GammaB, 
                                   float p_GammaC, float p_GammaM, float p_GammaY,
                                   float p_MidgreyR, float p_MidgreyG, float p_MidgreyB,
                                   float p_MidgreyC, float p_MidgreyM, float p_MidgreyY,
                                   float p_TiltCR, float p_TiltGM, float p_TiltBY)
{
    // Store RGB gamma values
    _gammaR = p_GammaR; _gammaG = p_GammaG; _gammaB = p_GammaB;
    
    // Store CYM gamma values
    _gammaC = p_GammaC; _gammaM = p_GammaM; _gammaY = p_GammaY;
    
    // Store RGB midgrey values
    _midgreyR = p_MidgreyR; _midgreyG = p_MidgreyG; _midgreyB = p_MidgreyB;
    
    // Store CYM midgrey values
    _midgreyC = p_MidgreyC; _midgreyM = p_MidgreyM; _midgreyY = p_MidgreyY;
    
    // Store tilt/blend values
    _tiltCR = p_TiltCR; _tiltGM = p_TiltGM; _tiltBY = p_TiltBY;
}
*/
////////////////////////////////////////////////////////////////////////////////
// MAIN PLUGIN CLASS - HANDLES OFX PLUGIN INTERFACE
////////////////////////////////////////////////////////////////////////////////
// The plugin that does our work */

// BOILERPLATE: Main plugin class - rename if needed Hint Change all Class names with update all occurences
//OpenDRT Should match everywhere in the code
class OpenDRT : public OFX::ImageEffect  // <-- RENAME: Change class name
{
public:
    explicit OpenDRT(OfxImageEffectHandle p_Handle);

    // BOILERPLATE: Keep these overrides
    virtual void render(const OFX::RenderArguments& p_Args);
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);
    virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);
    virtual void changedClip(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ClipName);

    void setEnabledness();
    void setupAndProcess(ImageProcessor &p_Processor, const OFX::RenderArguments& p_Args);

private:
    // BOILERPLATE: Keep these basic clips
    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;

    // Add preset application method
    void applyPresetValues(const OFX::InstanceChangedArgs& p_Args);

    // Group Parameter Pointers
    OFX::GroupParam* m_HighContrastGroup;
    OFX::GroupParam* m_LowContrastGroup;
    OFX::GroupParam* m_PurityLowGroup;
    OFX::GroupParam* m_MidPurityGroup;
    OFX::GroupParam* m_BrillianceGroup;
    OFX::GroupParam* m_HueshiftRgbGroup;
    OFX::GroupParam* m_HueshiftCmyGroup;
    OFX::GroupParam* m_HueContrastGroup;

    ////////////////////////////////////////////////////////////////////////////////
    // PARAMETER POINTERS - CONNECTIONS TO UI CONTROLS
    ////////////////////////////////////////////////////////////////////////////////


    
    // OpenDRT Parameters
    // Input Settings
    OFX::ChoiceParam* m_InGamut;
    OFX::ChoiceParam* m_InOetf;
    
    // Tonescale Parameters
    OFX::DoubleParam* m_TnLp;          // Display Peak Luminance
    OFX::DoubleParam* m_TnGb;          // HDR Grey Boost
    OFX::DoubleParam* m_PtHdr;         // HDR Purity
    
    // Clamp Parameters
    OFX::BooleanParam* m_Clamp;
    OFX::DoubleParam* m_TnLg;          // Grey Luminance
    OFX::DoubleParam* m_TnCon;         // Contrast
    OFX::DoubleParam* m_TnSh;          // Shoulder Clip
    OFX::DoubleParam* m_TnToe;         // Toe
    OFX::DoubleParam* m_TnOff;         // Offset
    
    // High Contrast Parameters
    OFX::BooleanParam* m_TnHconEnable;
    OFX::DoubleParam* m_TnHcon;        // Contrast High
    OFX::DoubleParam* m_TnHconPv;      // Contrast High Pivot
    OFX::DoubleParam* m_TnHconSt;      // Contrast High Strength
    
    // Low Contrast Parameters
    OFX::BooleanParam* m_TnLconEnable;
    OFX::DoubleParam* m_TnLcon;        // Contrast Low
    OFX::DoubleParam* m_TnLconW;       // Contrast Low Width
    OFX::DoubleParam* m_TnLconPc;      // Contrast Low Per-Channel
    
    // Creative White Parameters
    OFX::ChoiceParam* m_Cwp;           // Creative White
    OFX::DoubleParam* m_CwpRng;        // Creative White Range
    
    // Render Space Parameters
    OFX::DoubleParam* m_RsSa;          // Render Space Strength
    OFX::DoubleParam* m_RsRw;          // Render Space Red Weight
    OFX::DoubleParam* m_RsBw;          // Render Space Blue Weight
    
    // Purity Compress Parameters
    OFX::DoubleParam* m_PtR;           // Purity Compress R
    OFX::DoubleParam* m_PtG;           // Purity Compress G
    OFX::DoubleParam* m_PtB;           // Purity Compress B
    OFX::DoubleParam* m_PtRngLow;      // Purity Range Low
    OFX::DoubleParam* m_PtRngHigh;     // Purity Range High
    
    // Purity Enable/Disable
    OFX::BooleanParam* m_PtlEnable;    // Enable Purity Compress Low
    OFX::BooleanParam* m_PtmEnable;    // Enable Mid Purity
    
    // Mid Purity Parameters
    OFX::DoubleParam* m_PtmLow;        // Mid Purity Low
    OFX::DoubleParam* m_PtmLowSt;      // Mid Purity Low Strength
    OFX::DoubleParam* m_PtmHigh;       // Mid Purity High
    OFX::DoubleParam* m_PtmHighSt;     // Mid Purity High Strength
    
    // Brilliance Parameters
    OFX::BooleanParam* m_BrlEnable;    // Enable Brilliance
    OFX::DoubleParam* m_BrlR;          // Brilliance R
    OFX::DoubleParam* m_BrlG;          // Brilliance G
    OFX::DoubleParam* m_BrlB;          // Brilliance B
    OFX::DoubleParam* m_BrlC;          // Brilliance C
    OFX::DoubleParam* m_BrlM;          // Brilliance M
    OFX::DoubleParam* m_BrlY;          // Brilliance Y
    OFX::DoubleParam* m_BrlRng;        // Brilliance Range
    
    // Hueshift RGB Parameters
    OFX::BooleanParam* m_HsRgbEnable;  // Enable Hueshift RGB
    OFX::DoubleParam* m_HsR;           // Hueshift R
    OFX::DoubleParam* m_HsG;           // Hueshift G
    OFX::DoubleParam* m_HsB;           // Hueshift B
    OFX::DoubleParam* m_HsRgbRng;      // Hueshift RGB Range
    
    // Hueshift CMY Parameters
    OFX::BooleanParam* m_HsCmyEnable;  // Enable Hueshift CMY
    OFX::DoubleParam* m_HsC;           // Hueshift C
    OFX::DoubleParam* m_HsM;           // Hueshift M
    OFX::DoubleParam* m_HsY;           // Hueshift Y
    
    // Hue Contrast Parameters
    OFX::BooleanParam* m_HcEnable;     // Enable Hue Contrast
    OFX::DoubleParam* m_HcR;           // Hue Contrast R
    
    // Advanced Hue Contrast Parameters
    OFX::DoubleParam* m_AdvHcG;        // Advanced Hue Contrast G
    OFX::DoubleParam* m_AdvHcB;        // Advanced Hue Contrast B
    OFX::DoubleParam* m_AdvHcC;        // Advanced Hue Contrast C
    OFX::DoubleParam* m_AdvHcM;        // Advanced Hue Contrast M
    OFX::DoubleParam* m_AdvHcY;        // Advanced Hue Contrast Y
    
    // Display Parameters
    OFX::ChoiceParam* m_DisplayGamut;  // Display Gamut
    OFX::ChoiceParam* m_Eotf;          // Display EOTF
    
    // Preset Parameters
    OFX::ChoiceParam* m_LookPreset;    // Look Preset
    OFX::ChoiceParam* m_TonescalePreset; // Tonescale Preset
        // Lock parameter
    OFX::BooleanParam* m_LockStickshift; // Lock Stickshift

    // Individual Lock Parameters (ADD THESE HERE)
    OFX::BooleanParam* m_LockHighContrast;
    OFX::BooleanParam* m_LockLowContrast;
    OFX::BooleanParam* m_LockPurityLow;
    OFX::BooleanParam* m_LockMidPurity;
    OFX::BooleanParam* m_LockBrilliance;
    OFX::BooleanParam* m_LockHueshiftRgb;
    OFX::BooleanParam* m_LockHueshiftCmy;
    OFX::BooleanParam* m_LockHueContrast;

    // NEW PARAMETERS - Filmic Mode and Advanced Controls
    OFX::BooleanParam* m_FilmicMode;            // Enable Filmic Mode
    OFX::BooleanParam* m_AdvHueContrast;        // Advanced Hue Contrast
    OFX::BooleanParam* m_TonescaleMap;          // Tonescale Map
    OFX::BooleanParam* m_DiagnosticsMode;       // Diagnostics Mode
    OFX::BooleanParam* m_RgbChipsMode;          // RGB Chips Mode
    OFX::DoubleParam* m_FilmicDynamicRange;     // Filmic Dynamic Range
    OFX::ChoiceParam* m_FilmicProjectorSim;     // Filmic Projector Sim
    OFX::BooleanParam* m_BetaFeaturesEnable;    // Enable Beta Features
    OFX::BooleanParam* m_RgbChipsEnable;  // NEW: Enable RGB Chips parameter
    
    // NEW FILMIC PARAMETERS
    OFX::DoubleParam* m_FilmicSourceStops;      // Original Camera Range
    OFX::DoubleParam* m_FilmicTargetStops;      // Target Film Range
    OFX::DoubleParam* m_FilmicStrength;         // Filmic Strength
    
    // NEW GROUP PARAMETERS
    OFX::GroupParam* m_DiagnosticsGroup;        // Diagnostics Group
    OFX::GroupParam* m_FilmicDynamicRangeGroup; // Filmic Dynamic Range Group
    OFX::GroupParam* m_FilmicProjectorSimGroup; // Filmic Projector Sim Group
    OFX::GroupParam* m_BetaFeaturesGroup;       // Beta Features Group
    OFX::GroupParam* m_AdvHueContrastGroup;     // Advanced Hue Contrast Group

    OFX::PushButtonParam* m_CoffeeButton;        // Coffee support button
};
////////////////////////////////////////////////////////////////////////////////
// PLUGIN CONSTRUCTOR - CONNECTS TO ALL THE UI PARAMETERS
////////////////////////////////////////////////////////////////////////////////
// BOILERPLATE: Constructor - update parameter names
OpenDRT::OpenDRT(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    // Connect to input/output clips Don't Chage This
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

    // Fetch group parameters
    m_HighContrastGroup = fetchGroupParam("HighContrastGroup");
    m_LowContrastGroup = fetchGroupParam("LowContrastGroup");
    m_PurityLowGroup = fetchGroupParam("PurityLowGroup");
    m_MidPurityGroup = fetchGroupParam("MidPurityGroup");
    m_BrillianceGroup = fetchGroupParam("BrillianceGroup");
    m_HueshiftRgbGroup = fetchGroupParam("HueshiftRgbGroup");
    m_HueshiftCmyGroup = fetchGroupParam("HueshiftCmyGroup");
    m_HueContrastGroup = fetchGroupParam("HueContrastGroup");

    // NEW GROUP PARAMETERS
    m_DiagnosticsGroup = fetchGroupParam("DiagnosticsGroup");
    m_FilmicDynamicRangeGroup = fetchGroupParam("FilmicDynamicRangeGroup");
    m_FilmicProjectorSimGroup = fetchGroupParam("FilmicProjectorSimGroup");
    m_BetaFeaturesGroup = fetchGroupParam("BetaFeaturesGroup");
    m_AdvHueContrastGroup = fetchGroupParam("AdvHueContrastGroup");

    // Connect to all the parameter controls created in describeInContext()

    
    // OpenDRT Parameters
    // Input Settings
    m_InGamut = fetchChoiceParam("in_gamut");
    m_InOetf = fetchChoiceParam("in_oetf");
    
    // Tonescale Parameters
    m_TnLp = fetchDoubleParam("tn_Lp");
    m_TnGb = fetchDoubleParam("tn_gb");
    m_PtHdr = fetchDoubleParam("pt_hdr");
    
    // Clamp Parameters
    m_Clamp = fetchBooleanParam("_clamp");
    m_TnLg = fetchDoubleParam("_tn_Lg");
    m_TnCon = fetchDoubleParam("_tn_con");
    m_TnSh = fetchDoubleParam("_tn_sh");
    m_TnToe = fetchDoubleParam("_tn_toe");
    m_TnOff = fetchDoubleParam("_tn_off");
    
    // High Contrast Parameters
    m_TnHconEnable = fetchBooleanParam("_tn_hcon_enable");
    m_TnHcon = fetchDoubleParam("_tn_hcon");
    m_TnHconPv = fetchDoubleParam("_tn_hcon_pv");
    m_TnHconSt = fetchDoubleParam("_tn_hcon_st");
    
    // Low Contrast Parameters
    m_TnLconEnable = fetchBooleanParam("_tn_lcon_enable");
    m_TnLcon = fetchDoubleParam("_tn_lcon");
    m_TnLconW = fetchDoubleParam("_tn_lcon_w");
    m_TnLconPc = fetchDoubleParam("_tn_lcon_pc");
    
    // Creative White Parameters
    m_Cwp = fetchChoiceParam("_cwp");
    m_CwpRng = fetchDoubleParam("_cwp_rng");
    
    // Render Space Parameters
    m_RsSa = fetchDoubleParam("_rs_sa");
    m_RsRw = fetchDoubleParam("_rs_rw");
    m_RsBw = fetchDoubleParam("_rs_bw");
    
    // Purity Compress Parameters
    m_PtR = fetchDoubleParam("_pt_r");
    m_PtG = fetchDoubleParam("_pt_g");
    m_PtB = fetchDoubleParam("_pt_b");
    m_PtRngLow = fetchDoubleParam("_pt_rng_low");
    m_PtRngHigh = fetchDoubleParam("_pt_rng_high");
    
    // Purity Enable/Disable
    m_PtlEnable = fetchBooleanParam("_ptl_enable");
    m_PtmEnable = fetchBooleanParam("_ptm_enable");
    
    // Mid Purity Parameters
    m_PtmLow = fetchDoubleParam("_ptm_low");
    m_PtmLowSt = fetchDoubleParam("_ptm_low_st");
    m_PtmHigh = fetchDoubleParam("_ptm_high");
    m_PtmHighSt = fetchDoubleParam("_ptm_high_st");
    
    // Brilliance Parameters
    m_BrlEnable = fetchBooleanParam("_brl_enable");
    m_BrlR = fetchDoubleParam("_brl_r");
    m_BrlG = fetchDoubleParam("_brl_g");
    m_BrlB = fetchDoubleParam("_brl_b");
    m_BrlC = fetchDoubleParam("_brl_c");
    m_BrlM = fetchDoubleParam("_brl_m");
    m_BrlY = fetchDoubleParam("_brl_y");
    m_BrlRng = fetchDoubleParam("_brl_rng");
    
    // Hueshift RGB Parameters
    m_HsRgbEnable = fetchBooleanParam("_hs_rgb_enable");
    m_HsR = fetchDoubleParam("_hs_r");
    m_HsG = fetchDoubleParam("_hs_g");
    m_HsB = fetchDoubleParam("_hs_b");
    m_HsRgbRng = fetchDoubleParam("_hs_rgb_rng");
    
    // Hueshift CMY Parameters
    m_HsCmyEnable = fetchBooleanParam("_hs_cmy_enable");
    m_HsC = fetchDoubleParam("_hs_c");
    m_HsM = fetchDoubleParam("_hs_m");
    m_HsY = fetchDoubleParam("_hs_y");
    
    // Hue Contrast Parameters
    m_HcEnable = fetchBooleanParam("_hc_enable");
    m_HcR = fetchDoubleParam("_hc_r");
    
    // Advanced Hue Contrast Parameters
    m_AdvHcG = fetchDoubleParam("_adv_hc_g");
    m_AdvHcB = fetchDoubleParam("_adv_hc_b");
    m_AdvHcC = fetchDoubleParam("_adv_hc_c");
    m_AdvHcM = fetchDoubleParam("_adv_hc_m");
    m_AdvHcY = fetchDoubleParam("_adv_hc_y");
    
    // NEW PARAMETERS - Filmic Mode and Advanced Controls
    m_FilmicMode = fetchBooleanParam("_filmic_mode");
    m_AdvHueContrast = fetchBooleanParam("_adv_hue_contrast");
    m_TonescaleMap = fetchBooleanParam("_tonescale_map");
    m_DiagnosticsMode = fetchBooleanParam("_diagnostics_mode");
    m_RgbChipsMode = fetchBooleanParam("_rgbchips");
    m_FilmicDynamicRange = fetchDoubleParam("_filmic_dynamic_range");
    m_FilmicProjectorSim = fetchChoiceParam("_filmic_projector_sim");
    m_BetaFeaturesEnable = fetchBooleanParam("_beta_features_enable");
    
    // NEW FILMIC PARAMETERS
    m_FilmicSourceStops = fetchDoubleParam("_filmic_source_stops");
    m_FilmicTargetStops = fetchDoubleParam("_filmic_target_stops");
    m_FilmicStrength = fetchDoubleParam("_filmic_strength");
    
    // Display Parameters
    m_DisplayGamut = fetchChoiceParam("_display_gamut");
    m_Eotf = fetchChoiceParam("_eotf");
    
    // Preset Parameters - NO underscores for these
    m_LookPreset = fetchChoiceParam("look_preset");
    m_TonescalePreset = fetchChoiceParam("tonescale_preset");
    // Lock parameter - NOTE: Fixed parameter name
    m_LockStickshift = fetchBooleanParam("lockStickshift");

    // Fetch individual lock parameters
    m_LockHighContrast = fetchBooleanParam("_lock_hcon");
    m_LockLowContrast = fetchBooleanParam("_lock_lcon");
    m_LockPurityLow = fetchBooleanParam("_lock_ptl");
    m_LockMidPurity = fetchBooleanParam("_lock_ptm");
    m_LockBrilliance = fetchBooleanParam("_lock_brl");
    m_LockHueshiftRgb = fetchBooleanParam("_lock_hs_rgb");
    m_LockHueshiftCmy = fetchBooleanParam("_lock_hs_cmy");
    m_LockHueContrast = fetchBooleanParam("_lock_hc");

    m_CoffeeButton = fetchPushButtonParam("buymeacoffee");
    setEnabledness();
}

////////////////////////////////////////////////////////////////////////////////
// RENDER FUNCTION - MAIN ENTRY POINT FOR PROCESSING
////////////////////////////////////////////////////////////////////////////////
// BOILERPLATE: Keep render method structure
void OpenDRT::render(const OFX::RenderArguments& p_Args)
{
    if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && 
        (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
    {
        ImageProcessor processor(*this);  // <-- UPDATE: Match your class name
        setupAndProcess(processor, p_Args);
    }
    else
    {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}
////////////////////////////////////////////////////////////////////////////////
// IDENTITY CHECK - OPTIMIZATION TO SKIP PROCESSING WHEN NO CHANGE WOULD OCCUR
////////////////////////////////////////////////////////////////////////////////
bool OpenDRT::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
    
    
    // Check if all OpenDRT parameters are at their "pass-through" values
    double tnLp = m_TnLp->getValueAtTime(p_Args.time);
    double tnGb = m_TnGb->getValueAtTime(p_Args.time);
    double ptHdr = m_PtHdr->getValueAtTime(p_Args.time);
    bool clamp = m_Clamp->getValueAtTime(p_Args.time);
    
    // Check if processing is effectively disabled
    bool allFeaturesDisabled = !clamp && 
                              !m_TnHconEnable->getValueAtTime(p_Args.time) &&
                              !m_TnLconEnable->getValueAtTime(p_Args.time) &&
                              !m_PtlEnable->getValueAtTime(p_Args.time) &&
                              !m_PtmEnable->getValueAtTime(p_Args.time) &&
                              !m_BrlEnable->getValueAtTime(p_Args.time) &&
                              !m_HsRgbEnable->getValueAtTime(p_Args.time) &&
                              !m_HsCmyEnable->getValueAtTime(p_Args.time) &&
                              !m_HcEnable->getValueAtTime(p_Args.time);
                              
    // If all features are disabled, pass through unchanged
    if (allFeaturesDisabled)
    {
        p_IdentityClip = m_SrcClip;
        p_IdentityTime = p_Args.time;
        return true;
    }
    
    return false;
}
////////////////////////////////////////////////////////////////////////////////
// PARAMETER CHANGE HANDLER - RESPONDS TO UI CHANGES
////////////////////////////////////////////////////////////////////////////////
void OpenDRT::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
    // Handle preset parameter changes
    if (p_ParamName == "look_preset" || p_ParamName == "tonescale_preset")
    {
        // Apply preset values when presets change
        applyPresetValues(p_Args);
    }
    
    // Handle OpenDRT parameter changes
    if (p_ParamName == "_clamp" || 
        p_ParamName == "_tn_hcon_enable" || 
        p_ParamName == "_tn_lcon_enable" ||
        p_ParamName == "_ptl_enable" ||
        p_ParamName == "_ptm_enable" ||
        p_ParamName == "_brl_enable" ||
        p_ParamName == "_hs_rgb_enable" ||
        p_ParamName == "_hs_cmy_enable" ||
        p_ParamName == "_hc_enable" ||
        p_ParamName == "_filmic_mode" ||
        p_ParamName == "_adv_hue_contrast" ||
        p_ParamName == "_tonescale_map" ||
        p_ParamName == "_diagnostics_mode" ||
        p_ParamName == "_rgbchips" ||
        p_ParamName == "_beta_features_enable")
    {
        setEnabledness();
    }
        // ADD THIS NEW SECTION:
    else if (p_ParamName == "buymeacoffee")
    {
        // Open Buy Me a Coffee link
        std::string url = "https://dec18studios.com/open-drt-ofx"; // Replace with your actual URL
        
#ifdef __APPLE__
        std::string command = "open \"" + url + "\"";
#elif defined(_WIN32)
        std::string command = "start \"\" \"" + url + "\"";
#else
        std::string command = "xdg-open \"" + url + "\"";
#endif
        
        system(command.c_str());
    }

    
}
////////////////////////////////////////////////////////////////////////////////
// CLIP CHANGE HANDLER - RESPONDS TO INPUT CHANGES
////////////////////////////////////////////////////////////////////////////////
void OpenDRT::changedClip(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ClipName)
{
    if (p_ClipName == kOfxImageEffectSimpleSourceClipName)
    {
        setEnabledness();
    }
}
////////////////////////////////////////////////////////////////////////////////
// UI CONTROL ENABLEMENT - MANAGES WHICH CONTROLS ARE AVAILABLE
////////////////////////////////////////////////////////////////////////////////
void OpenDRT::setEnabledness()
{
    // Check input clip
    bool hasInput = (m_SrcClip && m_SrcClip->isConnected());
    
    // Get enable states
    bool clampEnabled = m_Clamp->getValue();
    bool hconEnabled = m_TnHconEnable->getValue();
    bool lconEnabled = m_TnLconEnable->getValue();
    bool ptlEnabled = m_PtlEnable->getValue();
    bool ptmEnabled = m_PtmEnable->getValue();
    bool brlEnabled = m_BrlEnable->getValue();
    bool hsRgbEnabled = m_HsRgbEnable->getValue();
    bool hsCmyEnabled = m_HsCmyEnable->getValue();
    bool hcEnabled = m_HcEnable->getValue();
    
    // NEW ENABLE STATES
    bool filmicModeEnabled = m_FilmicMode->getValue();
    bool diagnosticsEnabled = m_DiagnosticsMode->getValue();
    bool rgbChipsEnabled = m_RgbChipsMode->getValue();
    bool betaFeaturesEnabled = m_BetaFeaturesEnable->getValue();
    bool advHueContrastEnabled = m_AdvHueContrast->getValue();
    
    // Show/hide groups based on enable states using setIsSecret()
    m_HighContrastGroup->setIsSecret(!hconEnabled);
    m_LowContrastGroup->setIsSecret(!lconEnabled);
    m_PurityLowGroup->setIsSecret(!ptlEnabled);
    m_MidPurityGroup->setIsSecret(!ptmEnabled);
    m_BrillianceGroup->setIsSecret(!brlEnabled);
    m_HueshiftRgbGroup->setIsSecret(!hsRgbEnabled);
    m_HueshiftCmyGroup->setIsSecret(!hsCmyEnabled);
    m_HueContrastGroup->setIsSecret(!hcEnabled);
    
    // NEW GROUP VISIBILITY CONTROL
    // Diagnostics Group - Always visible but closed
    m_DiagnosticsGroup->setIsSecret(false);
    
    // ✅ STEP 1: First enable/disable the mode parameters based on Beta Features
    m_FilmicMode->setEnabled(betaFeaturesEnabled);
    m_AdvHueContrast->setEnabled(betaFeaturesEnabled);
    
    // ✅ STEP 2: Then show/hide groups based on BOTH conditions
    // If Beta Features is OFF, hide everything regardless of individual mode states
    // If Beta Features is ON, show groups only when individual modes are also ON
    bool showFilmicGroups = betaFeaturesEnabled && filmicModeEnabled;
    bool showAdvHueContrastGroup = betaFeaturesEnabled && advHueContrastEnabled;
    
    m_FilmicDynamicRangeGroup->setIsSecret(!showFilmicGroups);
    m_FilmicProjectorSimGroup->setIsSecret(!showFilmicGroups);
    m_AdvHueContrastGroup->setIsSecret(!showAdvHueContrastGroup);
    
    // Beta Features - Control individual parameter enablement (not group visibility)
    m_FilmicMode->setEnabled(betaFeaturesEnabled);
    m_AdvHueContrast->setEnabled(betaFeaturesEnabled);
    
    // The individual parameters within the groups are automatically handled
    // by the group visibility, so we can remove the individual setEnabled calls
    
}

////////////////////////////////////////////////////////////////////////////////
// PRESET APPLICATION - APPLIES PRESET VALUES TO PARAMETERS
////////////////////////////////////////////////////////////////////////////////
void OpenDRT::applyPresetValues(const OFX::InstanceChangedArgs& p_Args)
{
    int lookPreset;
    m_LookPreset->getValueAtTime(p_Args.time, lookPreset);
    
    int tonescalePreset;
    m_TonescalePreset->getValueAtTime(p_Args.time, tonescalePreset);
    
    // Check if stickshift adjustments should be locked
    bool lockStickshift = m_LockStickshift->getValue();
    
    // GET INDIVIDUAL LOCK VALUES (ADD THESE LINES):
    bool lockHighContrast = m_LockHighContrast->getValue();
    bool lockLowContrast = m_LockLowContrast->getValue();
    bool lockPurityLow = m_LockPurityLow->getValue();
    bool lockMidPurity = m_LockMidPurity->getValue();
    bool lockBrilliance = m_LockBrilliance->getValue();
    bool lockHueshiftRgb = m_LockHueshiftRgb->getValue();
    bool lockHueshiftCmy = m_LockHueshiftCmy->getValue();
    bool lockHueContrast = m_LockHueContrast->getValue();
    
    // Apply look preset values
    if (lookPreset >= 0 && lookPreset <= 4) {
        const OpenDRTLookPreset& preset = LOOK_PRESETS[lookPreset];
        
        // Always apply basic tonescale parameters (never locked)
        m_TnLg->setValue(preset.tn_Lg);
        m_TnCon->setValue(preset.tn_con);
        m_TnSh->setValue(preset.tn_sh);
        m_TnToe->setValue(preset.tn_toe);
        m_TnOff->setValue(preset.tn_off);
        
        // Always apply creative white parameters (never locked)
        m_Cwp->setValue(preset.cwp);
        m_CwpRng->setValue(preset.cwp_rng);
        
        // Always apply render space parameters (never locked)
        m_RsSa->setValue(preset.rs_sa);
        m_RsRw->setValue(preset.rs_rw);
        m_RsBw->setValue(preset.rs_bw);
        
        // GRANULAR LOCK LOGIC: Apply preset only if specific module is NOT locked
        
        // High Contrast: Apply only if not locked
        if (!lockHighContrast) {  // ✅ Now using the VALUE
            m_TnHcon->setValue(preset.tn_hcon);
            m_TnHconPv->setValue(preset.tn_hcon_pv);
            m_TnHconSt->setValue(preset.tn_hcon_st);
        }
        
        // Low Contrast: Apply only if not locked
        if (!lockLowContrast) {  // ✅ Now using the VALUE
            m_TnLcon->setValue(preset.tn_lcon);
            m_TnLconW->setValue(preset.tn_lcon_w);
            m_TnLconPc->setValue(preset.tn_lcon_pc);
        }
        
        // Purity Low: Apply only if not locked
        if (!lockPurityLow) {  // ✅ Now using the VALUE
            m_PtR->setValue(preset.pt_r);
            m_PtG->setValue(preset.pt_g);
            m_PtB->setValue(preset.pt_b);
            m_PtRngLow->setValue(preset.pt_rng_low);
            m_PtRngHigh->setValue(preset.pt_rng_high);
        }
        
        // Mid Purity: Apply only if not locked
        if (!lockMidPurity) {  // ✅ Now using the VALUE
            m_PtmLow->setValue(preset.ptm_low);
            m_PtmLowSt->setValue(preset.ptm_low_st);
            m_PtmHigh->setValue(preset.ptm_high);
            m_PtmHighSt->setValue(preset.ptm_high_st);
        }
        
        // Brilliance: Apply only if not locked
        if (!lockBrilliance) {  // ✅ Now using the VALUE
            m_BrlR->setValue(preset.brl_r);
            m_BrlG->setValue(preset.brl_g);
            m_BrlB->setValue(preset.brl_b);
            m_BrlC->setValue(preset.brl_c);
            m_BrlM->setValue(preset.brl_m);
            m_BrlY->setValue(preset.brl_y);
            m_BrlRng->setValue(preset.brl_rng);
        }
        
        // Hueshift RGB: Apply only if not locked
        if (!lockHueshiftRgb) {  // ✅ Now using the VALUE
            m_HsR->setValue(preset.hs_r);
            m_HsG->setValue(preset.hs_g);
            m_HsB->setValue(preset.hs_b);
            m_HsRgbRng->setValue(preset.hs_rgb_rng);
        }
        
        // Hueshift CMY: Apply only if not locked
        if (!lockHueshiftCmy) {  // ✅ Now using the VALUE
            m_HsC->setValue(preset.hs_c);
            m_HsM->setValue(preset.hs_m);
            m_HsY->setValue(preset.hs_y);
        }
        
        // Hue Contrast: Apply only if not locked
        if (!lockHueContrast) {  // ✅ Now using the VALUE
            m_HcR->setValue(preset.hc_r);
        }
    }
    
// Apply tonescale preset
if (tonescalePreset == 0) {
    // "Use Look Preset" - don't apply anything, just use the look preset's tonescale
    // This is handled by the look preset logic above
} else if (tonescalePreset >= 1 && tonescalePreset <= 9) {
    const OpenDRTTonescalePreset& preset = TONESCALE_PRESETS[tonescalePreset - 1];  // Offset by 1set];
        
        // Always apply basic tonescale values (never locked)
        m_TnLg->setValue(preset.tn_Lg);
        m_TnCon->setValue(preset.tn_con);
        m_TnSh->setValue(preset.tn_sh);
        m_TnToe->setValue(preset.tn_toe);
        m_TnOff->setValue(preset.tn_off);
        
        // Apply contrast settings only if not locked
        if (!lockHighContrast) {  // ✅ Now using the VALUE
            m_TnHcon->setValue(preset.tn_hcon);
            m_TnHconPv->setValue(preset.tn_hcon_pv);
            m_TnHconSt->setValue(preset.tn_hcon_st);
        }
        
        if (!lockLowContrast) {  // ✅ Now using the VALUE
            m_TnLcon->setValue(preset.tn_lcon);
            m_TnLconW->setValue(preset.tn_lcon_w);
            m_TnLconPc->setValue(preset.tn_lcon_pc);
        }
    }
    
    // Update enablement after applying presets
    setEnabledness();
}

////////////////////////////////////////////////////////////////////////////////
// SETUP AND PROCESS - COORDINATES THE ENTIRE PROCESSING PIPELINE
////////////////////////////////////////////////////////////////////////////////
// Step 15: UPDATE PARAMETER FETCHING AND PROCESSING
void OpenDRT::setupAndProcess(ImageProcessor& p_Processor, const OFX::RenderArguments& p_Args)
{
   
    ////////////////////////////////////////////////////////////////////////////////
    // IMAGE SETUP - GET INPUT AND OUTPUT IMAGES
    ////////////////////////////////////////////////////////////////////////////////
     // BOILERPLATE: Keep image setup
    // Get the destination (output) image
    std::unique_ptr<OFX::Image> dst(m_DstClip->fetchImage(p_Args.time));
    OFX::BitDepthEnum dstBitDepth = dst->getPixelDepth();
    OFX::PixelComponentEnum dstComponents = dst->getPixelComponents();

    // Get the source (input) image
    std::unique_ptr<OFX::Image> src(m_SrcClip->fetchImage(p_Args.time));
    OFX::BitDepthEnum srcBitDepth = src->getPixelDepth();
    OFX::PixelComponentEnum srcComponents = src->getPixelComponents();

    // Verify input and output are compatible
    if ((srcBitDepth != dstBitDepth) || (srcComponents != dstComponents))
    {
        OFX::throwSuiteStatusException(kOfxStatErrValue);
    }
    ////////////////////////////////////////////////////////////////////////////////
    // PARAMETER EXTRACTION WITH BLEND MODE AND GANG LOGIC
    ////////////////////////////////////////////////////////////////////////////////
    // Get all OpenDRT parameter values

    // OpenDRT Parameters
    // Input Settings
    int inGamut;
    m_InGamut->getValueAtTime(p_Args.time, inGamut);
    int inOetf;
    m_InOetf->getValueAtTime(p_Args.time, inOetf);
    
    // Tonescale Parameters
    float tnLp = m_TnLp->getValueAtTime(p_Args.time);
    float tnGb = m_TnGb->getValueAtTime(p_Args.time);
    float ptHdr = m_PtHdr->getValueAtTime(p_Args.time);
    
    // Clamp Parameters
    bool clamp = m_Clamp->getValueAtTime(p_Args.time);
    float tnLg = m_TnLg->getValueAtTime(p_Args.time);
    float tnCon = m_TnCon->getValueAtTime(p_Args.time);
    float tnSh = m_TnSh->getValueAtTime(p_Args.time);
    float tnToe = m_TnToe->getValueAtTime(p_Args.time);
    float tnOff = m_TnOff->getValueAtTime(p_Args.time);
    
    // High Contrast Parameters
    bool tnHconEnable = m_TnHconEnable->getValueAtTime(p_Args.time);
    float tnHcon = m_TnHcon->getValueAtTime(p_Args.time);
    float tnHconPv = m_TnHconPv->getValueAtTime(p_Args.time);
    float tnHconSt = m_TnHconSt->getValueAtTime(p_Args.time);
    
    // Low Contrast Parameters
    bool tnLconEnable = m_TnLconEnable->getValueAtTime(p_Args.time);
    float tnLcon = m_TnLcon->getValueAtTime(p_Args.time);
    float tnLconW = m_TnLconW->getValueAtTime(p_Args.time);
    float tnLconPc = m_TnLconPc->getValueAtTime(p_Args.time);
    
    // Creative White Parameters
    int cwp;
    m_Cwp->getValueAtTime(p_Args.time, cwp);
    float cwpRng = m_CwpRng->getValueAtTime(p_Args.time);
    
    // Render Space Parameters
    float rsSa = m_RsSa->getValueAtTime(p_Args.time);
    float rsRw = m_RsRw->getValueAtTime(p_Args.time);
    float rsBw = m_RsBw->getValueAtTime(p_Args.time);
    
    // Purity Compress Parameters
    float ptR = m_PtR->getValueAtTime(p_Args.time);
    float ptG = m_PtG->getValueAtTime(p_Args.time);
    float ptB = m_PtB->getValueAtTime(p_Args.time);
    float ptRngLow = m_PtRngLow->getValueAtTime(p_Args.time);
    float ptRngHigh = m_PtRngHigh->getValueAtTime(p_Args.time);
    
    // Purity Enable/Disable
    bool ptlEnable = m_PtlEnable->getValueAtTime(p_Args.time);
    bool ptmEnable = m_PtmEnable->getValueAtTime(p_Args.time);
    
    // Mid Purity Parameters
    float ptmLow = m_PtmLow->getValueAtTime(p_Args.time);
    float ptmLowSt = m_PtmLowSt->getValueAtTime(p_Args.time);
    float ptmHigh = m_PtmHigh->getValueAtTime(p_Args.time);
    float ptmHighSt = m_PtmHighSt->getValueAtTime(p_Args.time);
    
    // Brilliance Parameters
    bool brlEnable = m_BrlEnable->getValueAtTime(p_Args.time);
    float brlR = m_BrlR->getValueAtTime(p_Args.time);
    float brlG = m_BrlG->getValueAtTime(p_Args.time);
    float brlB = m_BrlB->getValueAtTime(p_Args.time);
    float brlC = m_BrlC->getValueAtTime(p_Args.time);
    float brlM = m_BrlM->getValueAtTime(p_Args.time);
    float brlY = m_BrlY->getValueAtTime(p_Args.time);
    float brlRng = m_BrlRng->getValueAtTime(p_Args.time);
    
    // Hueshift RGB Parameters
    bool hsRgbEnable = m_HsRgbEnable->getValueAtTime(p_Args.time);
    float hsR = m_HsR->getValueAtTime(p_Args.time);
    float hsG = m_HsG->getValueAtTime(p_Args.time);
    float hsB = m_HsB->getValueAtTime(p_Args.time);
    float hsRgbRng = m_HsRgbRng->getValueAtTime(p_Args.time);
    
    // Hueshift CMY Parameters
    bool hsCmyEnable = m_HsCmyEnable->getValueAtTime(p_Args.time);
    float hsC = m_HsC->getValueAtTime(p_Args.time);
    float hsM = m_HsM->getValueAtTime(p_Args.time);
    float hsY = m_HsY->getValueAtTime(p_Args.time);
    
    // Hue Contrast Parameters
    bool hcEnable = m_HcEnable->getValueAtTime(p_Args.time);
    float hcR = m_HcR->getValueAtTime(p_Args.time);
    
    // NEW PARAMETERS
    // Filmic Parameters
    bool filmicMode = m_FilmicMode->getValueAtTime(p_Args.time);
    float filmicDynamicRange = m_FilmicDynamicRange->getValueAtTime(p_Args.time);
    int filmicProjectorSim;
    m_FilmicProjectorSim->getValueAtTime(p_Args.time, filmicProjectorSim);
    
    // NEW FILMIC PARAMETERS
    float filmicSourceStops = m_FilmicSourceStops->getValueAtTime(p_Args.time);
    float filmicTargetStops = m_FilmicTargetStops->getValueAtTime(p_Args.time);
    float filmicStrength = m_FilmicStrength->getValueAtTime(p_Args.time);
    
    // Advanced Hue Contrast Parameters
    bool advHueContrast = m_AdvHueContrast->getValueAtTime(p_Args.time);
    
    // Tonescale Map Parameters
    bool tonescaleMap = m_TonescaleMap->getValueAtTime(p_Args.time);
    
    // Diagnostics Parameters
    bool diagnosticsMode = m_DiagnosticsMode->getValueAtTime(p_Args.time);
    bool rgbChipsMode = m_RgbChipsMode->getValueAtTime(p_Args.time);
    
    // Beta Features Parameters
    bool betaFeaturesEnable = m_BetaFeaturesEnable->getValueAtTime(p_Args.time);
    
    // Display Parameters
    int displayGamut;
    m_DisplayGamut->getValueAtTime(p_Args.time, displayGamut);
    int eotf;
    m_Eotf->getValueAtTime(p_Args.time, eotf);
    int lookPreset;
    m_LookPreset->getValueAtTime(p_Args.time, lookPreset);
    ////////////////////////////////////////////////////////////////////////////////
    // PROCESSOR SETUP - CONFIGURE THE IMAGE PROCESSOR
    ////////////////////////////////////////////////////////////////////////////////
    // BOILERPLATE: Keep processor setup
    p_Processor.setDstImg(dst.get());
    p_Processor.setSrcImg(src.get());
    p_Processor.setGPURenderArgs(p_Args);
    p_Processor.setRenderWindow(p_Args.renderWindow);

    // Pass all OpenDRT parameters to processor
    p_Processor.setOpenDRTParams(
        // Input Settings
        inGamut, inOetf,
        // Tonescale Parameters
        tnLp, tnGb, ptHdr,
        // Clamp Parameters
        clamp, tnLg, tnCon, tnSh, tnToe, tnOff,
        // High Contrast Parameters
        tnHconEnable, tnHcon, tnHconPv, tnHconSt,
        // Low Contrast Parameters
        tnLconEnable, tnLcon, tnLconW, tnLconPc,
        // Creative White Parameters
        cwp, cwpRng,
        // Render Space Parameters
        rsSa, rsRw, rsBw,
        // Purity Compress Parameters
        ptR, ptG, ptB, ptRngLow, ptRngHigh,
        // Purity Enable/Disable
        ptlEnable, ptmEnable,
        // Mid Purity Parameters
        ptmLow, ptmLowSt, ptmHigh, ptmHighSt,
        // Brilliance Parameters
        brlEnable, brlR, brlG, brlB, brlC, brlM, brlY, brlRng,
        // Hueshift RGB Parameters
        hsRgbEnable, hsR, hsG, hsB, hsRgbRng,
        // Hueshift CMY Parameters
        hsCmyEnable, hsC, hsM, hsY,
        // Hue Contrast Parameters
        hcEnable, hcR,
        // NEW PARAMETERS - Filmic Mode and Advanced Controls
        filmicMode, filmicDynamicRange, filmicProjectorSim,
        filmicSourceStops, filmicTargetStops, filmicStrength,
        advHueContrast, tonescaleMap, diagnosticsMode, rgbChipsMode, betaFeaturesEnable,
        // Display Parameters
        displayGamut, eotf,
        // Look Preset
        lookPreset
    );
    ////////////////////////////////////////////////////////////////////////////////
    // EXECUTE PROCESSING
    ////////////////////////////////////////////////////////////////////////////////
    
    // Start the actual image processing (calls multiThreadProcessImages)
    // BOILERPLATE: Keep process call
    p_Processor.process();
}

////////////////////////////////////////////////////////////////////////////////
// BOILERPLATE: Keep overlay interaction (remove if not needed)  Good for watermarks on onscreen debugging

// class TemplateOFXInteract : public OFX::OverlayInteract  // <-- RENAME: Change class name
// {
// public:
// TemplateOFXInteract(OfxInteractHandle p_Handle, OFX::ImageEffect* /*p_Effect*/)
//        : OFX::OverlayInteract(p_Handle)
//    { }

//    virtual bool draw(const OFX::DrawArgs& p_Args)
//    {
        // Step 18: ADD YOUR OVERLAY DRAWING HERE (or remove if not needed)
//        OfxDrawContextHandle contextHandle = OFX::Private::gDrawSuite ? p_Args.context : nullptr;
//        if (!contextHandle) return false;

//        const OfxRGBAColourF color = { 1.0f, 0.3f, 0.3f, 1.0f };
//        OFX::Private::gDrawSuite->setColour(contextHandle, &color);
        
        // Example drawing - replace or remove
//        const OfxPointD points[] = { {200.0f, 200.0f} };
//        OFX::Private::gDrawSuite->drawText(contextHandle, "Your Plugin", points, kOfxDrawTextAlignmentLeft);

//        return true;
//    }
// };

// class YourOverlayDescriptor : public OFX::DefaultEffectOverlayDescriptor<YourOverlayDescriptor, TemplateOFXInteract>
//{
// };

////////////////////////////////////////////////////////////////////////////////
// PLUGIN FACTORY CLASS - CREATES PLUGIN INSTANCES AND DEFINES UI
////////////////////////////////////////////////////////////////////////////////
// BOILERPLATE: Plugin factory

using namespace OFX;
////////////////////////////////////////////////////////////////////////////////
// FACTORY CONSTRUCTOR
////////////////////////////////////////////////////////////////////////////////
OpenDRTFactory::OpenDRTFactory()
    : OFX::PluginFactoryHelper<OpenDRTFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

////////////////////////////////////////////////////////////////////////////////
// PLUGIN DESCRIPTION - TELLS HOST ABOUT PLUGIN CAPABILITIES
////////////////////////////////////////////////////////////////////////////////
// BOILERPLATE: Keep basic plugin description
void OpenDRTFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
{
    p_Desc.setLabels(kPluginName, kPluginName, kPluginName);
    p_Desc.setPluginGrouping(kPluginGrouping);
    p_Desc.setPluginDescription(kPluginDescription);

    p_Desc.addSupportedContext(eContextFilter);
    p_Desc.addSupportedContext(eContextGeneral);
    p_Desc.addSupportedBitDepth(eBitDepthFloat);

    p_Desc.setSingleInstance(false);
    p_Desc.setHostFrameThreading(false);
    p_Desc.setSupportsMultiResolution(kSupportsMultiResolution);
    p_Desc.setSupportsTiles(kSupportsTiles);
    p_Desc.setTemporalClipAccess(false);
    p_Desc.setRenderTwiceAlways(false);
    p_Desc.setSupportsMultipleClipPARs(kSupportsMultipleClipPARs);

    // GPU support flags
    p_Desc.setSupportsOpenCLRender(true);
#ifndef __APPLE__
    p_Desc.setSupportsCudaRender(true);
    p_Desc.setSupportsCudaStream(true);
#endif
#ifdef __APPLE__
    p_Desc.setSupportsMetalRender(true);
#endif

    p_Desc.setNoSpatialAwareness(true);
    // p_Desc.setOverlayInteractDescriptor(new YourOverlayDescriptor());  // Remove if no overlay
}
////////////////////////////////////////////////////////////////////////////////
// PARAMETER DEFINITION HELPER FUNCTIONS
// These create different types of UI controls with appropriate defaults
////////////////////////////////////////////////////////////////////////////////
// BOILERPLATE: Helper function for creating parameters
static DoubleParamDescriptor* defineDoubleParam(OFX::ImageEffectDescriptor& p_Desc, 
                                               const std::string& p_Name, 
                                               const std::string& p_Label,
                                               const std::string& p_Hint, 
                                               GroupParamDescriptor* p_Parent = nullptr,
                                               double defaultValue = 1.0,
                                               double minValue = 0.0,
                                               double maxValue = 10.0,
                                               double increment = 0.1)
{
    DoubleParamDescriptor* param = p_Desc.defineDoubleParam(p_Name);
    param->setLabels(p_Label, p_Label, p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    param->setDefault(defaultValue);
    param->setRange(minValue, maxValue);
    param->setIncrement(increment);
    param->setDisplayRange(minValue, maxValue);
    param->setDoubleType(eDoubleTypePlain);

    if (p_Parent) {
        param->setParent(*p_Parent);
    }
    return param;
}

////////////////////////////////////////////////////////////////////////////////
// UI CREATION - DEFINES ALL THE PARAMETER CONTROLS AND LAYOUT
////////////////////////////////////////////////////////////////////////////////
void OpenDRTFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum)
{
    // BOILERPLATE: Keep clip setup
    ClipDescriptor* srcClip = p_Desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    srcClip->addSupportedComponent(ePixelComponentRGBA);
    srcClip->setTemporalClipAccess(false);
    srcClip->setSupportsTiles(kSupportsTiles);
    srcClip->setIsMask(false);

    ClipDescriptor* dstClip = p_Desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(ePixelComponentRGBA);
    dstClip->addSupportedComponent(ePixelComponentAlpha);
    dstClip->setSupportsTiles(kSupportsTiles);

    ////////////////////////////////////////////////////////////////////////////////
    // PAGE AND GROUP CREATION - UI ORGANIZATION
    ////////////////////////////////////////////////////////////////////////////////// 
    // BOILERPLATE: Create page and groups

    PageParamDescriptor* page = p_Desc.definePageParam("Controls");
    
    // OpenDRT Groups
    GroupParamDescriptor* inputGroup = p_Desc.defineGroupParam("InputGroup");
    inputGroup->setHint(" Defaults: DWG/I > Rec.709/gamma 2.4");
    inputGroup->setLabels("Input & Output Settings", "Input & Output Settings", "Input & Output Settings");
    inputGroup->setOpen(false);

     GroupParamDescriptor* presetGroup = p_Desc.defineGroupParam("PresetGroup");
    presetGroup->setHint("Preset configurations for quick setup");
    presetGroup->setLabels("Automatic Presets", "Automatic Presets", "Automatic Presets");
      
    GroupParamDescriptor* stickshiftGroup = p_Desc.defineGroupParam("StickshiftGroup");
    stickshiftGroup->setHint("Enable Stickshift Modes");
    stickshiftGroup->setLabels("Stickshift Mode", "Stickshift Mode", "Stickshift Mode");
    stickshiftGroup->setOpen(false);

    GroupParamDescriptor* tonescaleGroup = p_Desc.defineGroupParam("TonescaleGroup");
    tonescaleGroup->setHint("Primary tonescale parameters");
    tonescaleGroup->setLabels("Tonescale", "Tonescale", "Tonescale");
    tonescaleGroup->setOpen(false);

    GroupParamDescriptor* contrastGroup = p_Desc.defineGroupParam("ContrastGroup");
    contrastGroup->setHint("Basic contrast adjustment parameters");
    contrastGroup->setLabels("Basic Contrast", "Basic Contrast", "Basic Contrast");
    contrastGroup->setOpen(false);
    
    GroupParamDescriptor* colorGroup = p_Desc.defineGroupParam("ColorGroup");
    colorGroup->setHint("Color rendering and creative adjustments");
    colorGroup->setLabels("Creative White Point", "Creative White Point", "Creative White Point");
    colorGroup->setOpen(false);
    
    // High Contrast Group
    GroupParamDescriptor* highContrastGroup = p_Desc.defineGroupParam("HighContrastGroup");
    highContrastGroup->setHint("High frequency contrast adjustments");
    highContrastGroup->setLabels("High Contrast", "High Contrast", "High Contrast");
    highContrastGroup->setOpen(false);
    // Low Contrast Group
    GroupParamDescriptor* lowContrastGroup = p_Desc.defineGroupParam("LowContrastGroup");
    lowContrastGroup->setHint("Low frequency contrast adjustments");
    lowContrastGroup->setLabels("Low Contrast", "Low Contrast", "Low Contrast");
    lowContrastGroup->setOpen(false);
    // Purity Low Group
    GroupParamDescriptor* purityLowGroup = p_Desc.defineGroupParam("PurityLowGroup");
    purityLowGroup->setHint("Purity compression low controls");
    purityLowGroup->setLabels("Purity Compress Low", "Purity Compress Low", "Purity Compress Low");
    purityLowGroup->setOpen(false);
    // Mid Purity Group
    GroupParamDescriptor* midPurityGroup = p_Desc.defineGroupParam("MidPurityGroup");
    midPurityGroup->setHint("Mid purity adjustment controls");
    midPurityGroup->setLabels("Mid Purity", "Mid Purity", "Mid Purity");
    midPurityGroup->setOpen(false);
    // Brilliance Group
    GroupParamDescriptor* brillianceGroup = p_Desc.defineGroupParam("BrillianceGroup");
    brillianceGroup->setHint("Brilliance adjustment controls");
    brillianceGroup->setLabels("Brilliance", "Brilliance", "Brilliance");
    brillianceGroup->setOpen(false);
    // Hueshift RGB Group
    GroupParamDescriptor* hueshiftRgbGroup = p_Desc.defineGroupParam("HueshiftRgbGroup");
    hueshiftRgbGroup->setHint("RGB hue shifting controls");
    hueshiftRgbGroup->setLabels("Hueshift RGB", "Hueshift RGB", "Hueshift RGB");
    hueshiftRgbGroup->setOpen(false);
    // Hueshift CMY Group
    GroupParamDescriptor* hueshiftCmyGroup = p_Desc.defineGroupParam("HueshiftCmyGroup");
    hueshiftCmyGroup->setHint("CMY hue shifting controls");
    hueshiftCmyGroup->setLabels("Hueshift CMY", "Hueshift CMY", "Hueshift CMY");
    hueshiftCmyGroup->setOpen(false);
    // Hue Contrast Group
    GroupParamDescriptor* hueContrastGroup = p_Desc.defineGroupParam("HueContrastGroup");
    hueContrastGroup->setHint("Hue contrast adjustment controls");
    hueContrastGroup->setLabels("Hue Contrast", "Hue Contrast", "Hue Contrast");
    hueContrastGroup->setOpen(false);
    
    // NEW GROUPS
    // Diagnostics Group
    GroupParamDescriptor* diagnosticsGroup = p_Desc.defineGroupParam("DiagnosticsGroup");
    diagnosticsGroup->setHint("Diagnostic and visualization tools");
    diagnosticsGroup->setLabels("Diagnostics", "Diagnostics", "Diagnostics");
    diagnosticsGroup->setOpen(false); // Closed Always displayed
    
     // ADD: Buy Me a Coffee button at the end
    PushButtonParamDescriptor* coffeeButton = p_Desc.definePushButtonParam("buymeacoffee");
    coffeeButton->setLabel("Learn How to Use Open DRT");
    coffeeButton->setHint("Tutorials, documentation, and support for Open DRT");
    page->addChild(*coffeeButton);
    
    // Beta Features Group
    GroupParamDescriptor* betaFeaturesGroup = p_Desc.defineGroupParam("BetaFeaturesGroup");
    betaFeaturesGroup->setHint("Experimental or work-in-progress parameters");
    betaFeaturesGroup->setLabels("Beta Features", "Beta Features", "Beta Features");
    betaFeaturesGroup->setOpen(false); // Always visible but parameters controlled individually
    
    // Advanced Hue Contrast Group
    GroupParamDescriptor* advHueContrastGroup = p_Desc.defineGroupParam("AdvHueContrastGroup");
    advHueContrastGroup->setHint("Advanced hue contrast adjustment controls");
    advHueContrastGroup->setLabels("Advanced Hue Contrast", "Advanced Hue Contrast", "Advanced Hue Contrast");
    advHueContrastGroup->setOpen(false); // Hidden based on Advanced Hue Contrast Enable
    // Filmic Dynamic Range Group
    GroupParamDescriptor* filmicDynamicRangeGroup = p_Desc.defineGroupParam("FilmicDynamicRangeGroup");
    filmicDynamicRangeGroup->setHint("Filmic dynamic range controls");
    filmicDynamicRangeGroup->setLabels("Filmic Dynamic Range Beta", "Filmic Dynamic Range Beta", "Filmic Dynamic Range Beta");
    filmicDynamicRangeGroup->setOpen(false); // Hidden based on Filmic Mode
    
    // Filmic Projector Sim Group
    GroupParamDescriptor* filmicProjectorSimGroup = p_Desc.defineGroupParam("FilmicProjectorSimGroup");
    filmicProjectorSimGroup->setHint("Filmic projector simulation controls");
    filmicProjectorSimGroup->setLabels("Filmic Projector Sim Beta", "Filmic Projector Sim Beta", "Filmic Projector Sim Beta");
    filmicProjectorSimGroup->setOpen(false); // Hidden Based on Filmic Mode

    // Add groups to page
    page->addChild(*inputGroup);
    page->addChild(*presetGroup);
    page->addChild(*stickshiftGroup);
    page->addChild(*tonescaleGroup);
    page->addChild(*contrastGroup);
    page->addChild(*highContrastGroup);
    page->addChild(*lowContrastGroup);
    page->addChild(*colorGroup);
    page->addChild(*purityLowGroup);
    page->addChild(*midPurityGroup);
    page->addChild(*brillianceGroup);
    page->addChild(*hueshiftRgbGroup);
    page->addChild(*hueshiftCmyGroup);
    page->addChild(*hueContrastGroup);
        page->addChild(*betaFeaturesGroup);
    
    // ADD NEW GROUPS TO PAGE
    page->addChild(*diagnosticsGroup);
    page->addChild(*filmicDynamicRangeGroup);
    page->addChild(*filmicProjectorSimGroup);
    page->addChild(*advHueContrastGroup);

    ////////////////////////////////////////////////////////////////////////////////
    // INPUT SETTINGS
    ////////////////////////////////////////////////////////////////////////////////
    
    // Input Gamut
    ChoiceParamDescriptor* inGamutParam = p_Desc.defineChoiceParam("in_gamut");
    inGamutParam->setLabel("Input Gamut");
    inGamutParam->setHint("Input color gamut/primaries");
    inGamutParam->appendOption("XYZ");
    inGamutParam->appendOption("ACES 2065-1");
    inGamutParam->appendOption("ACEScg");
    inGamutParam->appendOption("P3D65");
    inGamutParam->appendOption("Rec.2020");
    inGamutParam->appendOption("Rec.709");
    inGamutParam->appendOption("Arri Wide Gamut 3");
    inGamutParam->appendOption("Arri Wide Gamut 4");
    inGamutParam->appendOption("Red Wide Gamut RGB");
    inGamutParam->appendOption("Sony SGamut3");
    inGamutParam->appendOption("Sony SGamut3Cine");
    inGamutParam->appendOption("Panasonic V-Gamut");
    inGamutParam->appendOption("Blackmagic Wide Gamut");
    inGamutParam->appendOption("Filmlight E-Gamut");
    inGamutParam->appendOption("Filmlight E-Gamut2");
    inGamutParam->appendOption("DaVinci Wide Gamut");
    inGamutParam->setDefault(15); // DaVinci Wide Gamut
    inGamutParam->setAnimates(true);
    inGamutParam->setParent(*inputGroup);
    page->addChild(*inGamutParam);

    // Input Transfer Function
    ChoiceParamDescriptor* inOetfParam = p_Desc.defineChoiceParam("in_oetf");
    inOetfParam->setLabel("Input Transfer Function");
    inOetfParam->setHint("Input transfer function (OETF)");
    inOetfParam->appendOption("Linear");
    inOetfParam->appendOption("Davinci Intermediate");
    inOetfParam->appendOption("Filmlight T-Log");
    inOetfParam->appendOption("ACEScct");
    inOetfParam->appendOption("Arri LogC3");
    inOetfParam->appendOption("Arri LogC4");
    inOetfParam->appendOption("RedLog3G10");
    inOetfParam->appendOption("Panasonic V-Log");
    inOetfParam->appendOption("Sony S-Log3");
    inOetfParam->setDefault(1); // Davinci Intermediate
    inOetfParam->setAnimates(true);
    inOetfParam->setParent(*inputGroup);
    page->addChild(*inOetfParam);

    ////////////////////////////////////////////////////////////////////////////////
    // PRESET PARAMETERS
    ////////////////////////////////////////////////////////////////////////////////
    
 // Look Preset
ChoiceParamDescriptor* lookPresetParam = p_Desc.defineChoiceParam("look_preset");
lookPresetParam->setLabel("Look Preset");
lookPresetParam->setHint("Select a film look preset to apply characteristic color grading");
lookPresetParam->appendOption("Default");      // Index 0 -> LOOK_PRESETS[0]
lookPresetParam->appendOption("Colorful");     // Index 1 -> LOOK_PRESETS[1]
lookPresetParam->appendOption("Umbra");        // Index 2 -> LOOK_PRESETS[2]
lookPresetParam->appendOption("Base");         // Index 3 -> LOOK_PRESETS[3]
lookPresetParam->setDefault(0); // Default
lookPresetParam->setAnimates(true);
lookPresetParam->setParent(*presetGroup);
page->addChild(*lookPresetParam);

// Tonescale Preset
ChoiceParamDescriptor* tonescalePresetParam = p_Desc.defineChoiceParam("tonescale_preset");
tonescalePresetParam->setLabel("Tonescale Preset");
tonescalePresetParam->setHint("Select a tonescale preset for quick adjustment");
tonescalePresetParam->appendOption("Use Look Preset");         // Index 0
tonescalePresetParam->appendOption("High-Contrast");           // Index 1 -> TONESCALE_PRESETS[0]
tonescalePresetParam->appendOption("Low-Contrast");            // Index 2 -> TONESCALE_PRESETS[1]
tonescalePresetParam->appendOption("ACES-1.x");                // Index 3 -> TONESCALE_PRESETS[2]
tonescalePresetParam->appendOption("ACES-2.0");                // Index 4 -> TONESCALE_PRESETS[3]
tonescalePresetParam->appendOption("Marvelous Tonescape");     // Index 5 -> TONESCALE_PRESETS[4]
tonescalePresetParam->appendOption("Arriba Tonecall");         // Index 6 -> TONESCALE_PRESETS[5]
tonescalePresetParam->appendOption("DaGrinchi Tonegroan");     // Index 7 -> TONESCALE_PRESETS[6]
tonescalePresetParam->appendOption("Aery Tonescale");          // Index 8 -> TONESCALE_PRESETS[7]
tonescalePresetParam->appendOption("Umbra Tonescale");         // Index 9 -> TONESCALE_PRESETS[8]
tonescalePresetParam->setDefault(0); // Use Look Preset
tonescalePresetParam->setAnimates(true);
tonescalePresetParam->setParent(*presetGroup);
page->addChild(*tonescalePresetParam);

    // Lock parameter
    OFX::BooleanParamDescriptor* lockStickshiftParam = p_Desc.defineBooleanParam("lockStickshift");
    lockStickshiftParam->setDefault(false);
    lockStickshiftParam->setHint("When enabled, prevents presets from overriding manual stickshift adjustments");
    lockStickshiftParam->setLabels("Lock Manual Adjustments", "Lock Manual Adjustments", "Lock Manual Adjustments");
    lockStickshiftParam->setParent(*presetGroup);
    page->addChild(*lockStickshiftParam);

    ////////////////////////////////////////////////////////////////////////////////
    // TONESCALE PARAMETERS
    ////////////////////////////////////////////////////////////////////////////////
    
    DoubleParamDescriptor* param;
    
    param = defineDoubleParam(p_Desc, "tn_Lp", "Display Peak Luminance", "Peak luminance of the display in nits", 
                             tonescaleGroup, 100.0, 100.0, 1000.0, 1.0);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "tn_gb", "HDR Grey Boost", "Boost grey levels for HDR displays", 
                             tonescaleGroup, 0.13, 0.0, 1.0, 0.01);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "pt_hdr", "HDR Purity", "Purity adjustment for HDR", 
                             tonescaleGroup, 0.5, 0.0, 1.0, 0.01);
    page->addChild(*param);

   
    ////////////////////////////////////////////////////////////////////////////////
    // CLAMP/TONESCALE ADJUSTMENT PARAMETERS
    ////////////////////////////////////////////////////////////////////////////////
    
    // Clamp enable
    BooleanParamDescriptor* clampParam = p_Desc.defineBooleanParam("_clamp");
    clampParam->setDefault(true);
    clampParam->setHint("Enable tone curve clamping");
    clampParam->setLabels("Clamp", "Clamp", "Clamp");
    clampParam->setParent(*contrastGroup);
    page->addChild(*clampParam);

    param = defineDoubleParam(p_Desc, "_tn_Lg", "Grey Luminance", "Grey point luminance", 
                             contrastGroup, 11.1, 4.0, 25.0, 0.1);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_tn_con", "Contrast", "Overall contrast adjustment", 
                             contrastGroup, 1.4, 1.0, 2.0, 0.01);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_tn_sh", "Shoulder Clip", "Highlight shoulder clipping", 
                             contrastGroup, 0.5, 0.0, 1.0, 0.01);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_tn_toe", "Toe", "Shadow toe adjustment", 
                             contrastGroup, 0.003, 0.0, 0.1, 0.001);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_tn_off", "Offset", "Black point offset", 
                             contrastGroup, 0.005, 0.0, 0.02, 0.001);
    page->addChild(*param);

    // High Contrast Enable (stays in stickshift group)
    BooleanParamDescriptor* hconEnableParam = p_Desc.defineBooleanParam("_tn_hcon_enable");
    hconEnableParam->setDefault(false);
    hconEnableParam->setHint("Enable high contrast adjustments");
    hconEnableParam->setLabels("Enable Contrast High", "Enable Contrast High", "Enable Contrast High");
    hconEnableParam->setParent(*stickshiftGroup);
    page->addChild(*hconEnableParam);

    BooleanParamDescriptor* lockHconParam = p_Desc.defineBooleanParam("_lock_hcon");
    lockHconParam->setDefault(false);
    lockHconParam->setHint("Values Persistent from preset changes, but can still be adjusted manually");
    lockHconParam->setLabels("🔒 Persistent Values", "🔒 Persistent Values", "🔒 Persistent Values");
    lockHconParam->setParent(*highContrastGroup);
    page->addChild(*lockHconParam);

    // High Contrast Parameters (move to highContrastGroup)
    param = defineDoubleParam(p_Desc, "_tn_hcon", "Contrast High", "High frequency contrast", 
                             highContrastGroup, 0.0, -1.0, 1.0, 0.01);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_tn_hcon_pv", "Contrast High Pivot", "Pivot point for high contrast", 
                             highContrastGroup, 1.0, 0.0, 4.0, 0.01);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_tn_hcon_st", "Contrast High Strength", "Strength of high contrast effect", 
                             highContrastGroup, 4.0, 0.0, 4.0, 0.01);
    page->addChild(*param);

    // Low Contrast Enable (stays in stickshift group)
    BooleanParamDescriptor* lconEnableParam = p_Desc.defineBooleanParam("_tn_lcon_enable");
    lconEnableParam->setDefault(false);
    lconEnableParam->setHint("Enable low contrast adjustments");
    lconEnableParam->setLabels("Enable Contrast Low", "Enable Contrast Low", "Enable Contrast Low");
    lconEnableParam->setParent(*stickshiftGroup);
    page->addChild(*lconEnableParam);

    BooleanParamDescriptor* lockLconParam = p_Desc.defineBooleanParam("_lock_lcon");
    lockLconParam->setDefault(false);
    lockLconParam->setHint("Values Persistent from preset changes, but can still be adjusted manually");
    lockLconParam->setLabels("🔒 Persistent Values", "🔒 Persistent Values", "🔒 Persistent Values");
    lockLconParam->setParent(*lowContrastGroup);
    page->addChild(*lockLconParam);

    // Low Contrast Parameters (move to lowContrastGroup)
    param = defineDoubleParam(p_Desc, "_tn_lcon", "Contrast Low", "Low frequency contrast", 
                             lowContrastGroup, 1.0, 0.0, 3.0, 0.001);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_tn_lcon_w", "Contrast Low Width", "Width of low contrast effect", 
                             lowContrastGroup, 0.5, 0.0, 2.0, 0.001);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_tn_lcon_pc", "Contrast Low Per-Channel", "Per-channel low contrast", 
                             lowContrastGroup, 1.0, 0.0, 1.0, 0.001);
    page->addChild(*param);

    ////////////////////////////////////////////////////////////////////////////////
    // COLOR CONTROLS
    ////////////////////////////////////////////////////////////////////////////////
    
    // Creative White
    ChoiceParamDescriptor* cwpParam = p_Desc.defineChoiceParam("_cwp");
    cwpParam->setLabel("Creative White");
    cwpParam->setHint("Creative white point selection");
    cwpParam->appendOption("D65");
    cwpParam->appendOption("D60");
    cwpParam->appendOption("D55");
    cwpParam->appendOption("D50");
    cwpParam->appendOption("Use Look Preset");
    cwpParam->setDefault(4); // D65
    cwpParam->setAnimates(true);
    cwpParam->setParent(*colorGroup);
    page->addChild(*cwpParam);

    param = defineDoubleParam(p_Desc, "_cwp_rng", "Creative White Range", "Range of creative white effect", 
                             colorGroup, 0.5, 0.0, 1.0, 0.001);
    page->addChild(*param);

    // Render Space Parameters
    param = defineDoubleParam(p_Desc, "_rs_sa", "Render Space Strength", "Strength of render space adjustment", 
                             
                             colorGroup, 0.35, 0.0, 0.6, 0.001);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_rs_rw", "Render Space Red Weight", "Red channel weight", 
                             colorGroup, 0.25, 0.0, 0.8, 0.001);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_rs_bw", "Render Space Blue Weight", "Blue channel weight", 
                             colorGroup, 0.55, 0.0, 0.8, 0.001);
    page->addChild(*param);

    // Purity Compress Parameters
    BooleanParamDescriptor* ptlEnableParam = p_Desc.defineBooleanParam("_ptl_enable");
    ptlEnableParam->setDefault(false); // CHANGED: was true
    ptlEnableParam->setHint("Enable purity compression low");
    ptlEnableParam->setLabels("Enable Purity Low", "Enable Purity Low", "Enable Purity Low");
    ptlEnableParam->setParent(*stickshiftGroup);
    page->addChild(*ptlEnableParam);

    BooleanParamDescriptor* lockPtlParam = p_Desc.defineBooleanParam("_lock_ptl");
    lockPtlParam->setDefault(false);
    lockPtlParam->setHint("Values Persistent from preset changes, but can still be adjusted manually");
    lockPtlParam->setLabels("🔒 Persistent Values", "🔒 Persistent Values", "🔒 Persistent Values");
    lockPtlParam->setParent(*purityLowGroup);
    page->addChild(*lockPtlParam);
   
// Purity Compress Parameters (move to purityLowGroup)
    param = defineDoubleParam(p_Desc, "_pt_r", "Purity Compress R", "Red purity compression", 
                             purityLowGroup, 0.5, 0.0, 4.0, 0.001);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_pt_g", "Purity Compress G", "Green purity compression", 
                             purityLowGroup, 2.0, 0.0, 4.0, 0.001);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_pt_b", "Purity Compress B", "Blue purity compression", 
                             purityLowGroup, 2.0, 0.0, 4.0, 0.001);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_pt_rng_low", "Purity Range Low", "Low range for purity compression", 
                             purityLowGroup, 0.2, 0.1, 0.6, 0.001);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_pt_rng_high", "Purity Range High", "High range for purity compression", 
                             purityLowGroup, 0.8, 0.25, 2.0, 0.001);
    page->addChild(*param);

    // Mid Purity Enable + Lock
    BooleanParamDescriptor* ptmEnableParam = p_Desc.defineBooleanParam("_ptm_enable");
    ptmEnableParam->setDefault(false); // CHANGED: was true
    ptmEnableParam->setHint("Enable mid purity adjustments");
    ptmEnableParam->setLabels("Enable Mid Purity", "Enable Mid Purity", "Enable Mid Purity");
    ptmEnableParam->setParent(*stickshiftGroup);
    page->addChild(*ptmEnableParam);

    BooleanParamDescriptor* lockPtmParam = p_Desc.defineBooleanParam("_lock_ptm");
    lockPtmParam->setDefault(false);
    lockPtmParam->setHint("Values Persistent from preset changes, but can still be adjusted manually");
    lockPtmParam->setLabels("🔒 Persistent Values", "🔒 Persistent Values", "🔒 Persistent Values");
    lockPtmParam->setParent(*midPurityGroup);
    page->addChild(*lockPtmParam);

    // Mid Purity Parameters (move to midPurityGroup)
    param = defineDoubleParam(p_Desc, "_ptm_low", "Mid Purity Low", "Mid purity low adjustment", 
                             midPurityGroup, 0.2, 0.0, 1.0, 0.001);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_ptm_low_st", "Mid Purity Low Strength", "Strength of mid purity low", 
                             midPurityGroup, 0.5, 0.1, 1.0, 0.001);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_ptm_high", "Mid Purity High", "Mid purity high adjustment", 
                             midPurityGroup, -0.8, -0.9, 0.0, 0.001);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_ptm_high_st", "Mid Purity High Strength", "Strength of mid purity high", 
                             midPurityGroup, 0.3, 0.2, 1.0, 0.001);
    page->addChild(*param);

    // Brilliance Enable + Lock
    BooleanParamDescriptor* brlEnableParam = p_Desc.defineBooleanParam("_brl_enable");
    brlEnableParam->setDefault(false);
    brlEnableParam->setHint("Enable brilliance adjustments");
    brlEnableParam->setLabels("Enable Brilliance", "Enable Brilliance", "Enable Brilliance");
    brlEnableParam->setParent(*stickshiftGroup);
    page->addChild(*brlEnableParam);

    BooleanParamDescriptor* lockBrlParam = p_Desc.defineBooleanParam("_lock_brl");
    lockBrlParam->setDefault(false);
    lockBrlParam->setHint("Values Persistent from preset changes, but can still be adjusted manually");
    lockBrlParam->setLabels("🔒 Persistent Values", "🔒 Persistent Values", "🔒 Persistent Values");
    lockBrlParam->setParent(*brillianceGroup);
    page->addChild(*lockBrlParam);

    // Brilliance Parameters (move to brillianceGroup)
    param = defineDoubleParam(p_Desc, "_brl_r", "Brilliance R", "Red brilliance adjustment", 
                             brillianceGroup, -0.5, -1.0, 1.0, 0.001);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_brl_g", "Brilliance G", "Green brilliance adjustment", 
                             brillianceGroup, -0.4, -1.0, 1.0, 0.001);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_brl_b", "Brilliance B", "Blue brilliance adjustment", 
                             brillianceGroup, -0.2, -1.0, 1.0, 0.001);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_brl_c", "Brilliance C", "Cyan brilliance adjustment", 
                             brillianceGroup, 0.0, -1.0, 1.0, 0.001);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_brl_m", "Brilliance M", "Magenta brilliance adjustment", 
                             brillianceGroup, 0.0, -1.0, 1.0, 0.001);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_brl_y", "Brilliance Y", "Yellow brilliance adjustment", 
                             brillianceGroup, 0.0, -1.0, 1.0, 0.001);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_brl_rng", "Brilliance Range", "Range of brilliance effect", 
                             brillianceGroup, 0.66, 0.0, 2.0, 0.001);
    page->addChild(*param);

    // Hueshift RGB Enable + Lock
    BooleanParamDescriptor* hsRgbEnableParam = p_Desc.defineBooleanParam("_hs_rgb_enable");
    hsRgbEnableParam->setDefault(false);
    hsRgbEnableParam->setHint("Enable RGB hue shifting");
    hsRgbEnableParam->setLabels("Enable Hueshift RGB", "Enable Hueshift RGB", "Enable Hueshift RGB");
    hsRgbEnableParam->setParent(*stickshiftGroup);
    page->addChild(*hsRgbEnableParam);

    BooleanParamDescriptor* lockHsRgbParam = p_Desc.defineBooleanParam("_lock_hs_rgb");
    lockHsRgbParam->setDefault(false);
    lockHsRgbParam->setHint("Values Persistent from preset changes, but can still be adjusted manually");
    lockHsRgbParam->setLabels("🔒 Persistent Values", "🔒 Persistent Values", "🔒 Persistent Values");
    lockHsRgbParam->setParent(*hueshiftRgbGroup);
    page->addChild(*lockHsRgbParam);

    // Hueshift RGB Parameters (move to hueshiftRgbGroup)
    param = defineDoubleParam(p_Desc, "_hs_r", "Hueshift R", "Red hue shifting", 
                             hueshiftRgbGroup, 0.350, -1.0, 1.0, 0.001);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_hs_g", "Hueshift G", "Green hue shifting", 
                             hueshiftRgbGroup, 0.25, -1.0, 1.0, 0.001);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_hs_b", "Hueshift B", "Blue hue shifting", 
                             hueshiftRgbGroup, 0.500, -1.0, 1.0, 0.001);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_hs_rgb_rng", "Hueshift RGB Range", "Range of RGB hue shifting effect", 
                             hueshiftRgbGroup, 0.6, 0.0, 2.0, 0.001);
    page->addChild(*param);

    // Hueshift CMY Enable + Lock
    BooleanParamDescriptor* hsCmyEnableParam = p_Desc.defineBooleanParam("_hs_cmy_enable");
    hsCmyEnableParam->setDefault(false);
    hsCmyEnableParam->setHint("Enable CMY hue shifting");
    hsCmyEnableParam->setLabels("Enable Hueshift CMY", "Enable Hueshift CMY", "Enable Hueshift CMY");
    hsCmyEnableParam->setParent(*stickshiftGroup);
    page->addChild(*hsCmyEnableParam);

    BooleanParamDescriptor* lockHsCmyParam = p_Desc.defineBooleanParam("_lock_hs_cmy");
    lockHsCmyParam->setDefault(false);
    lockHsCmyParam->setHint("Values Persistent from preset changes, but can still be adjusted manually");
    lockHsCmyParam->setLabels("🔒 Persistent Values", "🔒 Persistent Values", "🔒 Persistent Values");
    lockHsCmyParam->setParent(*hueshiftCmyGroup);
    page->addChild(*lockHsCmyParam);

    // Hueshift CMY Parameters (move to hueshiftCmyGroup)
    param = defineDoubleParam(p_Desc, "_hs_c", "Hueshift C", "Cyan hue shifting", 
                             hueshiftCmyGroup, 0.200, -1.0, 1.0, 0.001);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_hs_m", "Hueshift M", "Magenta hue shifting", 
                             hueshiftCmyGroup, 0.200, -1.0, 1.0, 0.001);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_hs_y", "Hueshift Y", "Yellow hue shifting", 
                             hueshiftCmyGroup, 0.200, -1.0, 1.0, 0.001);
    page->addChild(*param);

    // Hue Contrast Enable + Lock
    BooleanParamDescriptor* hcEnableParam = p_Desc.defineBooleanParam("_hc_enable");
    hcEnableParam->setDefault(false);
    hcEnableParam->setHint("Enable hue contrast adjustments");
    hcEnableParam->setLabels("Enable Hue Contrast", "Enable Hue Contrast", "Enable Hue Contrast");
    hcEnableParam->setParent(*stickshiftGroup);
    page->addChild(*hcEnableParam);
    
    // NEW PARAMETERS IN STICKSHIFT GROUP
    // Filmic Mode Enable
    BooleanParamDescriptor* filmicModeParam = p_Desc.defineBooleanParam("_filmic_mode");
    filmicModeParam->setDefault(false); // Deactivated by default
    filmicModeParam->setHint("Enable filmic mode rendering");
    filmicModeParam->setLabels("Enable Filmic Mode Beta", "Enable Filmic Mode Beta", "Enable Filmic Mode Beta");
    filmicModeParam->setParent(*betaFeaturesGroup);
    page->addChild(*filmicModeParam);
    
    // Advanced Hue Contrast Enable
    BooleanParamDescriptor* advHueContrastParam = p_Desc.defineBooleanParam("_adv_hue_contrast");
    advHueContrastParam->setDefault(false); // Deactivated by default
    advHueContrastParam->setHint("Enable advanced hue contrast adjustments");
    advHueContrastParam->setLabels("Enable Adv Hue Contrast", "Enable Adv Hue Contrast", "Enable Adv Hue Contrast");
    advHueContrastParam->setParent(*betaFeaturesGroup);
    page->addChild(*advHueContrastParam);
    
    // Beta Features Enable
    BooleanParamDescriptor* betaFeaturesEnableParam = p_Desc.defineBooleanParam("_beta_features_enable");
    betaFeaturesEnableParam->setDefault(false); // Deactivated by default
    betaFeaturesEnableParam->setHint("Turns on experimental and hidden controls");
    betaFeaturesEnableParam->setLabels("Enable Beta Features", "Enable Beta Features", "Enable Beta Features");
    betaFeaturesEnableParam->setParent(*betaFeaturesGroup);
    page->addChild(*betaFeaturesEnableParam);

    BooleanParamDescriptor* lockHcParam = p_Desc.defineBooleanParam("_lock_hc");
    lockHcParam->setDefault(false);
    lockHcParam->setHint("Values Persistent from preset changes, but can still be adjusted manually");
    lockHcParam->setLabels("🔒 Persistent Values", "🔒 Persistent Values", "🔒 Persistent Values");
    lockHcParam->setParent(*hueContrastGroup);
    page->addChild(*lockHcParam);

    // Hue Contrast Parameters (move to hueContrastGroup)
    param = defineDoubleParam(p_Desc, "_hc_r", "Hue Contrast R", "Red hue contrast adjustment", 
                             hueContrastGroup, 0.600, -1.0, 1.0, 0.001);
    page->addChild(*param);

    ////////////////////////////////////////////////////////////////////////////////
    // ADVANCED HUE CONTRAST GROUP PARAMETERS
    ////////////////////////////////////////////////////////////////////////////////
    
    // Advanced Hue Contrast Parameters (in Advanced Hue Contrast group)
    param = defineDoubleParam(p_Desc, "_adv_hc_g", "Advanced Hue Contrast G", "Green advanced hue contrast adjustment", 
                             advHueContrastGroup, 0.0, -1.0, 1.0, 0.001);
    page->addChild(*param);
    
    param = defineDoubleParam(p_Desc, "_adv_hc_b", "Advanced Hue Contrast B", "Blue advanced hue contrast adjustment", 
                             advHueContrastGroup, 0.0, -1.0, 1.0, 0.001);
    page->addChild(*param);
    
    param = defineDoubleParam(p_Desc, "_adv_hc_c", "Advanced Hue Contrast C", "Cyan advanced hue contrast adjustment", 
                             advHueContrastGroup, 0.0, -1.0, 1.0, 0.001);
    page->addChild(*param);
    
    param = defineDoubleParam(p_Desc, "_adv_hc_m", "Advanced Hue Contrast M", "Magenta advanced hue contrast adjustment", 
                             advHueContrastGroup, 0.0, -1.0, 1.0, 0.001);
    page->addChild(*param);
    
    param = defineDoubleParam(p_Desc, "_adv_hc_y", "Advanced Hue Contrast Y", "Yellow advanced hue contrast adjustment", 
                             advHueContrastGroup, 0.0, -1.0, 1.0, 0.001);
    page->addChild(*param);

    ////////////////////////////////////////////////////////////////////////////////
    // NEW PARAMETER GROUPS AND PARAMETERS
    ////////////////////////////////////////////////////////////////////////////////
    
    // DIAGNOSTICS GROUP PARAMETERS
    // Tonescale Map Parameter (in Diagnostics group)
    BooleanParamDescriptor* tonescaleMapParam = p_Desc.defineBooleanParam("_tonescale_map");
    tonescaleMapParam->setDefault(false); // Deactivated by default
    tonescaleMapParam->setHint("Enable tonescale mapping visualization");
    tonescaleMapParam->setLabels("Tonescale Curve", "Tonescale Curve", "Tonescale Curve");
    tonescaleMapParam->setParent(*diagnosticsGroup);
    page->addChild(*tonescaleMapParam);
    
    // Diagnostics Mode Parameter (in Diagnostics group)
    BooleanParamDescriptor* diagnosticsModeParam = p_Desc.defineBooleanParam("_diagnostics_mode");
    diagnosticsModeParam->setDefault(false); // Deactivated by default
    diagnosticsModeParam->setHint("Enable Grey Scale Ramp");
    diagnosticsModeParam->setLabels("Grey Scale Ramp", "Grey Scale Ramp", "Grey Scale Ramp");
    diagnosticsModeParam->setParent(*diagnosticsGroup);
    page->addChild(*diagnosticsModeParam);
    
    // RGB Chips Parameter (in Diagnostics group)
    BooleanParamDescriptor* rgbChipsModeParam = p_Desc.defineBooleanParam("_rgbchips");
    rgbChipsModeParam->setDefault(false); // Deactivated by default
    rgbChipsModeParam->setHint("Enable RGB Chips");
    rgbChipsModeParam->setLabels("RGB Chips", "RGB Chips", "RGB Chips");
    rgbChipsModeParam->setParent(*diagnosticsGroup);
    page->addChild(*rgbChipsModeParam);
    
    // FILMIC DYNAMIC RANGE GROUP PARAMETERS
    // Filmic Dynamic Range Parameter (in Filmic Dynamic Range group)
   
    
    // Original Camera Range Parameter
    param = defineDoubleParam(p_Desc, "_filmic_source_stops", "Original Camera Range", "Number of stops captured by the original camera or scene", 
                             filmicDynamicRangeGroup, 14.0, 1.0, 20.0, 1.0);
    page->addChild(*param);
    
    // Target Film Range Parameter
    param = defineDoubleParam(p_Desc, "_filmic_target_stops", "Target Film Range", "Stops the final image should be compressed into to mimic film", 
                             filmicDynamicRangeGroup, 10.0, 1.0, 20.0, 1.0);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "_filmic_dynamic_range", "Roll Off Characteristics", "Controls highlight rolloff characteristics. Lower = harder rolloff (reversal film), Higher = gentler rolloff (negative film)", 
                             filmicDynamicRangeGroup, 5.0, 1.0, 10.0, 0.1);
    page->addChild(*param);
    
    // Filmic Strength Parameter
    param = defineDoubleParam(p_Desc, "_filmic_strength", "Strength", "Blends between the tonescaled image and the compressed result. 1.0 = full compression, 0.0 = no change", 
                             filmicDynamicRangeGroup, 0.0, 0.0, 1.0, 0.01);
    page->addChild(*param);
    
    // FILMIC PROJECTOR SIM GROUP PARAMETERS
    // Filmic Projector Simulation Parameter (in Filmic Projector Sim group)
    ChoiceParamDescriptor* filmicProjectorSimParam = p_Desc.defineChoiceParam("_filmic_projector_sim");
    filmicProjectorSimParam->setLabel("Projector Simulation");
    filmicProjectorSimParam->setHint("Filmic projector simulation type");
    filmicProjectorSimParam->appendOption("None");
    filmicProjectorSimParam->appendOption("Xenon");
    filmicProjectorSimParam->appendOption("Tungsten");
    filmicProjectorSimParam->appendOption("LED");
    filmicProjectorSimParam->setDefault(0); // None
    filmicProjectorSimParam->setAnimates(true);
    filmicProjectorSimParam->setParent(*filmicProjectorSimGroup);
    page->addChild(*filmicProjectorSimParam);

    ////////////////////////////////////////////////////////////////////////////////
    // DISPLAY SETTINGS
    ////////////////////////////////////////////////////////////////////////////////
    
    // Display Gamut
    ChoiceParamDescriptor* displayGamutParam = p_Desc.defineChoiceParam("_display_gamut");
    displayGamutParam->setLabel("Display Gamut");
    displayGamutParam->setHint("Output display gamut");
    displayGamutParam->appendOption("Rec.709");
    displayGamutParam->appendOption("P3-D65");
    displayGamutParam->appendOption("Rec.2020 (P3 Limited)");
    displayGamutParam->setDefault(0); // Rec.709
    displayGamutParam->setAnimates(true);
    displayGamutParam->setParent(*inputGroup);
    page->addChild(*displayGamutParam);

    // Display EOTF
    ChoiceParamDescriptor* eotfParam = p_Desc.defineChoiceParam("_eotf");
    eotfParam->setLabel("Display EOTF");
    eotfParam->setHint("Output display transfer function (EOTF)");
    eotfParam->appendOption("Linear");
    eotfParam->appendOption("2.2 Power sRGB Display");
    eotfParam->appendOption("2.4 Power Rec.1886");
    eotfParam->appendOption("2.6 Power DCI");
    eotfParam->appendOption("ST 2084 PQ");
    eotfParam->appendOption("HLG");
    eotfParam->setDefault(2); // 2.4 Power Rec.1886
    eotfParam->setAnimates(true);
    eotfParam->setParent(*inputGroup);
    page->addChild(*eotfParam);
// Add this at the very end of describeInContext, just before the closing brace:


    
    // ADD: Beta Features Group after Learn More button


}
////////////////////////////////////////////////////////////////////////////////
// PLUGIN INSTANCE  Creation Boilerplate
////////////////////////////////////////////////////////////////////////////////
ImageEffect* OpenDRTFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new OpenDRT(p_Handle);
}
////////////////////////////////////////////////////////////////////////////////
// PLUGIN REGISTRATION - TELLS OFX SYSTEM ABOUT OUR PLUGIN
////////////////////////////////////////////////////////////////////////////////
// BOILERPLATE: Keep plugin registration
void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    // Initialize matrix manager on plugin load
    static bool matricesInitialized = false;
    if (!matricesInitialized) {
        std::string configPath = "./ColorMatrices.json"; // Default path
        
        // Try different possible paths
        const char* possiblePaths[] = {
            "./ColorMatrices.json",
            "../ColorMatrices.json", 
            "../../ColorMatrices.json",
            "./Open DRT/ColorMatrices.json",
            "../Open DRT/ColorMatrices.json"
        };
        
        bool loaded = false;
        for (const char* path : possiblePaths) {
            if (MatrixManager::loadMatrices(path)) {
                printf("OpenDRT: Loaded color matrices from %s\n", path);
                loaded = true;
                break;
            }
        }
        
        if (!loaded) {
            printf("OpenDRT Warning: Could not load color matrices, using fallback matrices\n");
        }
        
        matricesInitialized = true;
    }
    
    static OpenDRTFactory OpenDRT;
    p_FactoryArray.push_back(&OpenDRT);
}
