////////////////////////////////////////////////////////////////////////////////
// INCLUDES AND DEFINITIONS
////////////////////////////////////////////////////////////////////////////////
#include "ContrastSatVolume.h"

#include <stdio.h>
#include <algorithm>  // For std::max, std::min
#include <cmath>      // For pow function

#include "ofxsImageEffect.h"
#include "ofxsInteract.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"
#include "ofxsLog.h"
#include "ofxDrawSuite.h"
#include "ofxsSupportPrivate.h"

// Plugin metadata - defines what the plugin is called and how it appears in hosts
#define kPluginName "Contrast Sat Volume"
#define kPluginGrouping "create@Dec18Studios.com"
#define kPluginDescription "Apply seperate RGB and CYM Contrast Adjustments to each channel"
#define kPluginIdentifier "com.OpenFXSample.ContrastSatVolume"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 0

// Plugin capabilities - what features this plugin supports
#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

#define kParamBlendMode "blendMode"
#define kParamBlendModeLabel "Blend Mode"

enum BlendModeEnum {
    BLEND_MODE_PRIMARY = 0,    // Primary (RGB)
    BLEND_MODE_SECONDARY = 1,  // Secondary (CYM)
    BLEND_MODE_MIX = 2         // Blend Control
};

////////////////////////////////////////////////////////////////////////////////
// IMAGE PROCESSOR CLASS - WHERE THE ACTUAL IMAGE PROCESSING HAPPENS
////////////////////////////////////////////////////////////////////////////////

class ImageScaler : public OFX::ImageProcessor
{
public:
    explicit ImageScaler(OFX::ImageEffect& p_Instance);

    // GPU processing methods for different platforms
    virtual void processImagesCUDA();    // NVIDIA CUDA processing
    virtual void processImagesOpenCL();  // OpenCL processing (cross-platform)
    virtual void processImagesMetal();   // Apple Metal processing

    // CPU processing method - this is where our main algorithm runs
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

    // Methods to set up the processor with images and parameters
    void setSrcImg(OFX::Image* p_SrcImg);
    void setContrastParams(float p_GammaR, float p_GammaG, float p_GammaB, 
                          float p_GammaC, float p_GammaM, float p_GammaY,
                          float p_MidgreyR, float p_MidgreyG, float p_MidgreyB,
                          float p_MidgreyC, float p_MidgreyM, float p_MidgreyY,
                          float p_TiltCR, float p_TiltGM, float p_TiltBY);

private:
    OFX::Image* _srcImg;  // Pointer to source image
    
    // RGB gamma parameters - control contrast for Red, Green, Blue channels
    float _gammaR, _gammaG, _gammaB;
    
    // CYM gamma parameters - control contrast for Cyan, Magenta, Yellow channels
    float _gammaC, _gammaM, _gammaY;
    
    // RGB midgrey parameters - the point that doesn't change during gamma adjustment
    float _midgreyR, _midgreyG, _midgreyB;
    
    // CYM midgrey parameters
    float _midgreyC, _midgreyM, _midgreyY;
    
    // Tilt parameters - control blend between RGB and CYM adjustments (0=CYM, 1=RGB)
    float _tiltCR, _tiltGM, _tiltBY;
    
    // Helper function for gamma contrast with locked midgrey point
    float applyGammaContrast(float input, float gamma, float midgrey);

    // Helper function to apply gamma contrast to entire RGB pixel (like float3 in DCTL)
    void applyGammaContrastFloat3(float& r, float& g, float& b, float gamma, float midgrey);
};

////////////////////////////////////////////////////////////////////////////////
// IMAGE PROCESSOR IMPLEMENTATION
////////////////////////////////////////////////////////////////////////////////

ImageScaler::ImageScaler(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance)
{
}

////////////////////////////////////////////////////////////////////////////////
// GPU PROCESSING METHODS (CUDA/OPENCL/METAL)
// These call external kernel functions for GPU acceleration
////////////////////////////////////////////////////////////////////////////////

#ifndef __APPLE__
extern void RunCudaKernel(void* p_Stream, int p_Width, int p_Height, float* p_Gain, const float* p_Input, float* p_Output);
extern void RunComplexCudaKernel(void* p_Stream, int p_Width, int p_Height,
                                 float* p_RgbGammas, float* p_CymGammas,
                                 float* p_RgbMidgreys, float* p_CymMidgreys,
                                 float* p_Tilts,
                                 const float* p_Input, float* p_Output);
#endif

#ifdef __APPLE__
extern void RunMetalKernel(void* p_CmdQ, int p_Width, int p_Height, float* p_Gain, const float* p_Input, float* p_Output);
extern void RunComplexMetalKernel(void* p_CmdQ, int p_Width, int p_Height, 
                                  float* p_RgbGammas, float* p_CymGammas, 
                                  float* p_RgbMidgreys, float* p_CymMidgreys, 
                                  float* p_Tilts,
                                  const float* p_Input, float* p_Output);
#endif

extern void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float* p_Gain, const float* p_Input, float* p_Output);
extern void RunComplexOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, 
                                   float* p_RgbGammas, float* p_CymGammas,
                                   float* p_RgbMidgreys, float* p_CymMidgreys,
                                   float* p_Tilts,
                                   const float* p_Input, float* p_Output);
////////////////////////////////////////////////////////////////////////////////
// CORE ALGORITHM FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

// Helper function to apply gamma contrast with locked midgrey point
// This implements the mathematical formula: output = (input/midgrey)^gamma * midgrey
float ImageScaler::applyGammaContrast(float input, float gamma, float midgrey)
{
    // Clamp input to avoid issues with negative values or extreme ranges
    float safeInput = std::max(0.001f, std::min(0.999f, input));
    float safeMidgrey = std::max(0.001f, std::min(0.999f, midgrey));
    
    // Apply gamma curve with locked midgrey point
    // Formula: output = (input/midgrey)^gamma * midgrey
    // This ensures the midgrey value stays exactly the same after processing
    float normalized = safeInput / safeMidgrey;
    float adjusted = pow(normalized, gamma) * safeMidgrey;
    
    // Clamp output to valid range [0,1]
    return std::max(0.0f, std::min(1.0f, adjusted));
}

// Helper function to apply gamma contrast to entire RGB pixel (like float3 in DCTL)
// This applies the same gamma adjustment to all three RGB channels simultaneously
void ImageScaler::applyGammaContrastFloat3(float& r, float& g, float& b, float gamma, float midgrey)
{
    // Clamp inputs to safe range
    float safeR = std::max(0.001f, std::min(0.999f, r));
    float safeG = std::max(0.001f, std::min(0.999f, g));
    float safeB = std::max(0.001f, std::min(0.999f, b));
    float safeMidgrey = std::max(0.001f, std::min(0.999f, midgrey));
    
    // Apply gamma curve to all three channels simultaneously
    float recipMidgrey = 1.0f / safeMidgrey;
    
    // Apply same gamma to all RGB channels
    r = pow(safeR * recipMidgrey, gamma) * safeMidgrey;
    g = pow(safeG * recipMidgrey, gamma) * safeMidgrey;
    b = pow(safeB * recipMidgrey, gamma) * safeMidgrey;
    
    // Clamp outputs
    r = std::max(0.0f, std::min(1.0f, r));
    g = std::max(0.0f, std::min(1.0f, g));
    b = std::max(0.0f, std::min(1.0f, b));
}

////////////////////////////////////////////////////////////////////////////////
// MAIN CPU PROCESSING FUNCTION
// This is where the actual pixel-by-pixel image processing happens
////////////////////////////////////////////////////////////////////////////////

void ImageScaler::multiThreadProcessImages(OfxRectI p_ProcWindow)
{
    for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y)
    {
        if (_effect.abort()) break;
        float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));

        for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x)
        {
            float* srcPix = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(x, y) : 0);

            if (srcPix)
            {
                float r = srcPix[0];
                float g = srcPix[1];
                float b = srcPix[2];
                float a = srcPix[3];
                
                // Apply contrast to ENTIRE RGB using different gammas (like your DCTL)
                
                // RGB adjustments - apply each gamma to all RGB channels
                float rAdj_R = r, rAdj_G = g, rAdj_B = b;
                applyGammaContrastFloat3(rAdj_R, rAdj_G, rAdj_B, _gammaR, _midgreyR);
                
                float gAdj_R = r, gAdj_G = g, gAdj_B = b;
                applyGammaContrastFloat3(gAdj_R, gAdj_G, gAdj_B, _gammaG, _midgreyG);
                
                float bAdj_R = r, bAdj_G = g, bAdj_B = b;
                applyGammaContrastFloat3(bAdj_R, bAdj_G, bAdj_B, _gammaB, _midgreyB);
                
                // CYM adjustments - apply each gamma to all RGB channels  
                float cAdj_R = r, cAdj_G = g, cAdj_B = b;
                applyGammaContrastFloat3(cAdj_R, cAdj_G, cAdj_B, _gammaC, _midgreyC);
                
                float mAdj_R = r, mAdj_G = g, mAdj_B = b;
                applyGammaContrastFloat3(mAdj_R, mAdj_G, mAdj_B, _gammaM, _midgreyM);
                
                float yAdj_R = r, yAdj_G = g, yAdj_B = b;
                applyGammaContrastFloat3(yAdj_R, yAdj_G, yAdj_B, _gammaY, _midgreyY);
                
                // Mix results - each output channel gets specific adjustment blend
                float finalR = cAdj_R + (rAdj_R - cAdj_R) * _tiltCR;  // Red output uses Red gamma vs Cyan gamma
                float finalG = mAdj_G + (gAdj_G - mAdj_G) * _tiltGM;  // Green output uses Green gamma vs Magenta gamma
                float finalB = yAdj_B + (bAdj_B - yAdj_B) * _tiltBY;  // Blue output uses Blue gamma vs Yellow gamma
                
                dstPix[0] = finalR;
                dstPix[1] = finalG;
                dstPix[2] = finalB;
                dstPix[3] = a;
            }
            else
            {
                for (int c = 0; c < 4; ++c)
                {
                    dstPix[c] = 0;
                }
            }

            dstPix += 4;
        }
    }
}

void ImageScaler::processImagesCUDA()
{
#ifndef __APPLE__
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    // Pass all parameter arrays
    float rgbGammas[3] = {_gammaR, _gammaG, _gammaB};
    float cymGammas[3] = {_gammaC, _gammaM, _gammaY};
    float rgbMidgreys[3] = {_midgreyR, _midgreyG, _midgreyB};
    float cymMidgreys[3] = {_midgreyC, _midgreyM, _midgreyY};
    float tilts[3] = {_tiltCR, _tiltGM, _tiltBY};
    
    RunComplexCudaKernel(_pCudaStream, width, height,
                        rgbGammas, cymGammas, rgbMidgreys, cymMidgreys, tilts,
                        input, output);
#endif
}

void ImageScaler::processImagesMetal()
{
#ifdef __APPLE__
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    // Pass all parameter arrays
    float rgbGammas[3] = {_gammaR, _gammaG, _gammaB};
    float cymGammas[3] = {_gammaC, _gammaM, _gammaY};
    float rgbMidgreys[3] = {_midgreyR, _midgreyG, _midgreyB};
    float cymMidgreys[3] = {_midgreyC, _midgreyM, _midgreyY};
    float tilts[3] = {_tiltCR, _tiltGM, _tiltBY};
    
    RunComplexMetalKernel(_pMetalCmdQ, width, height,
                         rgbGammas, cymGammas, rgbMidgreys, cymMidgreys, tilts,
                         input, output);
#endif
}

void ImageScaler::processImagesOpenCL()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    // Pass all parameter arrays
    float rgbGammas[3] = {_gammaR, _gammaG, _gammaB};
    float cymGammas[3] = {_gammaC, _gammaM, _gammaY};
    float rgbMidgreys[3] = {_midgreyR, _midgreyG, _midgreyB};
    float cymMidgreys[3] = {_midgreyC, _midgreyM, _midgreyY};
    float tilts[3] = {_tiltCR, _tiltGM, _tiltBY};
    
    RunComplexOpenCLKernel(_pOpenCLCmdQ, width, height,
                          rgbGammas, cymGammas, rgbMidgreys, cymMidgreys, tilts,
                          input, output);
}

////////////////////////////////////////////////////////////////////////////////
// PARAMETER SETTER FUNCTION
// This receives all the UI parameter values and stores them for processing
////////////////////////////////////////////////////////////////////////////////

void ImageScaler::setContrastParams(float p_GammaR, float p_GammaG, float p_GammaB, 
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

////////////////////////////////////////////////////////////////////////////////
// SOURCE IMAGE SETTER FUNCTION
// This sets the input image for processing
////////////////////////////////////////////////////////////////////////////////

void ImageScaler::setSrcImg(OFX::Image* p_SrcImg)
{
    _srcImg = p_SrcImg;
}

////////////////////////////////////////////////////////////////////////////////
// MAIN PLUGIN CLASS - HANDLES OFX PLUGIN INTERFACE
////////////////////////////////////////////////////////////////////////////////

/** @brief The plugin that does our work */
class ContrastSatVolume : public OFX::ImageEffect
{
public:
    explicit ContrastSatVolume(OfxImageEffectHandle p_Handle);

    /* Override the render - called when the host wants to process frames */
    virtual void render(const OFX::RenderArguments& p_Args);

    /* Override is identity - tells host if processing can be skipped */
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);

    /* Override changedParam - called when user changes a parameter */
    virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);

    /* Override changed clip - called when input clip changes */
    virtual void changedClip(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ClipName);

    /* Set the enabledness of the component scale params depending on the type of input image and the state of the scaleComponents param */
    void setEnabledness();

    /* Set up and run a processor */
    void setupAndProcess(ImageScaler &p_ImageScaler, const OFX::RenderArguments& p_Args);

private:
    ////////////////////////////////////////////////////////////////////////////////
    // CLIP POINTERS - CONNECTIONS TO INPUT/OUTPUT IMAGES
    ////////////////////////////////////////////////////////////////////////////////
    OFX::Clip* m_DstClip;  // Output image clip
    OFX::Clip* m_SrcClip;  // Input image clip

    ////////////////////////////////////////////////////////////////////////////////
    // PARAMETER POINTERS - CONNECTIONS TO UI CONTROLS
    ////////////////////////////////////////////////////////////////////////////////
    
    // RGB gamma parameters - control contrast for each color channel
    OFX::DoubleParam* m_ScaleR;  // Red gamma
    OFX::DoubleParam* m_ScaleG;  // Green gamma
    OFX::DoubleParam* m_ScaleB;  // Blue gamma
    OFX::DoubleParam* m_ScaleA;  // Alpha scale (not used in gamma processing)

    // CYM gamma parameters - control contrast for complementary colors
    OFX::DoubleParam* m_ScaleC;  // Cyan gamma
    OFX::DoubleParam* m_ScaleM;  // Magenta gamma
    OFX::DoubleParam* m_ScaleY;  // Yellow gamma

    // RGB midgrey parameters - the "pivot point" for gamma adjustments
    OFX::DoubleParam* m_ScaleMidGreyR;  // Red midgrey point
    OFX::DoubleParam* m_ScaleMidGreyG;  // Green midgrey point
    OFX::DoubleParam* m_ScaleMidGreyB;  // Blue midgrey point
    
    // CYM midgrey parameters
    OFX::DoubleParam* m_ScaleMidGreyC;  // Cyan midgrey point
    OFX::DoubleParam* m_ScaleMidGreyM;  // Magenta midgrey point
    OFX::DoubleParam* m_ScaleMidGreyY;  // Yellow midgrey point

    // Tilt parameters - control blend between RGB and CYM adjustments
    OFX::DoubleParam* m_TiltCR;  // Red-Cyan tilt
    OFX::DoubleParam* m_TiltGM;  // Green-Magenta tilt
    OFX::DoubleParam* m_TiltBY;  // Blue-Yellow tilt

    // Boolean control parameters
    OFX::BooleanParam* m_GangRGB;                 // Gang RGB controls together
    OFX::BooleanParam* m_GangCYM;                 // Gang CYM controls together

    // Blend Mode dropdown
    OFX::ChoiceParam* m_BlendMode;               // Blend mode (Primary, Secondary, Blend Control)
    
    // Buy Me a Coffee button
    OFX::PushButtonParam* m_CoffeeButton;        // Coffee support button
};

////////////////////////////////////////////////////////////////////////////////
// PLUGIN CONSTRUCTOR - CONNECTS TO ALL THE UI PARAMETERS
////////////////////////////////////////////////////////////////////////////////

ContrastSatVolume::ContrastSatVolume(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    // Connect to input/output clips
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

    // Connect to all the parameter controls created in describeInContext()
    // These parameter names must match exactly what was defined in describeInContext()
    
    // RGB gamma parameters
    m_ScaleR = fetchDoubleParam("scaleR");
    m_ScaleG = fetchDoubleParam("scaleG");
    m_ScaleB = fetchDoubleParam("scaleB");
    m_ScaleA = fetchDoubleParam("scaleA");

    // CYM gamma parameters
    m_ScaleC = fetchDoubleParam("scaleC");
    m_ScaleM = fetchDoubleParam("scaleM");
    m_ScaleY = fetchDoubleParam("scaleY");

    // RGB midgrey parameters
    m_ScaleMidGreyR = fetchDoubleParam("scaleMidGreyR");
    m_ScaleMidGreyG = fetchDoubleParam("scaleMidGreyG");
    m_ScaleMidGreyB = fetchDoubleParam("scaleMidGreyB");
    
    // CYM midgrey parameters
    m_ScaleMidGreyC = fetchDoubleParam("scaleMidGreyC");
    m_ScaleMidGreyM = fetchDoubleParam("scaleMidGreyM");
    m_ScaleMidGreyY = fetchDoubleParam("scaleMidGreyY");

    // Tilt parameters
    m_TiltCR = fetchDoubleParam("tiltCR");
    m_TiltGM = fetchDoubleParam("tiltGM");
    m_TiltBY = fetchDoubleParam("tiltBY");

    // Boolean parameters
    m_GangRGB = fetchBooleanParam("GangRGB");
    m_GangCYM = fetchBooleanParam("GangCYM");

    // Blend mode parameter
    m_BlendMode = fetchChoiceParam("blendMode");
    
    // Buy Me a Coffee button
    m_CoffeeButton = fetchPushButtonParam("buymeacoffee");

    // Set the initial enabledness of our UI controls
    setEnabledness();
}

////////////////////////////////////////////////////////////////////////////////
// RENDER FUNCTION - MAIN ENTRY POINT FOR PROCESSING
////////////////////////////////////////////////////////////////////////////////

void ContrastSatVolume::render(const OFX::RenderArguments& p_Args)
{
    // Only process 32-bit float RGBA images
    if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
    {
        // Create our image processor and run it
        ImageScaler imageScaler(*this);
        setupAndProcess(imageScaler, p_Args);
    }
    else
    {
        // Unsupported image format - throw error
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

////////////////////////////////////////////////////////////////////////////////
// IDENTITY CHECK - OPTIMIZATION TO SKIP PROCESSING WHEN NO CHANGE WOULD OCCUR
////////////////////////////////////////////////////////////////////////////////

bool ContrastSatVolume::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
    // Get blend mode - FIXED: ChoiceParam needs two arguments
    int blendMode;
    m_BlendMode->getValueAtTime(p_Args.time, blendMode);
    
    // Check parameters based on blend mode
    if (blendMode == BLEND_MODE_PRIMARY) {
        // Only check RGB parameters
        double gammaR = m_ScaleR->getValueAtTime(p_Args.time);
        double gammaG = m_ScaleG->getValueAtTime(p_Args.time);
        double gammaB = m_ScaleB->getValueAtTime(p_Args.time);
        if ((gammaR != 1.0) || (gammaG != 1.0) || (gammaB != 1.0)) {
            return false;
        }
    } else if (blendMode == BLEND_MODE_SECONDARY) {
        // Only check CYM parameters
        double gammaC = m_ScaleC->getValueAtTime(p_Args.time);
        double gammaM = m_ScaleM->getValueAtTime(p_Args.time);
        double gammaY = m_ScaleY->getValueAtTime(p_Args.time);
        if ((gammaC != 1.0) || (gammaM != 1.0) || (gammaY != 1.0)) {
            return false;
        }
    } else {
        // Check all parameters for blend mode
        double gammaR = m_ScaleR->getValueAtTime(p_Args.time);
        double gammaG = m_ScaleG->getValueAtTime(p_Args.time);
        double gammaB = m_ScaleB->getValueAtTime(p_Args.time);
        
        double gammaC = m_ScaleC->getValueAtTime(p_Args.time);
        double gammaM = m_ScaleM->getValueAtTime(p_Args.time);
        double gammaY = m_ScaleY->getValueAtTime(p_Args.time);
        
        double tiltCR = m_TiltCR->getValueAtTime(p_Args.time);
        double tiltGM = m_TiltGM->getValueAtTime(p_Args.time);
        double tiltBY = m_TiltBY->getValueAtTime(p_Args.time);
        
        // If any parameter is not at neutral, we need to process
        if ((gammaR != 1.0) || (gammaG != 1.0) || (gammaB != 1.0) ||
            (gammaC != 1.0) || (gammaM != 1.0) || (gammaY != 1.0) ||
            (tiltCR != 0.5) || (tiltGM != 0.5) || (tiltBY != 0.5))
        {
            return false; // Process the image
        }
    }

    // All parameters are at neutral values, skip processing
    p_IdentityClip = m_SrcClip;
    p_IdentityTime = p_Args.time;
    return true;
}

////////////////////////////////////////////////////////////////////////////////
// PARAMETER CHANGE HANDLER - RESPONDS TO UI CHANGES
////////////////////////////////////////////////////////////////////////////////

void ContrastSatVolume::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
    // When gang control parameters change, update which controls are enabled
    if (p_ParamName == "GangRGB" || p_ParamName == "GangCYM")
    {
        setEnabledness();
    }
    else if (p_ParamName == "buymeacoffee")
    {
        // Open Buy Me a Coffee link
        std::string url = "https://www.dec18studios.com/coffee"; // Replace with Greg's actual URL
        
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

void ContrastSatVolume::changedClip(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ClipName)
{
    // When the input clip changes, update which controls are enabled
    if (p_ClipName == kOfxImageEffectSimpleSourceClipName)
    {
        setEnabledness();
    }
}

////////////////////////////////////////////////////////////////////////////////
// UI CONTROL ENABLEMENT - MANAGES WHICH CONTROLS ARE AVAILABLE
////////////////////////////////////////////////////////////////////////////////

void ContrastSatVolume::setEnabledness()
{
    // Check if we have RGBA input
    const bool hasRGBAInput = (m_SrcClip->getPixelComponents() == OFX::ePixelComponentRGBA);
    
    // Get gang control states
    const bool gangRGB = m_GangRGB->getValue();
    const bool gangCYM = m_GangCYM->getValue();

    // Enable base controls if we have RGBA input
    m_ScaleA->setEnabled(hasRGBAInput);
    
    // Enable RGB gamma parameters
    m_ScaleR->setEnabled(hasRGBAInput);
    m_ScaleG->setEnabled(hasRGBAInput && !gangRGB);  // Disable G when ganged to R
    m_ScaleB->setEnabled(hasRGBAInput && !gangRGB);  // Disable B when ganged to R
    
    // Enable CYM gamma parameters
    m_ScaleC->setEnabled(hasRGBAInput);
    m_ScaleM->setEnabled(hasRGBAInput && !gangCYM);  // Disable M when ganged to C
    m_ScaleY->setEnabled(hasRGBAInput && !gangCYM);  // Disable Y when ganged to C
    
    // Enable RGB midgrey parameters
    m_ScaleMidGreyR->setEnabled(hasRGBAInput);
    m_ScaleMidGreyG->setEnabled(hasRGBAInput && !gangRGB);  // Disable G when ganged to R
    m_ScaleMidGreyB->setEnabled(hasRGBAInput && !gangRGB);  // Disable B when ganged to R
    
    // Enable CYM midgrey parameters
    m_ScaleMidGreyC->setEnabled(hasRGBAInput);
    m_ScaleMidGreyM->setEnabled(hasRGBAInput && !gangCYM);  // Disable M when ganged to C
    m_ScaleMidGreyY->setEnabled(hasRGBAInput && !gangCYM);  // Disable Y when ganged to C
    
    // Enable tilt parameters
    m_TiltCR->setEnabled(hasRGBAInput);
    m_TiltGM->setEnabled(hasRGBAInput);
    m_TiltBY->setEnabled(hasRGBAInput);
}

////////////////////////////////////////////////////////////////////////////////
// SETUP AND PROCESS - COORDINATES THE ENTIRE PROCESSING PIPELINE
////////////////////////////////////////////////////////////////////////////////

void ContrastSatVolume::setupAndProcess(ImageScaler& p_ImageScaler, const OFX::RenderArguments& p_Args)
{
    ////////////////////////////////////////////////////////////////////////////////
    // IMAGE SETUP - GET INPUT AND OUTPUT IMAGES
    ////////////////////////////////////////////////////////////////////////////////
    
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
    
    // Get base parameter values
    float gammaR = m_ScaleR->getValueAtTime(p_Args.time);
    float gammaG = m_ScaleG->getValueAtTime(p_Args.time);
    float gammaB = m_ScaleB->getValueAtTime(p_Args.time);
    
    float gammaC = m_ScaleC->getValueAtTime(p_Args.time);
    float gammaM = m_ScaleM->getValueAtTime(p_Args.time);
    float gammaY = m_ScaleY->getValueAtTime(p_Args.time);
    
    float midgreyR = m_ScaleMidGreyR->getValueAtTime(p_Args.time);
    float midgreyG = m_ScaleMidGreyG->getValueAtTime(p_Args.time);
    float midgreyB = m_ScaleMidGreyB->getValueAtTime(p_Args.time);
    
    float midgreyC = m_ScaleMidGreyC->getValueAtTime(p_Args.time);
    float midgreyM = m_ScaleMidGreyM->getValueAtTime(p_Args.time);
    float midgreyY = m_ScaleMidGreyY->getValueAtTime(p_Args.time);
    
    float tiltCR = m_TiltCR->getValueAtTime(p_Args.time);
    float tiltGM = m_TiltGM->getValueAtTime(p_Args.time);
    float tiltBY = m_TiltBY->getValueAtTime(p_Args.time);

    // Get UI control parameters - FIXED: ChoiceParam needs two arguments
    int blendMode;
    m_BlendMode->getValueAtTime(p_Args.time, blendMode);
    
    bool gangRGB = m_GangRGB->getValueAtTime(p_Args.time);
    bool gangCYM = m_GangCYM->getValueAtTime(p_Args.time);

    // Apply gang logic (like your DCTL)
    if (gangRGB) {
        gammaG = gammaR;  // Gang Green to Red
        gammaB = gammaR;  // Gang Blue to Red
        midgreyG = midgreyR;  // Gang midgreys too
        midgreyB = midgreyR;
    }
    
    if (gangCYM) {
        gammaM = gammaC;  // Gang Magenta to Cyan
        gammaY = gammaC;  // Gang Yellow to Cyan
        midgreyM = midgreyC;  // Gang midgreys too
        midgreyY = midgreyC;
    }

    // Apply blend mode logic (like your DCTL)
    float finalTiltCR, finalTiltGM, finalTiltBY;
    switch (blendMode) {
        case BLEND_MODE_PRIMARY:    // Primary (RGB only)
            finalTiltCR = 1.0f;
            finalTiltGM = 1.0f;
            finalTiltBY = 1.0f;
            break;
        case BLEND_MODE_SECONDARY:  // Secondary (CYM only)
            finalTiltCR = 0.0f;
            finalTiltGM = 0.0f;
            finalTiltBY = 0.0f;
            break;
        case BLEND_MODE_MIX:        // Blend Control (use tilt values)
        default:
            finalTiltCR = tiltCR;
            finalTiltGM = tiltGM;
            finalTiltBY = tiltBY;
            break;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // PROCESSOR SETUP - CONFIGURE THE IMAGE PROCESSOR
    ////////////////////////////////////////////////////////////////////////////////
    
    // Set the input and output images
    p_ImageScaler.setDstImg(dst.get());
    p_ImageScaler.setSrcImg(src.get());

    // Setup GPU rendering arguments (for CUDA/OpenCL/Metal)
    p_ImageScaler.setGPURenderArgs(p_Args);

    // Set the region of the image to process
    p_ImageScaler.setRenderWindow(p_Args.renderWindow);

    // Pass all the parameter values to the processor
    p_ImageScaler.setContrastParams(gammaR, gammaG, gammaB, gammaC, gammaM, gammaY,
                                   midgreyR, midgreyG, midgreyB, midgreyC, midgreyM, midgreyY,
                                   finalTiltCR, finalTiltGM, finalTiltBY);

    ////////////////////////////////////////////////////////////////////////////////
    // EXECUTE PROCESSING
    ////////////////////////////////////////////////////////////////////////////////
    
    // Start the actual image processing (calls multiThreadProcessImages)
    p_ImageScaler.process();
}

////////////////////////////////////////////////////////////////////////////////
// PLUGIN FACTORY CLASS - CREATES PLUGIN INSTANCES AND DEFINES UI
////////////////////////////////////////////////////////////////////////////////

// class ContrastSatVolumeInteract : public OFX::OverlayInteract
// (This would be for custom overlay graphics - not implemented yet)

using namespace OFX;

////////////////////////////////////////////////////////////////////////////////
// FACTORY CONSTRUCTOR
////////////////////////////////////////////////////////////////////////////////

ContrastSatVolumeFactory::ContrastSatVolumeFactory()
    : OFX::PluginFactoryHelper<ContrastSatVolumeFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

////////////////////////////////////////////////////////////////////////////////
// PLUGIN DESCRIPTION - TELLS HOST ABOUT PLUGIN CAPABILITIES
////////////////////////////////////////////////////////////////////////////////

void ContrastSatVolumeFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
{
    // Set basic plugin information
    p_Desc.setLabels(kPluginName, kPluginName, kPluginName);
    p_Desc.setPluginGrouping(kPluginGrouping);
    p_Desc.setPluginDescription(kPluginDescription);

    // Add supported contexts (where this plugin can be used)
    p_Desc.addSupportedContext(eContextFilter);    // As a filter effect
    p_Desc.addSupportedContext(eContextGeneral);   // As a general effect

    // Add supported pixel formats
    p_Desc.addSupportedBitDepth(eBitDepthFloat);   // 32-bit float only

    // Set plugin capabilities flags
    p_Desc.setSingleInstance(false);                              // Multiple instances allowed
    p_Desc.setHostFrameThreading(false);                         // Handle our own threading
    p_Desc.setSupportsMultiResolution(kSupportsMultiResolution); // Multi-res support
    p_Desc.setSupportsTiles(kSupportsTiles);                     // Tile-based rendering
    p_Desc.setTemporalClipAccess(false);                         // No need for previous frames
    p_Desc.setRenderTwiceAlways(false);                          // Single pass rendering
    p_Desc.setSupportsMultipleClipPARs(kSupportsMultipleClipPARs); // Pixel aspect ratio support

    // Setup GPU rendering capabilities
    p_Desc.setSupportsOpenCLRender(true);  // OpenCL support

#ifndef __APPLE__
    // CUDA support on non-Apple systems
    p_Desc.setSupportsCudaRender(true);
    p_Desc.setSupportsCudaStream(false);
#endif

#ifdef __APPLE__
    // Metal support on Apple systems
    p_Desc.setSupportsMetalRender(true);
#endif

    // This plugin doesn't depend on neighboring pixels - good for LUT generation
    p_Desc.setNoSpatialAwareness(true);

    // Remove this line - no overlay interface needed for now
    // p_Desc.setOverlayInteractDescriptor(new ContrastSatVolumeOverlayInteractDescriptor());
}

////////////////////////////////////////////////////////////////////////////////
// PARAMETER DEFINITION HELPER FUNCTIONS
// These create different types of UI controls with appropriate defaults
////////////////////////////////////////////////////////////////////////////////

static DoubleParamDescriptor* defineScaleParam(OFX::ImageEffectDescriptor& p_Desc, const std::string& p_Name, const std::string& p_Label,
                                               const std::string& p_Hint, GroupParamDescriptor* p_Parent)
{
    // Generic scale parameter (legacy - for overall scale)
    DoubleParamDescriptor* param = p_Desc.defineDoubleParam(p_Name);
    param->setLabels(p_Label, p_Label, p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    param->setDefault(1);          // No change by default
    param->setRange(0, 10);        // Wide range for extreme adjustments
    param->setIncrement(0.1);      // Step size for UI
    param->setDisplayRange(0, 10); // Display range in UI
    param->setDoubleType(eDoubleTypeScale);

    if (p_Parent)
    {
        param->setParent(*p_Parent);
    }

    return param;
}

static DoubleParamDescriptor* defineGammaParam(OFX::ImageEffectDescriptor& p_Desc, const std::string& p_Name, const std::string& p_Label,
                                              const std::string& p_Hint, GroupParamDescriptor* p_Parent)
{
    // Gamma parameter for contrast adjustment
    DoubleParamDescriptor* param = p_Desc.defineDoubleParam(p_Name);
    param->setLabels(p_Label, p_Label, p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    param->setDefault(1.0);        // Gamma 1.0 = no change
    param->setRange(0.1, 4.0);     // Useful gamma range
    param->setDisplayRange(0.1, 4.0);
    param->setIncrement(0.01);     // Fine adjustment steps
    param->setDoubleType(eDoubleTypePlain);
    
    if (p_Parent) {
        param->setParent(*p_Parent);
    }
    return param;
}

static DoubleParamDescriptor* defineMidgreyParam(OFX::ImageEffectDescriptor& p_Desc, const std::string& p_Name, const std::string& p_Label,
                                                const std::string& p_Hint, GroupParamDescriptor* p_Parent, bool isRGB = true)
{
    // Midgrey parameter - the "pivot point" for gamma adjustments
    DoubleParamDescriptor* param = p_Desc.defineDoubleParam(p_Name);
    param->setLabels(p_Label, p_Label, p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    param->setDefault(isRGB ? 0.18 : 0.82);  // RGB = 18% grey, CYM = 82% grey (complement)
    param->setRange(0.01, 0.99);             // Avoid extreme values
    param->setDisplayRange(0.01, 0.99);
    param->setIncrement(0.01);               // Fine adjustment
    param->setDoubleType(eDoubleTypePlain);  // Changed from eDoubleTypeScale
    
    if (p_Parent) {
        param->setParent(*p_Parent);
    }
    return param;
}

static DoubleParamDescriptor* defineTiltParam(OFX::ImageEffectDescriptor& p_Desc, const std::string& p_Name, const std::string& p_Label,
                                             const std::string& p_Hint, GroupParamDescriptor* p_Parent)
{
    // Tilt parameter - controls blend between RGB and CYM
    DoubleParamDescriptor* param = p_Desc.defineDoubleParam(p_Name);
    param->setLabels(p_Label, p_Label, p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    param->setDefault(0.5);        // 50/50 blend by default
    param->setRange(0.0, 1.0);     // 0=pure CYM, 1=pure RGB
    param->setDisplayRange(0.0, 1.0);
    param->setIncrement(0.01);     // Fine blend control
    param->setDoubleType(eDoubleTypePlain);  // Changed from eDoubleTypeScale
    
    if (p_Parent) {
        param->setParent(*p_Parent);
    }
    return param;
}

////////////////////////////////////////////////////////////////////////////////
// UI CREATION - DEFINES ALL THE PARAMETER CONTROLS AND LAYOUT
////////////////////////////////////////////////////////////////////////////////

void ContrastSatVolumeFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
{
    ////////////////////////////////////////////////////////////////////////////////
    // CLIP DEFINITIONS - INPUT AND OUTPUT CONNECTIONS
    ////////////////////////////////////////////////////////////////////////////////
    
    // Create the mandated source clip (input image)
    ClipDescriptor* srcClip = p_Desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    srcClip->addSupportedComponent(ePixelComponentRGBA);  // RGBA images only
    srcClip->setTemporalClipAccess(false);                // No need for previous frames
    srcClip->setSupportsTiles(kSupportsTiles);            // Support tiled rendering
    srcClip->setIsMask(false);                            // Not a mask input

    // Create the mandated output clip
    ClipDescriptor* dstClip = p_Desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(ePixelComponentRGBA);  // RGBA output
    dstClip->addSupportedComponent(ePixelComponentAlpha); // Alpha-only output
    dstClip->setSupportsTiles(kSupportsTiles);            // Support tiled rendering

    ////////////////////////////////////////////////////////////////////////////////
    // PAGE AND GROUP CREATION - UI ORGANIZATION
    ////////////////////////////////////////////////////////////////////////////////
    
    // Create main page for all controls
    PageParamDescriptor* page = p_Desc.definePageParam("Controls");

    // Create groups to organize related controls
    GroupParamDescriptor* configGroup = p_Desc.defineGroupParam("ConfigGroup");
    configGroup->setHint("Configuration options");
    configGroup->setLabels("Configuration", "Configuration", "Configuration");


    // Blend Mode dropdown - FIRST in config group
    ChoiceParamDescriptor* blendMode = p_Desc.defineChoiceParam(kParamBlendMode);
    blendMode->setLabel(kParamBlendModeLabel);
    blendMode->setHint("Choose between RGB, CYM, or Blend Control modes");
    blendMode->appendOption("Primary (RGB)");     // Index 0
    blendMode->appendOption("Secondary (CYM)");   // Index 1  
    blendMode->appendOption("Blend Control");     // Index 2
    blendMode->setDefault(2); // Default to "Blend Control"
    blendMode->setAnimates(true);
    blendMode->setParent(*configGroup);
    page->addChild(*blendMode);

    // Gang RGB checkbox
    BooleanParamDescriptor* boolParam = p_Desc.defineBooleanParam("GangRGB");
    boolParam->setDefault(false);
    boolParam->setHint("Gang RGB controls together");
    boolParam->setLabels("Gang RGB", "Gang RGB", "Gang RGB");
    boolParam->setParent(*configGroup);
    page->addChild(*boolParam);

    // Gang CYM checkbox
    boolParam = p_Desc.defineBooleanParam("GangCYM");
    boolParam->setDefault(false);
    boolParam->setHint("Gang CYM controls together");
    boolParam->setLabels("Gang CYM", "Gang CYM", "Gang CYM");
    boolParam->setParent(*configGroup);
    page->addChild(*boolParam);

    // Alpha scale parameter
    DoubleParamDescriptor* param = defineScaleParam(p_Desc, "scaleA", "Alpha", "Scales the Alpha component", configGroup);
    page->addChild(*param);

    ////////////////////////////////////////////////////////////////////////////////
    // COLOR ADJUSTMENT GROUPS
    ////////////////////////////////////////////////////////////////////////////////
    
    // Red/Cyan Group
    GroupParamDescriptor* RedCyanGroup = p_Desc.defineGroupParam("RedCyan");
    RedCyanGroup->setHint("Red and Cyan contrast adjustments");
    RedCyanGroup->setLabels("Red/Cyan", "Red/Cyan", "Red/Cyan");

    
    // Red-Cyan tilt control - FIRST in group
    param = defineTiltParam(p_Desc, "tiltCR", "Red-Cyan Tilt", "Blend between Red and Cyan adjustments (0=Cyan, 1=Red)", RedCyanGroup);
    page->addChild(*param);
    
    // Red controls
    param = defineGammaParam(p_Desc, "scaleR", "Red Gamma", "Gamma contrast adjustment for Red channel", RedCyanGroup);
    page->addChild(*param);
    param = defineMidgreyParam(p_Desc, "scaleMidGreyR", "Red Mid Grey", "Midgrey pivot point for Red gamma", RedCyanGroup, true);
    page->addChild(*param);

    // Cyan controls
    param = defineGammaParam(p_Desc, "scaleC", "Cyan Gamma", "Gamma contrast adjustment for Cyan channel", RedCyanGroup);
    page->addChild(*param);
    param = defineMidgreyParam(p_Desc, "scaleMidGreyC", "Cyan Mid Grey", "Midgrey pivot point for Cyan gamma", RedCyanGroup, false);
    page->addChild(*param);

    // Green/Magenta Group
    GroupParamDescriptor* GreenMagentaGroup = p_Desc.defineGroupParam("GreenMagenta");
    GreenMagentaGroup->setHint("Green and Magenta contrast adjustments");
    GreenMagentaGroup->setLabels("Green/Magenta", "Green/Magenta", "Green/Magenta");

    
    param = defineTiltParam(p_Desc, "tiltGM", "Green-Magenta Tilt", "Blend between Green and Magenta adjustments (0=Magenta, 1=Green)", GreenMagentaGroup);
    page->addChild(*param);
    
    param = defineGammaParam(p_Desc, "scaleG", "Green Gamma", "Gamma contrast adjustment for Green channel", GreenMagentaGroup);
    page->addChild(*param);
    param = defineMidgreyParam(p_Desc, "scaleMidGreyG", "Green Mid Grey", "Midgrey pivot point for Green gamma", GreenMagentaGroup, true);
    page->addChild(*param);

    param = defineGammaParam(p_Desc, "scaleM", "Magenta Gamma", "Gamma contrast adjustment for Magenta channel", GreenMagentaGroup);
    page->addChild(*param);
    param = defineMidgreyParam(p_Desc, "scaleMidGreyM", "Magenta Mid Grey", "Midgrey pivot point for Magenta gamma", GreenMagentaGroup, false);
    page->addChild(*param);

    // Blue/Yellow Group
    GroupParamDescriptor* BlueYellowGroup = p_Desc.defineGroupParam("BlueYellow");
    BlueYellowGroup->setHint("Blue and Yellow contrast adjustments");
    BlueYellowGroup->setLabels("Blue/Yellow", "Blue/Yellow", "Blue/Yellow");

    
    param = defineTiltParam(p_Desc, "tiltBY", "Blue-Yellow Tilt", "Blend between Blue and Yellow adjustments (0=Yellow, 1=Blue)", BlueYellowGroup);
    page->addChild(*param);
    
    param = defineGammaParam(p_Desc, "scaleB", "Blue Gamma", "Gamma contrast adjustment for Blue channel", BlueYellowGroup);
    page->addChild(*param);
    param = defineMidgreyParam(p_Desc, "scaleMidGreyB", "Blue Mid Grey", "Midgrey pivot point for Blue gamma", BlueYellowGroup, true);
    page->addChild(*param);

    param = defineGammaParam(p_Desc, "scaleY", "Yellow Gamma", "Gamma contrast adjustment for Yellow channel", BlueYellowGroup);
    page->addChild(*param);
    param = defineMidgreyParam(p_Desc, "scaleMidGreyY", "Yellow Mid Grey", "Midgrey pivot point for Yellow gamma", BlueYellowGroup, false);
    page->addChild(*param);

    // ADD: Buy Me a Coffee button at the end
    PushButtonParamDescriptor* coffeeButton = p_Desc.definePushButtonParam("buymeacoffee");
    coffeeButton->setLabel("☕ Buy Me a Coffee");
    coffeeButton->setHint("Support the developer - opens external link to Buy Me a Coffee");
    page->addChild(*coffeeButton);
}

////////////////////////////////////////////////////////////////////////////////
// PLUGIN INSTANCE CREATION
////////////////////////////////////////////////////////////////////////////////

ImageEffect* ContrastSatVolumeFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    // Create and return a new instance of our plugin
    return new ContrastSatVolume(p_Handle);
}

////////////////////////////////////////////////////////////////////////////////
// PLUGIN REGISTRATION - TELLS OFX SYSTEM ABOUT OUR PLUGIN
////////////////////////////////////////////////////////////////////////////////

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    // Create our factory and register it with the OFX system
    static ContrastSatVolumeFactory contrastSatVolumePlugin;
    p_FactoryArray.push_back(&contrastSatVolumePlugin);
}

