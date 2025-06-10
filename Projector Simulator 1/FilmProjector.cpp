////////////////////////////////////////////////////////////////////////////////
// INCLUDES AND DEFINITIONS
////////////////////////////////////////////////////////////////////////////////
#include "FilmProjector.h"

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
#define kPluginName "Film Projector Sim"
#define kPluginGrouping "Greg Enright"
#define kPluginDescription "Apply seperate RGB and CYM Contrast Adjustments to each channel"
#define kPluginIdentifier "com.OpenFXSample.FilmProjector"
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
    
    // Replace setContrastParams with new parameter setter
    void setProjectorParams(float p_NegMean, float p_NegLeft, float p_NegRight, float p_NegMax, float p_NegMin,
                           float p_SNegMean, float p_SNegLeft, float p_SNegRight, float p_SNegMax, float p_SNegMin, float p_SNegRatio,
                           float p_PrintMean, float p_PrintLeft, float p_PrintRight, float p_PrintMax, float p_PrintMin,
                           float p_SPrintMean, float p_SPrintLeft, float p_SPrintRight, float p_SPrintMax, float p_SPrintMin, float p_SPrintRatio);

private:
    OFX::Image* _srcImg;
    
    // Replace all the old gamma/midgrey/tilt parameters with new ones
    float _negMean, _negLeft, _negRight, _negMax, _negMin;
    float _sNegMean, _sNegLeft, _sNegRight, _sNegMax, _sNegMin, _sNegRatio;
    float _printMean, _printLeft, _printRight, _printMax, _printMin;
    float _sPrintMean, _sPrintLeft, _sPrintRight, _sPrintMax, _sPrintMin, _sPrintRatio;
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
extern void RunProjectorCudaKernel(void* p_Stream, int p_Width, int p_Height,
                                  float* p_NegativePreset,  // 22 float array
                                  float* p_PrintPreset,     // 22 float array
                                  const float* p_Input, float* p_Output);
#endif

#ifdef __APPLE__
extern void RunMetalKernel(void* p_CmdQ, int p_Width, int p_Height, float* p_Gain, const float* p_Input, float* p_Output);
extern void RunProjectorMetalKernel(void* p_CmdQ, int p_Width, int p_Height, 
                                   float* p_NegativePreset,  // 22 float array
                                   float* p_PrintPreset,     // 22 float array
                                   const float* p_Input, float* p_Output);
#endif

extern void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float* p_Gain, const float* p_Input, float* p_Output);
extern void RunProjectorOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, 
                                    float* p_NegativePreset,  // 22 float array
                                    float* p_PrintPreset,     // 22 float array
                                    const float* p_Input, float* p_Output);
////////////////////////////////////////////////////////////////////////////////
// CORE ALGORITHM FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

// Helper function 

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
                

                
                // Mix results - each output channel gets specific adjustment blend
                float finalR = r;  // Red output uses Red gamma vs Cyan gamma
                float finalG = g;  // Green output uses Green gamma vs Magenta gamma
                float finalB = b;  // Blue output uses Blue gamma vs Yellow gamma
                
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

    // Create arrays for the projector simulation parameters
    float negParams[5] = {_negMean, _negLeft, _negRight, _negMax, _negMin};
    float sNegParams[6] = {_sNegMean, _sNegLeft, _sNegRight, _sNegMax, _sNegMin, _sNegRatio};
    float printParams[5] = {_printMean, _printLeft, _printRight, _printMax, _printMin};
    float sPrintParams[6] = {_sPrintMean, _sPrintLeft, _sPrintRight, _sPrintMax, _sPrintMin, _sPrintRatio};
    
    RunProjectorCudaKernel(_pCudaStream, width, height,
                          negParams, sNegParams, printParams, sPrintParams,
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

    // Create arrays for the projector simulation parameters
    float negParams[5] = {_negMean, _negLeft, _negRight, _negMax, _negMin};
    float sNegParams[6] = {_sNegMean, _sNegLeft, _sNegRight, _sNegMax, _sNegMin, _sNegRatio};
    float printParams[5] = {_printMean, _printLeft, _printRight, _printMax, _printMin};
    float sPrintParams[6] = {_sPrintMean, _sPrintLeft, _sPrintRight, _sPrintMax, _sPrintMin, _sPrintRatio};
    
    RunProjectorMetalKernel(_pMetalCmdQ, width, height,
                           negParams, sNegParams, printParams, sPrintParams,
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

    // Create arrays for the projector simulation parameters
    float negParams[5] = {_negMean, _negLeft, _negRight, _negMax, _negMin};
    float sNegParams[6] = {_sNegMean, _sNegLeft, _sNegRight, _sNegMax, _sNegMin, _sNegRatio};
    float printParams[5] = {_printMean, _printLeft, _printRight, _printMax, _printMin};
    float sPrintParams[6] = {_sPrintMean, _sPrintLeft, _sPrintRight, _sPrintMax, _sPrintMin, _sPrintRatio};
    
    RunProjectorOpenCLKernel(_pOpenCLCmdQ, width, height,
                            negParams, sNegParams, printParams, sPrintParams,
                            input, output);
}

////////////////////////////////////////////////////////////////////////////////
// PARAMETER SETTER FUNCTION
// This receives all the UI parameter values and stores them for processing
////////////////////////////////////////////////////////////////////////////////

void ImageScaler::setProjectorParams(float p_NegMean, float p_NegLeft, float p_NegRight, float p_NegMax, float p_NegMin,
                                     float p_SNegMean, float p_SNegLeft, float p_SNegRight, float p_SNegMax, float p_SNegMin, float p_SNegRatio,
                                     float p_PrintMean, float p_PrintLeft, float p_PrintRight, float p_PrintMax, float p_PrintMin,
                                     float p_SPrintMean, float p_SPrintLeft, float p_SPrintRight, float p_SPrintMax, float p_SPrintMin, float p_SPrintRatio)
{
    // Store negative parameters
    _negMean = p_NegMean; _negLeft = p_NegLeft; _negRight = p_NegRight; _negMax = p_NegMax; _negMin = p_NegMin;
    
    // Store negative silver parameters
    _sNegMean = p_SNegMean; _sNegLeft = p_SNegLeft; _sNegRight = p_SNegRight; 
    _sNegMax = p_SNegMax; _sNegMin = p_SNegMin; _sNegRatio = p_SNegRatio;
    
    // Store print parameters
    _printMean = p_PrintMean; _printLeft = p_PrintLeft; _printRight = p_PrintRight; _printMax = p_PrintMax; _printMin = p_PrintMin;
    
    // Store print silver parameters
    _sPrintMean = p_SPrintMean; _sPrintLeft = p_SPrintLeft; _sPrintRight = p_SPrintRight;
    _sPrintMax = p_SPrintMax; _sPrintMin = p_SPrintMin; _sPrintRatio = p_SPrintRatio;
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
class FilmProjector : public OFX::ImageEffect
{
public:
    explicit FilmProjector(OfxImageEffectHandle p_Handle);

    virtual void render(const OFX::RenderArguments& p_Args);
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);
    virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);
    virtual void changedClip(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ClipName);

    void setEnabledness();
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

    // Replace all old parameters with projector simulation parameters
    OFX::DoubleParam* m_NegMean;
    OFX::DoubleParam* m_NegLeft;
    OFX::DoubleParam* m_NegRight;
    OFX::DoubleParam* m_NegMax;
    OFX::DoubleParam* m_NegMin;
    
    OFX::DoubleParam* m_SNegMean;
    OFX::DoubleParam* m_SNegLeft;
    OFX::DoubleParam* m_SNegRight;
    OFX::DoubleParam* m_SNegMax;
    OFX::DoubleParam* m_SNegMin;
    OFX::DoubleParam* m_SNegRatio;
    
    OFX::DoubleParam* m_PrintMean;
    OFX::DoubleParam* m_PrintLeft;
    OFX::DoubleParam* m_PrintRight;
    OFX::DoubleParam* m_PrintMax;
    OFX::DoubleParam* m_PrintMin;
    
    OFX::DoubleParam* m_SPrintMean;
    OFX::DoubleParam* m_SPrintLeft;
    OFX::DoubleParam* m_SPrintRight;
    OFX::DoubleParam* m_SPrintMax;
    OFX::DoubleParam* m_SPrintMin;
    OFX::DoubleParam* m_SPrintRatio;
};

////////////////////////////////////////////////////////////////////////////////
// PLUGIN CONSTRUCTOR - CONNECTS TO ALL THE UI PARAMETERS
////////////////////////////////////////////////////////////////////////////////

FilmProjector::FilmProjector(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

    // Connect to projector simulation parameters
    m_NegMean = fetchDoubleParam(kParamNegMean);
    m_NegLeft = fetchDoubleParam(kParamNegLeft);
    m_NegRight = fetchDoubleParam(kParamNegRight);
    m_NegMax = fetchDoubleParam(kParamNegMax);
    m_NegMin = fetchDoubleParam(kParamNegMin);
    
    m_SNegMean = fetchDoubleParam(kParamSNegMean);
    m_SNegLeft = fetchDoubleParam(kParamSNegLeft);
    m_SNegRight = fetchDoubleParam(kParamSNegRight);
    m_SNegMax = fetchDoubleParam(kParamSNegMax);
    m_SNegMin = fetchDoubleParam(kParamSNegMin);
    m_SNegRatio = fetchDoubleParam(kParamSNegRatio);
    
    m_PrintMean = fetchDoubleParam(kParamPrintMean);
    m_PrintLeft = fetchDoubleParam(kParamPrintLeft);
    m_PrintRight = fetchDoubleParam(kParamPrintRight);
    m_PrintMax = fetchDoubleParam(kParamPrintMax);
    m_PrintMin = fetchDoubleParam(kParamPrintMin);
    
    m_SPrintMean = fetchDoubleParam(kParamSPrintMean);
    m_SPrintLeft = fetchDoubleParam(kParamSPrintLeft);
    m_SPrintRight = fetchDoubleParam(kParamSPrintRight);
    m_SPrintMax = fetchDoubleParam(kParamSPrintMax);
    m_SPrintMin = fetchDoubleParam(kParamSPrintMin);
    m_SPrintRatio = fetchDoubleParam(kParamSPrintRatio);

    setEnabledness();
}

////////////////////////////////////////////////////////////////////////////////
// RENDER FUNCTION - MAIN ENTRY POINT FOR PROCESSING
////////////////////////////////////////////////////////////////////////////////

void FilmProjector::render(const OFX::RenderArguments& p_Args)
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

bool FilmProjector::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
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

void FilmProjector::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
    // When gang control parameters change, update which controls are enabled
    if (p_ParamName == "GangRGB" || p_ParamName == "GangCYM")
    {
        setEnabledness();
    }
}

////////////////////////////////////////////////////////////////////////////////
// CLIP CHANGE HANDLER - RESPONDS TO INPUT CHANGES
////////////////////////////////////////////////////////////////////////////////

void FilmProjector::changedClip(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ClipName)
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

void FilmProjector::setEnabledness()
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

void FilmProjector::setupAndProcess(ImageScaler& p_ImageScaler, const OFX::RenderArguments& p_Args)
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
    p_ImageScaler.setProjectorParams(gammaR, gammaG, gammaB, gammaC, gammaM, gammaY,
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

// class FilmProjectorInteract : public OFX::OverlayInteract
// (This would be for custom overlay graphics - not implemented yet)

using namespace OFX;

////////////////////////////////////////////////////////////////////////////////
// FACTORY CONSTRUCTOR
////////////////////////////////////////////////////////////////////////////////

FilmProjectorFactory::FilmProjectorFactory()
    : OFX::PluginFactoryHelper<FilmProjectorFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

////////////////////////////////////////////////////////////////////////////////
// PLUGIN DESCRIPTION - TELLS HOST ABOUT PLUGIN CAPABILITIES
////////////////////////////////////////////////////////////////////////////////

void FilmProjectorFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
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
    // p_Desc.setOverlayInteractDescriptor(new FilmProjectorOverlayInteractDescriptor());
}

////////////////////////////////////////////////////////////////////////////////
// PARAMETER DEFINITIONS - UPDATED FROM PROJECTORSIM
////////////////////////////////////////////////////////////////////////////////

// Replace the existing parameter definitions with these from ProjectorSim
#define kParamNegMean "c_neg_mean"
#define kParamNegMeanLabel "Negative Mean"

#define kParamNegLeft "c_neg_left"
#define kParamNegLeftLabel "Negative Left"

#define kParamNegRight "c_neg_right"
#define kParamNegRightLabel "Negative Right"

#define kParamNegMax "c_neg_max"
#define kParamNegMaxLabel "Negative Max"

#define kParamNegMin "c_neg_min"
#define kParamNegMinLabel "Negative Min"

#define kParamSNegMean "c_sneg_mean"
#define kParamSNegMeanLabel "Neg Silver Mean"

#define kParamSNegLeft "c_sneg_left"
#define kParamSNegLeftLabel "Neg Silver Left"

#define kParamSNegRight "c_sneg_right"
#define kParamSNegRightLabel "Neg Silver Right"

#define kParamSNegMax "c_sneg_max"
#define kParamSNegMaxLabel "Neg Silver Max"

#define kParamSNegMin "c_sneg_min"
#define kParamSNegMinLabel "Neg Silver Min"

#define kParamSNegRatio "c_sratio"
#define kParamSNegRatioLabel "Neg Silver Ratio"

#define kParamPrintMean "c_print_mean"
#define kParamPrintMeanLabel "Print Mean"

#define kParamPrintLeft "c_print_left"
#define kParamPrintLeftLabel "Print Left"

#define kParamPrintRight "c_print_right"
#define kParamPrintRightLabel "Print Right"

#define kParamPrintMax "c_print_max"
#define kParamPrintMaxLabel "Print Max"

#define kParamPrintMin "c_print_min"
#define kParamPrintMinLabel "Print Min"

#define kParamSPrintMean "c_sprint_mean"
#define kParamSPrintMeanLabel "Print Silver Mean"

#define kParamSPrintLeft "c_sprint_left"
#define kParamSPrintLeftLabel "Print Silver Left"

#define kParamSPrintRight "c_sprint_right"
#define kParamSPrintRightLabel "Print Silver Right"

#define kParamSPrintMax "c_sprint_max"
#define kParamSPrintMaxLabel "Print Silver Max"

#define kParamSPrintMin "c_sprint_min"
#define kParamSPrintMinLabel "Print Silver Min"

#define kParamSPrintRatio "c_spratio"
#define kParamSPrintRatioLabel "Print Silver Ratio"

// Remove old blend mode enum and parameter definitions
////////////////////////////////////////////////////////////////////////////////
// PARAMETER DEFINITION HELPER FUNCTIONS
// These create different types of UI controls with appropriate defaults
////////////////////////////////////////////////////////////////////////////////

static DoubleParamDescriptor* defineMeanParam(OFX::ImageEffectDescriptor& p_Desc, const std::string& p_Name, const std::string& p_Label,
                                               const std::string& p_Hint, GroupParamDescriptor* p_Parent)
{
    DoubleParamDescriptor* param = p_Desc.defineDoubleParam(p_Name);
    param->setLabels(p_Label, p_Label, p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    param->setDefault(650);
    param->setRange(300, 800);
    param->setIncrement(0.01);
    param->setDisplayRange(300, 800);
    param->setDoubleType(eDoubleTypePlain);

    if (p_Parent)
    {
        param->setParent(*p_Parent);
    }

    return param;
}

static DoubleParamDescriptor* defineLeftStdParam(OFX::ImageEffectDescriptor& p_Desc, const std::string& p_Name, const std::string& p_Label,
                                              const std::string& p_Hint, GroupParamDescriptor* p_Parent)
{
    DoubleParamDescriptor* param = p_Desc.defineDoubleParam(p_Name);
    param->setLabels(p_Label, p_Label, p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    param->setDefault(50);
    param->setRange(10, 150);
    param->setIncrement(0.01);
    param->setDisplayRange(10, 150);
    param->setDoubleType(eDoubleTypePlain);
    
    if (p_Parent) {
        param->setParent(*p_Parent);
    }
    return param;
}

static DoubleParamDescriptor* defineRightStdParam(OFX::ImageEffectDescriptor& p_Desc, const std::string& p_Name, const std::string& p_Label,
                                              const std::string& p_Hint, GroupParamDescriptor* p_Parent)
{
    DoubleParamDescriptor* param = p_Desc.defineDoubleParam(p_Name);
    param->setLabels(p_Label, p_Label, p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    param->setDefault(50);
    param->setRange(10, 150);
    param->setIncrement(0.01);
    param->setDisplayRange(10, 150);
    param->setDoubleType(eDoubleTypePlain);
    
    if (p_Parent) {
        param->setParent(*p_Parent);
    }
    return param;
}

static DoubleParamDescriptor* defineMaxParam(OFX::ImageEffectDescriptor& p_Desc, const std::string& p_Name, const std::string& p_Label,
                                             const std::string& p_Hint, GroupParamDescriptor* p_Parent)
{
    DoubleParamDescriptor* param = p_Desc.defineDoubleParam(p_Name);
    param->setLabels(p_Label, p_Label, p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    param->setDefault(4.0);
    param->setRange(1.0, 10.0);
    param->setIncrement(0.01);
    param->setDisplayRange(1.0, 10.0);
    param->setDoubleType(eDoubleTypePlain);
    
    if (p_Parent) {
        param->setParent(*p_Parent);
    }
    return param;
}

static DoubleParamDescriptor* defineMinParam(OFX::ImageEffectDescriptor& p_Desc, const std::string& p_Name, const std::string& p_Label,
                                             const std::string& p_Hint, GroupParamDescriptor* p_Parent)
{
    DoubleParamDescriptor* param = p_Desc.defineDoubleParam(p_Name);
    param->setLabels(p_Label, p_Label, p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    param->setDefault(0.04);
    param->setRange(0.01, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.01, 1.0);
    param->setDoubleType(eDoubleTypePlain);
    
    if (p_Parent) {
        param->setParent(*p_Parent);
    }
    return param;
}

static DoubleParamDescriptor* defineRatioParam(OFX::ImageEffectDescriptor& p_Desc, const std::string& p_Name, const std::string& p_Label,
                                             const std::string& p_Hint, GroupParamDescriptor* p_Parent)
{
    DoubleParamDescriptor* param = p_Desc.defineDoubleParam(p_Name);
    param->setLabels(p_Label, p_Label, p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    param->setDefault(0.5);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.01);
    param->setDisplayRange(0.0, 1.0);
    param->setDoubleType(eDoubleTypePlain);
    
    if (p_Parent) {
        param->setParent(*p_Parent);
    }
    return param;
}

////////////////////////////////////////////////////////////////////////////////
// UI CREATION - DEFINES ALL THE PARAMETER CONTROLS AND LAYOUT
////////////////////////////////////////////////////////////////////////////////

void FilmProjectorFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
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

    ////////////////////////////////////////////////////////////////////////////////
    // NEGATIVE FILM PARAMETERS
    ////////////////////////////////////////////////////////////////////////////////
    
    // Negative Group
    GroupParamDescriptor* negativeGroup = p_Desc.defineGroupParam("NegativeGroup");
    negativeGroup->setHint("Negative film characteristics");
    negativeGroup->setLabels("Negative Film", "Negative Film", "Negative Film");

    DoubleParamDescriptor* param = defineMeanParam(p_Desc, kParamNegMean, kParamNegMeanLabel, "Mean density of negative film", negativeGroup);
    page->addChild(*param);
    
    param = defineLeftStdParam(p_Desc, kParamNegLeft, kParamNegLeftLabel, "Left standard deviation for negative", negativeGroup);
    page->addChild(*param);
    
    param = defineRightStdParam(p_Desc, kParamNegRight, kParamNegRightLabel, "Right standard deviation for negative", negativeGroup);
    page->addChild(*param);
    
    param = defineMaxParam(p_Desc, kParamNegMax, kParamNegMaxLabel, "Maximum density for negative", negativeGroup);
    page->addChild(*param);
    
    param = defineMinParam(p_Desc, kParamNegMin, kParamNegMinLabel, "Minimum density for negative", negativeGroup);
    page->addChild(*param);

    ////////////////////////////////////////////////////////////////////////////////
    // NEGATIVE SILVER PARAMETERS
    ////////////////////////////////////////////////////////////////////////////////
    
    // Negative Silver Group
    GroupParamDescriptor* negSilverGroup = p_Desc.defineGroupParam("NegSilverGroup");
    negSilverGroup->setHint("Negative silver halide characteristics");
    negSilverGroup->setLabels("Negative Silver", "Negative Silver", "Negative Silver");

    param = defineMeanParam(p_Desc, kParamSNegMean, kParamSNegMeanLabel, "Mean density of negative silver", negSilverGroup);
    page->addChild(*param);
    
    param = defineLeftStdParam(p_Desc, kParamSNegLeft, kParamSNegLeftLabel, "Left standard deviation for negative silver", negSilverGroup);
    page->addChild(*param);
    
    param = defineRightStdParam(p_Desc, kParamSNegRight, kParamSNegRightLabel, "Right standard deviation for negative silver", negSilverGroup);
    page->addChild(*param);
    
    param = defineMaxParam(p_Desc, kParamSNegMax, kParamSNegMaxLabel, "Maximum density for negative silver", negSilverGroup);
    page->addChild(*param);
    
    param = defineMinParam(p_Desc, kParamSNegMin, kParamSNegMinLabel, "Minimum density for negative silver", negSilverGroup);
    page->addChild(*param);
    
    param = defineRatioParam(p_Desc, kParamSNegRatio, kParamSNegRatioLabel, "Silver ratio for negative", negSilverGroup);
    page->addChild(*param);

    ////////////////////////////////////////////////////////////////////////////////
    // PRINT FILM PARAMETERS
    ////////////////////////////////////////////////////////////////////////////////
    
    // Print Group
    GroupParamDescriptor* printGroup = p_Desc.defineGroupParam("PrintGroup");
    printGroup->setHint("Print film characteristics");
    printGroup->setLabels("Print Film", "Print Film", "Print Film");

    param = defineMeanParam(p_Desc, kParamPrintMean, kParamPrintMeanLabel, "Mean density of print film", printGroup);
    page->addChild(*param);
    
    param = defineLeftStdParam(p_Desc, kParamPrintLeft, kParamPrintLeftLabel, "Left standard deviation for print", printGroup);
    page->addChild(*param);
    
    param = defineRightStdParam(p_Desc, kParamPrintRight, kParamPrintRightLabel, "Right standard deviation for print", printGroup);
    page->addChild(*param);
    
    param = defineMaxParam(p_Desc, kParamPrintMax, kParamPrintMaxLabel, "Maximum density for print", printGroup);
    page->addChild(*param);
    
    param = defineMinParam(p_Desc, kParamPrintMin, kParamPrintMinLabel, "Minimum density for print", printGroup);
    page->addChild(*param);

    ////////////////////////////////////////////////////////////////////////////////
    // PRINT SILVER PARAMETERS
    ////////////////////////////////////////////////////////////////////////////////
    
    // Print Silver Group
    GroupParamDescriptor* printSilverGroup = p_Desc.defineGroupParam("PrintSilverGroup");
    printSilverGroup->setHint("Print silver halide characteristics");
    printSilverGroup->setLabels("Print Silver", "Print Silver", "Print Silver");

    param = defineMeanParam(p_Desc, kParamSPrintMean, kParamSPrintMeanLabel, "Mean density of print silver", printSilverGroup);
    page->addChild(*param);
    
    param = defineLeftStdParam(p_Desc, kParamSPrintLeft, kParamSPrintLeftLabel, "Left standard deviation for print silver", printSilverGroup);
    page->addChild(*param);
    
    param = defineRightStdParam(p_Desc, kParamSPrintRight, kParamSPrintRightLabel, "Right standard deviation for print silver", printSilverGroup);
    page->addChild(*param);
    
    param = defineMaxParam(p_Desc, kParamSPrintMax, kParamSPrintMaxLabel, "Maximum density for print silver", printSilverGroup);
    page->addChild(*param);
    
    param = defineMinParam(p_Desc, kParamSPrintMin, kParamSPrintMinLabel, "Minimum density for print silver", printSilverGroup);
    page->addChild(*param);
    
    param = defineRatioParam(p_Desc, kParamSPrintRatio, kParamSPrintRatioLabel, "Silver ratio for print", printSilverGroup);
    page->addChild(*param);
}
////////////////////////////////////////////////////////////////////////////////
// PLUGIN INSTANCE CREATION
////////////////////////////////////////////////////////////////////////////////

ImageEffect* FilmProjectorFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    // Create and return a new instance of our plugin
    return new FilmProjector(p_Handle);
}

////////////////////////////////////////////////////////////////////////////////
// PLUGIN REGISTRATION - TELLS OFX SYSTEM ABOUT OUR PLUGIN
////////////////////////////////////////////////////////////////////////////////

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    // Create our factory and register it with the OFX system
    static FilmProjectorFactory FilmProjectorPlugin;
    p_FactoryArray.push_back(&FilmProjectorPlugin);
}
{
    // Create and return a new instance of our plugin    return new FilmProjector(p_Handle);}////////////////////////////////////////////////////////////////////////////////// PLUGIN REGISTRATION - TELLS OFX SYSTEM ABOUT OUR PLUGIN////////////////////////////////////////////////////////////////////////////////void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray){    // Create our factory and register it with the OFX system    static FilmProjectorFactory FilmProjectorPlugin;    p_FactoryArray.push_back(&FilmProjectorPlugin);}