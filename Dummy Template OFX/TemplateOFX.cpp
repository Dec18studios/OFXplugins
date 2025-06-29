/////// Step 1: Match your Header file name
#include "TemplateOFX.h"

#include <stdio.h>

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
#define kPluginName "Template OFX"        // <-- UPDATE: Change this
#define kPluginGrouping "Your Company"        // <-- UPDATE: Change this
#define kPluginDescription "Describe what your plugin does"  // <-- UPDATE: Change this
//// Don't change the openfx part of this just the name...///////
#define kPluginIdentifier "com.OpenFXSample.TemplateOFX""  // <-- UPDATE: Change this
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
    void setScales(float p_Scale1, float p_Scale2, float p_Scale3, float p_Scale4);
    
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
    float _scales[4];
    
    // Step 5: ADD YOUR PRIVATE MEMBERS HERE
    // Replace with your own parameters like:
    // float _yourParameter1;
    // bool _yourParameter2;
    // int _yourParameter3;
    //   // RGB gamma parameters - control contrast for Red, Green, Blue channels
    // float _gammaR, _gammaG, _gammaB;
    
    // CUSTOM Build Large Arrays use buildPresetArrays() method
    // Add your own helper methods if needed
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
extern void RunMetalKernel(void* p_CmdQ, int p_Width, int p_Height,
                          const float* p_Input, float* p_Output);

/* extern void RunComplexMetalKernel(void* p_CmdQ, int p_Width, int p_Height, 
                                  float* p_RgbGammas, float* p_CymGammas, 
                                  float* p_RgbMidgreys, float* p_CymMidgreys, 
                                  float* p_Tilts,
                                  const float* p_Input, float* p_Output);
*/
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

/*
    // Pass all parameter arrays
    float rgbGammas[3] = {_gammaR, _gammaG, _gammaB};
    float cymGammas[3] = {_gammaC, _gammaM, _gammaY};
    float rgbMidgreys[3] = {_midgreyR, _midgreyG, _midgreyB};
    float cymMidgreys[3] = {_midgreyC, _midgreyM, _midgreyY};
    float tilts[3] = {_tiltCR, _tiltGM, _tiltBY};
    
    RunComplexMetalKernel(_pMetalCmdQ, width, height,
                         rgbGammas, cymGammas, rgbMidgreys, cymMidgreys, tilts,
                         input, output);
*/

    RunMetalKernel(_pMetalCmdQ, width, height, input, output);
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

void ImageProcessor::setScales(float p_Scale1, float p_Scale2, float p_Scale3, float p_Scale4)
{
    _scales[0] = p_Scale1;
    _scales[1] = p_Scale2;
    _scales[2] = p_Scale3;
    _scales[3] = p_Scale4;
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
//TemplateOFX Should match everywhere in the code
class TemplateOFX : public OFX::ImageEffect  // <-- RENAME: Change class name
{
public:
    explicit TemplateOFX(OfxImageEffectHandle p_Handle);

    // BOILERPLATE: Keep these overrides
    virtual void render(const OFX::RenderArguments& p_Args);
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);
    virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);
    virtual void changedClip(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ClipName);

    void setEnabledness();
    void setupAndProcess(ImageProcessor &p_Processor, const OFX::RenderArguments& p_Args);

private:
    ////////////////////////////////////////////////////////////////////////////////
    // CLIP POINTERS - CONNECTIONS TO INPUT/OUTPUT IMAGES
    ////////////////////////////////////////////////////////////////////////////////

    // BOILERPLATE: Keep these basic clips
    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;

    ////////////////////////////////////////////////////////////////////////////////
    // PARAMETER POINTERS - CONNECTIONS TO UI CONTROLS
    ////////////////////////////////////////////////////////////////////////////////

    // Step 10: ADD YOUR PARAMETER DECLARATIONS HERE
    // BOILERPLATE: Basic scale parameters (modify as needed)
    OFX::DoubleParam* m_Scale;
    OFX::DoubleParam* m_Param1;  // <-- RENAME: Replace with your parameter names
    OFX::DoubleParam* m_Param2;
    OFX::DoubleParam* m_Param3;
    OFX::DoubleParam* m_Param4;
    OFX::BooleanParam* m_BoolParam;
/*
    // CYM gamma parameters - control contrast for complementary colors
    OFX::DoubleParam* m_ScaleC;  // Cyan gamma
    OFX::DoubleParam* m_ScaleM;  // Magenta gamma
    OFX::DoubleParam* m_ScaleY;  // Yellow gamma
    */
};
////////////////////////////////////////////////////////////////////////////////
// PLUGIN CONSTRUCTOR - CONNECTS TO ALL THE UI PARAMETERS
////////////////////////////////////////////////////////////////////////////////
// BOILERPLATE: Constructor - update parameter names
TemplateOFX::TemplateOFX(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    // Connect to input/output clips Don't Chage This
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

    // Step 11: FETCH YOUR PARAMETERS HERE
     // Connect to all the parameter controls created in describeInContext()
    // These parameter names must match exactly what was defined in describeInContext()
    m_Scale = fetchDoubleParam("scale");
    m_Param1 = fetchDoubleParam("param1");  // <-- UPDATE: Use your parameter names
    m_Param2 = fetchDoubleParam("param2");
    m_Param3 = fetchDoubleParam("param3");
    m_Param4 = fetchDoubleParam("param4");
    m_BoolParam = fetchBooleanParam("boolparam");
/*
    // CYM gamma parameters
    m_ScaleC = fetchDoubleParam("scaleC");
    m_ScaleM = fetchDoubleParam("scaleM");
    m_ScaleY = fetchDoubleParam("scaleY");
*/
    setEnabledness();
}

////////////////////////////////////////////////////////////////////////////////
// RENDER FUNCTION - MAIN ENTRY POINT FOR PROCESSING
////////////////////////////////////////////////////////////////////////////////
// BOILERPLATE: Keep render method structure
void TemplateOFX::render(const OFX::RenderArguments& p_Args)
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
// Step 12: UPDATE IDENTITY CHECK FOR YOUR EFFECT
bool TemplateOFX::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
    // CUSTOM REMOVED: Complex film simulation identity check
    // Replace with your own logic:
    double scale = m_Scale->getValueAtTime(p_Args.time);
    /*
        double gammaC = m_ScaleC->getValueAtTime(p_Args.time);
        double gammaM = m_ScaleM->getValueAtTime(p_Args.time);
        double gammaY = m_ScaleY->getValueAtTime(p_Args.time);
    */
    if (scale == 1.0) // Example: if scale is 1.0, pass through unchanged
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
// Step 13: UPDATE PARAMETER CHANGE HANDLING
void TemplateOFX::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
    // CUSTOM REMOVED: Film-specific parameter change handling
    // Add your own parameter change logic:
    if (p_ParamName == "boolparam") // Example
    {
        setEnabledness();
    }
}
////////////////////////////////////////////////////////////////////////////////
// CLIP CHANGE HANDLER - RESPONDS TO INPUT CHANGES
////////////////////////////////////////////////////////////////////////////////
void TemplateOFX::changedClip(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ClipName)
{
    if (p_ClipName == kOfxImageEffectSimpleSourceClipName)
    {
        setEnabledness();
    }
}
////////////////////////////////////////////////////////////////////////////////
// UI CONTROL ENABLEMENT - MANAGES WHICH CONTROLS ARE AVAILABLE
////////////////////////////////////////////////////////////////////////////////
// Step 14: UPDATE PARAMETER ENABLEMENT LOGIC
void TemplateOFX::setEnabledness()
{
    // CUSTOM REMOVED: Complex film stock enablement logic
    // Add your own parameter enablement logic:
    bool boolValue = m_BoolParam->getValue();
    m_Param1->setEnabled(boolValue);  // Example: enable param1 when bool is true

    /*
      // Enable CYM gamma parameters
    m_ScaleC->setEnabled(hasRGBAInput);
    m_ScaleM->setEnabled(hasRGBAInput && !gangCYM);  // Disable M when ganged to C
    m_ScaleY->setEnabled(hasRGBAInput && !gangCYM);  // Disable Y when ganged to C
    */
}
////////////////////////////////////////////////////////////////////////////////
// SETUP AND PROCESS - COORDINATES THE ENTIRE PROCESSING PIPELINE
////////////////////////////////////////////////////////////////////////////////
// Step 15: UPDATE PARAMETER FETCHING AND PROCESSING
void TemplateOFX::setupAndProcess(ImageProcessor& p_Processor, const OFX::RenderArguments& p_Args)
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
    // Step 16: GET YOUR PARAMETER VALUES HERE
    // Replace with your own:
    double scale = m_Scale->getValueAtTime(p_Args.time);
    double param1 = m_Param1->getValueAtTime(p_Args.time);
    double param2 = m_Param2->getValueAtTime(p_Args.time);
    double param3 = m_Param3->getValueAtTime(p_Args.time);
    double param4 = m_Param4->getValueAtTime(p_Args.time);
    bool boolValue = m_BoolParam->getValueAtTime(p_Args.time);

    /*
    float gammaC = m_ScaleC->getValueAtTime(p_Args.time);
    float gammaM = m_ScaleM->getValueAtTime(p_Args.time);
    float gammaY = m_ScaleY->getValueAtTime(p_Args.time);
    */
    ////////////////////////////////////////////////////////////////////////////////
    // PROCESSOR SETUP - CONFIGURE THE IMAGE PROCESSOR
    ////////////////////////////////////////////////////////////////////////////////
    // BOILERPLATE: Keep processor setup
    p_Processor.setDstImg(dst.get());
    p_Processor.setSrcImg(src.get());
    p_Processor.setGPURenderArgs(p_Args);
    p_Processor.setRenderWindow(p_Args.renderWindow);

    // Step 17: PASS YOUR PARAMETERS TO PROCESSOR
    // CUSTOM REMOVED: Call to setProjectorParams with 20+ parameters
    // Replace with your own parameter setter:
    p_Processor.setScales(param1, param2, param3, param4);
    // p_Processor.setYourParams(param1, boolValue, etc...);
    /*
    p_Processor.setContrastParams(gammaR, gammaG, gammaB, gammaC, gammaM, gammaY,
                                   midgreyR, midgreyG, midgreyB, midgreyC, midgreyM, midgreyY,
                                   finalTiltCR, finalTiltGM, finalTiltBY);
    */
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
/*
ContrastSatVolumeFactory::ContrastSatVolumeFactory()
    : OFX::PluginFactoryHelper<ContrastSatVolumeFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}
*/
// RENAME: Change class name to match your plugin
class TemplateOFXFactory : public OFX::PluginFactoryHelper<TemplateOFXFactory>
{
public:
    TemplateOFXFactory() : OFX::PluginFactoryHelper<TemplateOFXFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor) {}
    virtual void describe(OFX::ImageEffectDescriptor& p_Desc);
    virtual void describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum p_Context);
    virtual ImageEffect* createInstance(OfxImageEffectHandle p_Handle, ContextEnum p_Context);
};

////////////////////////////////////////////////////////////////////////////////
// PLUGIN DESCRIPTION - TELLS HOST ABOUT PLUGIN CAPABILITIES
////////////////////////////////////////////////////////////////////////////////
// BOILERPLATE: Keep basic plugin description
void TemplateOFXFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
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
    p_Desc.setOverlayInteractDescriptor(new YourOverlayDescriptor());  // Remove if no overlay
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
void TemplateOFXFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
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
    
    // Step 20: ADD YOUR PARAMETER GROUPS HERE
    GroupParamDescriptor* mainGroup = p_Desc.defineGroupParam("MainGroup");
    mainGroup->setHint("Main parameters for your effect");
    mainGroup->setLabels("Main Controls", "Main Controls", "Main Controls");

    // Step 21: ADD YOUR PARAMETERS HERE
    // Replace with your own parameters:
    
    DoubleParamDescriptor* param = defineDoubleParam(p_Desc, "scale", "Scale", "Overall scale factor", 
                                                    nullptr, 1.0, 0.0, 10.0, 0.1);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "param1", "Parameter 1", "First parameter", 
                             mainGroup, 1.0, 0.0, 5.0, 0.1);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "param2", "Parameter 2", "Second parameter", 
                             mainGroup, 1.0, 0.0, 5.0, 0.1);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "param3", "Parameter 3", "Third parameter", 
                             mainGroup, 1.0, 0.0, 5.0, 0.1);
    page->addChild(*param);

    param = defineDoubleParam(p_Desc, "param4", "Parameter 4", "Fourth parameter", 
                             mainGroup, 1.0, 0.0, 5.0, 0.1);
    page->addChild(*param);

    // Example boolean parameter
    BooleanParamDescriptor* boolParam = p_Desc.defineBooleanParam("boolparam");
    boolParam->setDefault(false);
    boolParam->setHint("Enable special processing");
    boolParam->setLabels("Enable Effect", "Enable Effect", "Enable Effect");
    boolParam->setParent(*mainGroup);
    page->addChild(*boolParam);

    // Example choice parameter
    ChoiceParamDescriptor* choiceParam = p_Desc.defineChoiceParam("choiceparam");
    choiceParam->setLabel("Mode");
    choiceParam->setHint("Choose processing mode");
    choiceParam->appendOption("Mode 1");
    choiceParam->appendOption("Mode 2");
    choiceParam->appendOption("Mode 3");
    choiceParam->setDefault(0);
    choiceParam->setAnimates(true);
    choiceParam->setParent(*mainGroup);
    page->addChild(*choiceParam);
}
////////////////////////////////////////////////////////////////////////////////
// PLUGIN INSTANCE  Creation Boilerplate
////////////////////////////////////////////////////////////////////////////////
ImageEffect* TemplateOFXFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new TemplateOFX(p_Handle);  // <-- UPDATE: Match your class name
}
////////////////////////////////////////////////////////////////////////////////
// PLUGIN REGISTRATION - TELLS OFX SYSTEM ABOUT OUR PLUGIN
////////////////////////////////////////////////////////////////////////////////
// BOILERPLATE: Keep plugin registration
void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static TemplateOFXFactory TemplateOFX;  // <-- UPDATE: Match your factory name
    p_FactoryArray.push_back(&TemplateOFX);
}
