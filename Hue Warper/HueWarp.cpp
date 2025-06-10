/////// Step 1: Match your Header file name
#include "HueWarp.h"

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
#define kPluginName "Hue Warp"        
#define kPluginGrouping "create@Dec18Studios.com"        
#define kPluginDescription "Warps Hues"  
#define kPluginIdentifier "com.OpenFXSample.HueWarp"  
#define kPluginVersionMajor 1
#define kPluginVersionMinor 0

// BOILERPLATE: Keep these unless you need different capabilities
#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

// Step 3: ADD CONSTANTS for multiple hue mappings
#define MAX_COLOR_MAPPINGS 8

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
    void setColorParams(float origR, float origG, float origB, 
                   float targR, float targG, float targB);

    // ADD: Multiple mapping support
    void setMappingCount(int count);
    void setMappingParams(int index, float srcHue, float srcRange, float targetHue, 
                         float targetRange, float strength, float bend, bool enabled, bool anchorHue); // Added anchorHue parameter

    // ADD: Missing setDstImg method
    void setDstImg(OFX::Image* p_DstImg);

private:
    // BOILERPLATE: Keep these basic members
    OFX::Image* _srcImg;
    OFX::Image* _dstImg; // ADD: Destination image member
    float _scales[4];
    
    // ADD: Color parameters (legacy support)
    float _originalR, _originalG, _originalB;
    float _targetR, _targetG, _targetB;
    int _choiceMode;
    
    // ADD: Multiple mapping support
    int _mappingCount;
    struct HueMapping {
        float srcHue;
        float srcRange;
        float targetHue;
        float targetRange;
        float strength;
        float bend;
        bool enabled;
        bool anchorHue; // NEW: Indicates if this mapping is an anchor
    } _mappings[MAX_COLOR_MAPPINGS];
};

////////////////////////////////////////////////////////////////////////////////
// IMAGE PROCESSOR IMPLEMENTATION
////////////////////////////////////////////////////////////////////////////////
// BOILERPLATE: Constructor
ImageProcessor::ImageProcessor(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance), _mappingCount(1)
{
    // Initialize mappings with defaults
    for (int i = 0; i < MAX_COLOR_MAPPINGS; i++) {
        _mappings[i] = {0.0f, 0.1f, 0.0f, 0.1f, 0.5f, 1.0f, false, false};
    }
    _mappings[0].enabled = true; // First mapping enabled by default
}

////////////////////////////////////////////////////////////////////////////////
// GPU PROCESSING METHODS (CUDA/OPENCL/METAL)
////////////////////////////////////////////////////////////////////////////////

#ifndef __APPLE__
extern void RunCudaKernel(void* p_Stream, int p_Width, int p_Height, 
                         const float* p_Input, float* p_Output);
#endif

#ifdef __APPLE__
extern "C" void RunMetalKernel(void* p_CmdQ, int p_Width, int p_Height,
                              float p_Range, float p_TargetHueRange, float p_PushStrength, float p_Bend,
                              float* p_OriginalColor, float* p_TargetColor,
                              const float* p_Input, float* p_Output);

// ADD: Multi-mapping Metal kernel
extern "C" void RunMultiMetalKernel(void* p_CmdQ, int p_Width, int p_Height,
                                   int p_MappingCount, float* p_MappingData,
                                   const float* p_Input, float* p_Output);
#endif

#ifndef __APPLE__
extern void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, 
                           const float* p_Input, float* p_Output);
#endif

////////////////////////////////////////////////////////////////////////////////
// PROCESSING METHODS
////////////////////////////////////////////////////////////////////////////////

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

    if (_mappingCount == 1) {
        // Legacy single mapping mode
        float originalColor[3] = {_originalR, _originalG, _originalB};
        float targetColor[3] = {_targetR, _targetG, _targetB};

        RunMetalKernel(_pMetalCmdQ, width, height,
                       _scales[0], _scales[1], _scales[2], _scales[3],
                       originalColor, targetColor,
                       input, output);
    } else {
        // Multi-mapping mode with anchor support
        float mappingData[MAX_COLOR_MAPPINGS * 8]; // 8 floats per mapping (added anchor)
        for (int i = 0; i < _mappingCount; i++) {
            int offset = i * 8;
            mappingData[offset + 0] = _mappings[i].srcHue;
            mappingData[offset + 1] = _mappings[i].srcRange;
            mappingData[offset + 2] = _mappings[i].targetHue;
            mappingData[offset + 3] = _mappings[i].targetRange;
            mappingData[offset + 4] = _mappings[i].strength;
            mappingData[offset + 5] = _mappings[i].bend;
            mappingData[offset + 6] = _mappings[i].enabled ? 1.0f : 0.0f;
            mappingData[offset + 7] = _mappings[i].anchorHue ? 1.0f : 0.0f; // NEW: Anchor flag
        }
        
        RunMultiMetalKernel(_pMetalCmdQ, width, height, _mappingCount, mappingData, input, output);
    }
#endif
}

void ImageProcessor::processImagesOpenCL()
{
#ifndef __APPLE__
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunOpenCLKernel(_pOpenCLCmdQ, width, height, input, output);
#else
    const OfxRectI& bounds = _srcImg->getBounds();
    multiThreadProcessImages(bounds);
#endif
}

void ImageProcessor::multiThreadProcessImages(OfxRectI p_ProcWindow)
{
    // CPU fallback - implement multi-mapping logic here
    for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y)
    {
        if (_effect.abort()) break;

        float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));

        for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x)
        {
            float* srcPix = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(x, y) : 0);

            if (srcPix)
            {
                // For now, simple passthrough - implement CPU hue mapping logic
                dstPix[0] = srcPix[0]; // Red
                dstPix[1] = srcPix[1]; // Green
                dstPix[2] = srcPix[2]; // Blue
                dstPix[3] = srcPix[3]; // Alpha
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

////////////////////////////////////////////////////////////////////////////////
// PARAMETER SETTER FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

void ImageProcessor::setSrcImg(OFX::Image* p_SrcImg)
{
    _srcImg = p_SrcImg;
}

// ADD: Missing setDstImg method
void ImageProcessor::setDstImg(OFX::Image* p_DstImg)
{
    _dstImg = p_DstImg;
}

void ImageProcessor::setScales(float p_Scale1, float p_Scale2, float p_Scale3, float p_Scale4)
{
    _scales[0] = p_Scale1;
    _scales[1] = p_Scale2;
    _scales[2] = p_Scale3;
    _scales[3] = p_Scale4;
}

void ImageProcessor::setColorParams(float origR, float origG, float origB, 
                                   float targR, float targG, float targB)
{
    _originalR = origR; _originalG = origG; _originalB = origB;
    _targetR = targR; _targetG = targG; _targetB = targB;
}

void ImageProcessor::setMappingCount(int count)
{
    _mappingCount = std::min(count, MAX_COLOR_MAPPINGS);
}

void ImageProcessor::setMappingParams(int index, float srcHue, float srcRange, float targetHue, 
                                     float targetRange, float strength, float bend, bool enabled, bool anchorHue)
{
    if (index >= 0 && index < MAX_COLOR_MAPPINGS) {
        _mappings[index].srcHue = srcHue;
        _mappings[index].srcRange = srcRange;
        _mappings[index].targetHue = targetHue;
        _mappings[index].targetRange = targetRange;
        _mappings[index].strength = strength;
        _mappings[index].bend = bend;
        _mappings[index].enabled = enabled;
        _mappings[index].anchorHue = anchorHue;  // NEW: Set anchor flag
    }
}

////////////////////////////////////////////////////////////////////////////////
// MAIN PLUGIN CLASS
////////////////////////////////////////////////////////////////////////////////

class HueWarp : public OFX::ImageEffect
{
public:
    explicit HueWarp(OfxImageEffectHandle p_Handle);

    virtual void render(const OFX::RenderArguments& p_Args);
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);
    virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);
    virtual void changedClip(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ClipName);

    void setEnabledness();
    void setupAndProcess(ImageProcessor &p_Processor, const OFX::RenderArguments& p_Args);

private:
    // Helper function to convert RGB to hue
    float rgbToHue(double r, double g, double b) {
        float cmax = std::max({(float)r, (float)g, (float)b});
        float cmin = std::min({(float)r, (float)g, (float)b});
        float delta = cmax - cmin;
        
        if (delta == 0.0f) return 0.0f;
        
        float hue;
        if (cmax == r) {
            hue = std::fmod((g - b) / delta + 6.0f, 6.0f) / 6.0f;
        } else if (cmax == g) {
            hue = ((b - r) / delta + 2.0f) / 6.0f;
        } else {
            hue = ((r - g) / delta + 4.0f) / 6.0f;
        }
        
        return hue;
    }

    // Clips
    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;

    // Legacy parameters
    OFX::DoubleParam* m_Scale;
    OFX::DoubleParam* m_Range;
    OFX::DoubleParam* m_TargetHueRange;
    OFX::DoubleParam* m_PushStrength;
    OFX::DoubleParam* m_Bend;
    OFX::BooleanParam* m_BoolParam;
    OFX::ChoiceParam* m_ChoiceParam;
    OFX::RGBParam* m_OriginalColor;
    OFX::RGBParam* m_TargetColor;
    
    // ADD: Fetch the main group parameter
    OFX::GroupParam* m_MainGroup;
    
    // Multiple mapping parameters
    OFX::IntParam* m_MappingCount;
    
    // Arrays for multiple mappings - using RGB color pickers
    OFX::RGBParam* m_SrcColor[MAX_COLOR_MAPPINGS];
    OFX::DoubleParam* m_SrcRange[MAX_COLOR_MAPPINGS];
    OFX::RGBParam* m_TargetColor_Multi[MAX_COLOR_MAPPINGS];
    OFX::DoubleParam* m_TargetRange[MAX_COLOR_MAPPINGS];
    OFX::DoubleParam* m_Strength[MAX_COLOR_MAPPINGS];
    OFX::DoubleParam* m_MappingBend[MAX_COLOR_MAPPINGS];
    OFX::BooleanParam* m_Enabled[MAX_COLOR_MAPPINGS];
    OFX::BooleanParam* m_Anchor[MAX_COLOR_MAPPINGS];      // NEW: Anchor parameters
    OFX::GroupParam* m_MappingGroups[MAX_COLOR_MAPPINGS];    // Store group references

    // Buy Me a Coffee button
    OFX::PushButtonParam* m_CoffeeButton;
};

////////////////////////////////////////////////////////////////////////////////
// PLUGIN CONSTRUCTOR
////////////////////////////////////////////////////////////////////////////////

HueWarp::HueWarp(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

    // Legacy parameters
    m_Scale = fetchDoubleParam("scale");
    m_Range = fetchDoubleParam("range");
    m_TargetHueRange = fetchDoubleParam("param2");
    m_PushStrength = fetchDoubleParam("param3");
    m_Bend = fetchDoubleParam("param4");
    m_BoolParam = fetchBooleanParam("boolparam");
    m_ChoiceParam = fetchChoiceParam("choiceparam");
    m_OriginalColor = fetchRGBParam("originalcolor");
    m_TargetColor = fetchRGBParam("targetcolor");
    
    // ADD: Fetch the main group parameter
    m_MainGroup = fetchGroupParam("MainGroup");
    
    // Multiple mapping parameters
    m_MappingCount = fetchIntParam("mappingcount");
    
    for (int i = 0; i < MAX_COLOR_MAPPINGS; i++) {
        std::string suffix = std::to_string(i);
        m_SrcColor[i] = fetchRGBParam("srccolor_" + suffix);
        m_SrcRange[i] = fetchDoubleParam("srcrange_" + suffix);
        m_TargetColor_Multi[i] = fetchRGBParam("targetcolor_" + suffix);
        m_TargetRange[i] = fetchDoubleParam("targetrange_" + suffix);
        m_Strength[i] = fetchDoubleParam("strength_" + suffix);
        m_MappingBend[i] = fetchDoubleParam("mappingbend_" + suffix);
        m_Enabled[i] = fetchBooleanParam("enabled_" + suffix);
        m_Anchor[i] = fetchBooleanParam("anchor_" + suffix);  // NEW: Fetch anchor parameters
        m_MappingGroups[i] = fetchGroupParam("mapping_group_" + suffix);
    }

    // Buy Me a Coffee button
    m_CoffeeButton = fetchPushButtonParam("buymeacoffee");
    
    setEnabledness();
}

////////////////////////////////////////////////////////////////////////////////
// RENDER AND PROCESSING
////////////////////////////////////////////////////////////////////////////////

void HueWarp::render(const OFX::RenderArguments& p_Args)
{
    if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && 
        (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
    {
        ImageProcessor processor(*this);
        setupAndProcess(processor, p_Args);
    }
    else
    {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

bool HueWarp::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
    double scale = m_Scale->getValueAtTime(p_Args.time);
    double pushStrength = m_PushStrength->getValueAtTime(p_Args.time);
    
    if (scale == 1.0 && pushStrength == 0.0)
    {
        p_IdentityClip = m_SrcClip;
        p_IdentityTime = p_Args.time;
        return true;
    }
    return false;
}

void HueWarp::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
    if (p_ParamName == "boolparam" || p_ParamName == "mappingcount")
    {
        setEnabledness();
    }
    // NEW: Handle anchor parameter changes
    else if (p_ParamName.find("anchor_") == 0)
    {
        // Extract the mapping index from parameter name
        std::string suffix = p_ParamName.substr(7); // Remove "anchor_" prefix
        int index = std::stoi(suffix);
        
        if (index >= 0 && index < MAX_COLOR_MAPPINGS) {
            bool isAnchored = m_Anchor[index]->getValue();
            
            if (isAnchored) {
                // Gang target color to match source color
                double srcR, srcG, srcB;
                m_SrcColor[index]->getValueAtTime(p_Args.time, srcR, srcG, srcB);
                m_TargetColor_Multi[index]->setValue(srcR, srcG, srcB);
            }
            
            // FIXED: Always update UI enablement regardless of anchor state
            m_TargetColor_Multi[index]->setEnabled(!isAnchored);
        }
    }
    // NEW: Handle source color changes for anchored mappings
    else if (p_ParamName.find("srccolor_") == 0)
    {
        std::string suffix = p_ParamName.substr(9); // Remove "srccolor_" prefix
        int index = std::stoi(suffix);
        
        if (index >= 0 && index < MAX_COLOR_MAPPINGS) {
            bool isAnchored = m_Anchor[index]->getValue();
            
            if (isAnchored) {
                // Auto-update target color to match source color
                double srcR, srcG, srcB;
                m_SrcColor[index]->getValueAtTime(p_Args.time, srcR, srcG, srcB);
                m_TargetColor_Multi[index]->setValue(srcR, srcG, srcB);
            }
        }
    }
    else if (p_ParamName == "buymeacoffee")
    {
        // Open Buy Me a Coffee link
        std::string url = "https://dec18studios.com/coffee";
        
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

void HueWarp::changedClip(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ClipName)
{
    if (p_ClipName == kOfxImageEffectSimpleSourceClipName)
    {
        setEnabledness();
    }
}

void HueWarp::setEnabledness()
{
    bool boolValue = m_BoolParam->getValue();
    int mappingCount = m_MappingCount->getValue();
    
    // Hide main controls when multi-mapping has more than 1 mapping
    bool hideMainControls = (mappingCount > 1);
    
    // Control main group visibility
    m_MainGroup->setIsSecret(hideMainControls);
    
    if (!hideMainControls) {
        // Enable/disable main controls based on Enable Effect toggle
        m_OriginalColor->setEnabled(boolValue);
        m_Range->setEnabled(boolValue);
        m_TargetColor->setEnabled(boolValue);
        m_TargetHueRange->setEnabled(boolValue);
        m_PushStrength->setEnabled(boolValue);
        m_Bend->setEnabled(boolValue);
    }
    
    // Show/hide mapping parameters based on count using setIsSecret
    for (int i = 0; i < MAX_COLOR_MAPPINGS; i++) {
        bool visible = (i < mappingCount);
        
        // Hide/show the entire group using setIsSecret
        m_MappingGroups[i]->setIsSecret(!visible);
        
        // Also hide/show individual parameters for consistency
        m_SrcColor[i]->setIsSecret(!visible);
        m_SrcRange[i]->setIsSecret(!visible);
        m_TargetColor_Multi[i]->setIsSecret(!visible);
        m_TargetRange[i]->setIsSecret(!visible);
        m_Strength[i]->setIsSecret(!visible);
        m_MappingBend[i]->setIsSecret(!visible);
        m_Enabled[i]->setIsSecret(!visible);
        m_Anchor[i]->setIsSecret(!visible);
        
        // FIXED: Handle anchor logic properly - only for visible parameters
        if (visible) {
            // Check if parameter exists before calling getValue()
            try {
                bool isAnchored = m_Anchor[i]->getValue();
                m_TargetColor_Multi[i]->setEnabled(!isAnchored);
            } catch (...) {
                // If there's an issue getting the value, default to enabled
                m_TargetColor_Multi[i]->setEnabled(true);
            }
        }
    }
}

void HueWarp::setupAndProcess(ImageProcessor& p_Processor, const OFX::RenderArguments& p_Args)
{
    // Get images
    std::unique_ptr<OFX::Image> dst(m_DstClip->fetchImage(p_Args.time));
    std::unique_ptr<OFX::Image> src(m_SrcClip->fetchImage(p_Args.time));

    // Setup processor with BOTH source and destination images
    p_Processor.setDstImg(dst.get());
    p_Processor.setSrcImg(src.get());
    p_Processor.setGPURenderArgs(p_Args);
    p_Processor.setRenderWindow(p_Args.renderWindow);

    // Legacy parameters
    double scale = m_Scale->getValueAtTime(p_Args.time);
    double range = m_Range->getValueAtTime(p_Args.time);
    double targetHueRange = m_TargetHueRange->getValueAtTime(p_Args.time);
    double pushStrength = m_PushStrength->getValueAtTime(p_Args.time);
    double bend = m_Bend->getValueAtTime(p_Args.time);

    double originalR, originalG, originalB;
    m_OriginalColor->getValueAtTime(p_Args.time, originalR, originalG, originalB);

    double targetR, targetG, targetB;
    m_TargetColor->getValueAtTime(p_Args.time, targetR, targetG, targetB);

    // Multiple mapping parameters
    int mappingCount = m_MappingCount->getValueAtTime(p_Args.time);

    // Pass legacy parameters
    p_Processor.setScales(range, targetHueRange, pushStrength, bend);
    p_Processor.setColorParams(originalR, originalG, originalB, 
                              targetR, targetG, targetB);

    // Pass multiple mapping parameters
    p_Processor.setMappingCount(mappingCount);
    for (int i = 0; i < mappingCount; i++) {
        // Get RGB colors from color pickers
        double srcR, srcG, srcB;
        m_SrcColor[i]->getValueAtTime(p_Args.time, srcR, srcG, srcB);
        
        double targetR_multi, targetG_multi, targetB_multi;
        m_TargetColor_Multi[i]->getValueAtTime(p_Args.time, targetR_multi, targetG_multi, targetB_multi);
        
        // Convert RGB to HSV to extract hue values
        float srcHue = rgbToHue(srcR, srcG, srcB);
        float targetHue = rgbToHue(targetR_multi, targetG_multi, targetB_multi);
        
        double srcRange = m_SrcRange[i]->getValueAtTime(p_Args.time);
        double targetRange = m_TargetRange[i]->getValueAtTime(p_Args.time);
        double strength = m_Strength[i]->getValueAtTime(p_Args.time);
        double mappingBend = m_MappingBend[i]->getValueAtTime(p_Args.time);
        bool enabled = m_Enabled[i]->getValueAtTime(p_Args.time);
        bool anchorHue = m_Anchor[i]->getValueAtTime(p_Args.time);  // NEW: Get anchor state
        
        p_Processor.setMappingParams(i, srcHue, srcRange, targetHue, targetRange, strength, mappingBend, enabled, anchorHue);
    }

    p_Processor.process();
}

////////////////////////////////////////////////////////////////////////////////
// PLUGIN FACTORY
////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

HueWarpFactory::HueWarpFactory()
    : OFX::PluginFactoryHelper<HueWarpFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void HueWarpFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
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

    p_Desc.setSupportsOpenCLRender(true);
#ifndef __APPLE__
    p_Desc.setSupportsCudaRender(true);
    p_Desc.setSupportsCudaStream(true);
#endif
#ifdef __APPLE__
    p_Desc.setSupportsMetalRender(true);
#endif

    p_Desc.setNoSpatialAwareness(true);
}

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

void HueWarpFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
{
    // Clips
    ClipDescriptor* srcClip = p_Desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    srcClip->addSupportedComponent(ePixelComponentRGBA);
    srcClip->setTemporalClipAccess(false);
    srcClip->setSupportsTiles(kSupportsTiles);
    srcClip->setIsMask(false);

    ClipDescriptor* dstClip = p_Desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(ePixelComponentRGBA);
    dstClip->addSupportedComponent(ePixelComponentAlpha);
    dstClip->setSupportsTiles(kSupportsTiles);

    // Page and groups
    PageParamDescriptor* page = p_Desc.definePageParam("Controls");
    
    GroupParamDescriptor* mainGroup = p_Desc.defineGroupParam("MainGroup");
    mainGroup->setHint("Single hue mapping controls");
    mainGroup->setLabels("Single Mapping", "Single Mapping", "Single Mapping");
    mainGroup->setOpen(true);  // Start open
    page->addChild(*mainGroup);

    // Enable Effect toggle - controls visibility of all main controls
    BooleanParamDescriptor* boolParam = p_Desc.defineBooleanParam("boolparam");
    boolParam->setDefault(true);  // Default to enabled
    boolParam->setHint("Enable single hue mapping");
    boolParam->setLabels("Enable Effect", "Enable Effect", "Enable Effect");
    boolParam->setParent(*mainGroup);
    page->addChild(*boolParam);

    // Source Color (reorganized to match multi-mapping layout)
    RGBParamDescriptor* colorParam1 = p_Desc.defineRGBParam("originalcolor");
    colorParam1->setLabel("Source Color");
    colorParam1->setHint("Choose the source color for hue warping");
    colorParam1->setDefault(1.0, 0.0, 0.0);
    colorParam1->setAnimates(true);
    colorParam1->setParent(*mainGroup);
    page->addChild(*colorParam1);

    // Source Range
    DoubleParamDescriptor* param = defineDoubleParam(p_Desc, "range", "Source Range", "Range around source hue", 
                             mainGroup, 0.1, 0.0, 1.0, 0.01);
    page->addChild(*param);

    // Target Color
    RGBParamDescriptor* colorParam2 = p_Desc.defineRGBParam("targetcolor");
    colorParam2->setLabel("Target Color");
    colorParam2->setHint("Choose the target color for hue warping");
    colorParam2->setDefault(0.0, 1.0, 0.0);
    colorParam2->setAnimates(true);
    colorParam2->setParent(*mainGroup);
    page->addChild(*colorParam2);

    // Target Range
    param = defineDoubleParam(p_Desc, "param2", "Target Range", "Range around target hue", 
                             mainGroup, 0.1, 0.0, 1.0, 0.01);
    page->addChild(*param);

    // Strength
    param = defineDoubleParam(p_Desc, "param3", "Strength", "Mapping strength", 
                             mainGroup, 0.5, 0.0, 2.0, 0.1);
    page->addChild(*param);

    // Bend
    param = defineDoubleParam(p_Desc, "param4", "Bend", "Mapping curve bend", 
                             mainGroup, 1.0, 0.1, 3.0, 0.1);
    page->addChild(*param);

    // HIDDEN: Scale parameter (kept for legacy compatibility)
    DoubleParamDescriptor* scaleParam = defineDoubleParam(p_Desc, "scale", "Scale", "Overall scale factor", 
                                                         nullptr, 1.0, 0.0, 10.0, 0.1);
    scaleParam->setIsSecret(true);  // Hide scale parameter
    page->addChild(*scaleParam);

    // HIDDEN: Mode parameter (kept for legacy compatibility)
    ChoiceParamDescriptor* choiceParam = p_Desc.defineChoiceParam("choiceparam");
    choiceParam->setLabel("Mode");
    choiceParam->setHint("Choose processing mode");
    choiceParam->appendOption("Mode 1");
    choiceParam->appendOption("Mode 2");
    choiceParam->appendOption("Mode 3");
    choiceParam->setDefault(0);
    choiceParam->setAnimates(true);
    choiceParam->setIsSecret(true);  // Hide mode parameter
    page->addChild(*choiceParam);

    // ADD: Multiple mapping controls
    GroupParamDescriptor* multiGroup = p_Desc.defineGroupParam("MultiMappingGroup");
    multiGroup->setHint("Multiple hue mapping controls");
    multiGroup->setLabels("Multi-Mapping", "Multi-Mapping", "Multi-Mapping");
    multiGroup->setOpen(true);  // Start with group closed
    page->addChild(*multiGroup);

    // Mapping count control
    IntParamDescriptor* mappingCountParam = p_Desc.defineIntParam("mappingcount");
    mappingCountParam->setLabel("Mapping Count");
    mappingCountParam->setHint("Number of active hue mappings");
    mappingCountParam->setDefault(2);  // CHANGED: Default to 2 mappings
    mappingCountParam->setRange(1, MAX_COLOR_MAPPINGS);
    mappingCountParam->setDisplayRange(1, MAX_COLOR_MAPPINGS);
    mappingCountParam->setParent(*multiGroup);
    page->addChild(*mappingCountParam);

    // Create ALL parameters for each mapping (they'll be hidden/shown dynamically)
    for (int i = 0; i < MAX_COLOR_MAPPINGS; i++) {
        std::string suffix = std::to_string(i);
        std::string groupName = "Mapping " + std::to_string(i + 1);
        
        GroupParamDescriptor* mappingGroup = p_Desc.defineGroupParam("mapping_group_" + suffix);
        mappingGroup->setHint("Hue mapping " + std::to_string(i + 1) + " parameters");
        mappingGroup->setLabels(groupName, groupName, groupName);
        mappingGroup->setParent(*multiGroup);
        // Initially hide all groups except the first TWO
        mappingGroup->setIsSecret(i >= 2); // CHANGED: Show first two groups
        page->addChild(*mappingGroup);

        // Enabled toggle - ENABLE FIRST TWO BY DEFAULT
        BooleanParamDescriptor* enabledParam = p_Desc.defineBooleanParam("enabled_" + suffix);
        enabledParam->setLabel("Enabled");
        enabledParam->setHint("Enable this hue mapping");
        enabledParam->setDefault(i < 2); // CHANGED: First TWO mappings enabled by default
        enabledParam->setParent(*mappingGroup);
        enabledParam->setIsSecret(i >= 2); // CHANGED: Initially hidden except first two
        page->addChild(*enabledParam);

        // NEW: Anchor Hue toggle
        BooleanParamDescriptor* anchorParam = p_Desc.defineBooleanParam("anchor_" + suffix);
        anchorParam->setLabel("⚓ Anchor");
        anchorParam->setHint("Lock this source hue and resist changes from other mappings. Strength controls resistance level.");
        anchorParam->setDefault(false);
        anchorParam->setParent(*mappingGroup);
        anchorParam->setIsSecret(i >= 2); // CHANGED: Show first two
        page->addChild(*anchorParam);

        // Source Color Picker
        RGBParamDescriptor* srcColorParam = p_Desc.defineRGBParam("srccolor_" + suffix);
        srcColorParam->setLabel("Source Color");
        srcColorParam->setHint("Pick the source color to transform");
        // Set different default colors for each mapping for visual distinction
        switch(i % 6) {
            case 0: srcColorParam->setDefault(1.0, 0.0, 0.0); break; // Red
            case 1: srcColorParam->setDefault(1.0, 0.5, 0.0); break; // Orange
            case 2: srcColorParam->setDefault(1.0, 1.0, 0.0); break; // Yellow
            case 3: srcColorParam->setDefault(0.0, 1.0, 0.0); break; // Green
            case 4: srcColorParam->setDefault(0.0, 0.0, 1.0); break; // Blue
            case 5: srcColorParam->setDefault(0.5, 0.0, 1.0); break; // Purple
        }
        srcColorParam->setAnimates(true);
        srcColorParam->setParent(*mappingGroup);
        srcColorParam->setIsSecret(i >= 2);  // CHANGED
        page->addChild(*srcColorParam);

        // Source range
        DoubleParamDescriptor* srcRangeParam = defineDoubleParam(p_Desc, "srcrange_" + suffix, 
                                                                "Source Range", "Range around source hue", 
                                                                mappingGroup, 0.1, 0.0, 1.0, 0.01);
        srcRangeParam->setIsSecret(i >= 2);  // CHANGED
        page->addChild(*srcRangeParam);

        // Target Color Picker
        RGBParamDescriptor* targetColorParam = p_Desc.defineRGBParam("targetcolor_" + suffix);
        targetColorParam->setLabel("Target Color");
        targetColorParam->setHint("Pick the target color to map to");
        // Set different default target colors
        switch(i % 6) {
            case 0: targetColorParam->setDefault(0.0, 1.0, 0.0); break; // Green
            case 1: targetColorParam->setDefault(0.0, 0.0, 1.0); break; // Blue
            case 2: targetColorParam->setDefault(0.5, 0.0, 1.0); break; // Purple
            case 3: targetColorParam->setDefault(1.0, 0.0, 0.0); break; // Red
            case 4: targetColorParam->setDefault(1.0, 0.5, 0.0); break; // Orange
            case 5: targetColorParam->setDefault(1.0, 1.0, 0.0); break; // Yellow
        }
        targetColorParam->setAnimates(true);
        targetColorParam->setParent(*mappingGroup);
        targetColorParam->setIsSecret(i >= 2);  // CHANGED
        page->addChild(*targetColorParam);

        // Target range
        DoubleParamDescriptor* targetRangeParam = defineDoubleParam(p_Desc, "targetrange_" + suffix, 
                                                                   "Target Range", "Range around target hue", 
                                                                   mappingGroup, 0.1, 0.0, 1.0, 0.01);
        targetRangeParam->setIsSecret(i >= 2);  // CHANGED
        page->addChild(*targetRangeParam);

        // Strength
        DoubleParamDescriptor* strengthParam = defineDoubleParam(p_Desc, "strength_" + suffix, 
                                                                "Strength", "Mapping strength", 
                                                                mappingGroup, 0.5, 0.0, 2.0, 0.1);
        strengthParam->setIsSecret(i >= 2);  // CHANGED
        page->addChild(*strengthParam);

        // Bend
        DoubleParamDescriptor* bendParam = defineDoubleParam(p_Desc, "mappingbend_" + suffix, 
                                                            "Bend", "Mapping curve bend", 
                                                            mappingGroup, 1.0, 0.1, 3.0, 0.1);
        bendParam->setIsSecret(i >= 2);  // CHANGED
        page->addChild(*bendParam);
    }

    // ADD: Buy Me a Coffee button at the end
    PushButtonParamDescriptor* coffeeButton = p_Desc.definePushButtonParam("buymeacoffee");
    coffeeButton->setLabel("☕ Buy Me a Coffee");
    coffeeButton->setHint("Support the developer - opens external link to Buy Me a Coffee");
    page->addChild(*coffeeButton);
}

ImageEffect* HueWarpFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new HueWarp(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static HueWarpFactory HueWarp;
    p_FactoryArray.push_back(&HueWarp);
}
