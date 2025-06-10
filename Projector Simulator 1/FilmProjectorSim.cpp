#include "FilmProjector.h"

#include <stdio.h>

#include "ofxsImageEffect.h"
#include "ofxsInteract.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"
#include "ofxsLog.h"
#include "ofxDrawSuite.h"
#include "ofxsSupportPrivate.h"

// ADD THESE EXTERN DECLARATIONS AT THE TOP
#ifndef __APPLE__
extern void RunCudaKernel(void* p_Stream, int p_Width, int p_Height, float* p_Gain, const float* p_Input, float* p_Output);
extern void RunProjectorCudaKernel(void* p_Stream, int p_Width, int p_Height, 
                                  const float* negativePreset, const float* printPreset,
                                  const float* p_Input, float* p_Output);
#endif

#ifdef __APPLE__
extern void RunMetalKernel(void* p_CmdQ, int p_Width, int p_Height, float* p_Gain, const float* p_Input, float* p_Output);
// ADD THIS LINE:
extern void RunProjectorMetalKernel(void* p_CmdQ, int p_Width, int p_Height,
                                   const float* negativePreset, const float* printPreset,
                                   bool alphaPassThru, float alphaMin, float alphaMax, float linearAdjustment,
                                   const float* p_Input, float* p_Output);
#endif

extern void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float* p_Gain, const float* p_Input, float* p_Output);

#define kPluginName "Film Projector Sim"
#define kPluginGrouping "Greg Enright"
#define kPluginDescription "Make a Fusion Film Projector Sim"
#define kPluginIdentifier "com.OpenFXSample.FilmProjector"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 0

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

// ADD THESE CONSTANTS
#define RED_ONLY 0
#define GREEN_ONLY 1
#define BLUE_ONLY 2

////////////////////////////////////////////////////////////////////////////////

class ImagePrintMean : public OFX::ImageProcessor
{
public:
    explicit ImagePrintMean(OFX::ImageEffect& p_Instance);

    virtual void processImagesCUDA();
    virtual void processImagesOpenCL();
    virtual void processImagesMetal();
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

    void setSrcImg(OFX::Image* p_SrcImg);
    void setScales(float p_PrintMean, float p_printleftstd, float p_printrightstd, float p_ScaleA);
    
    // Add this method to set all parameters and build arrays
    void setProjectorParams(
        bool useCustomNegative, bool useCustomPrint, int negativeStock, int printStock, int layerMode,
        float negMean, float negLeft, float negRight, float negMax, float negMin,
        float sNegMean, float sNegLeft, float sNegRight, float sNegMax, float sNegMin, float sNegRatio,
        float printMean, float printLeft, float printRight, float printMax, float printMin,
        float sPrintMean, float sPrintLeft, float sPrintRight, float sPrintMax, float sPrintMin, float sPrintRatio,
        bool alphaPassThru, float alphaMin, float alphaMax, float linearAdjustment // ADD THESE
    );

private:
    OFX::Image* _srcImg;
    float _scales[4];
    
    // Store all parameters
    bool _useCustomNegative, _useCustomPrint;
    int _negativeStock, _printStock, _layerMode;
    float _negMean, _negLeft, _negRight, _negMax, _negMin;
    float _sNegMean, _sNegLeft, _sNegRight, _sNegMax, _sNegMin, _sNegRatio;
    float _printMean, _printLeft, _printRight, _printMax, _printMin;
    float _sPrintMean, _sPrintLeft, _sPrintRight, _sPrintMax, _sPrintMin, _sPrintRatio;
    
    // ADD THESE ALPHA PARAMETERS
    bool _alphaPassThru;
    float _alphaMin;
    float _alphaMax; 
    float _linearAdjustment;
    
    // Pre-built arrays - created once, used by all GPU methods
    float _negativePreset[22];
    float _printPreset[22];
    
    // Method to build both arrays
    void buildPresetArrays();
};

ImagePrintMean::ImagePrintMean(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance)
{
}

#ifndef __APPLE__
extern void RunCudaKernel(void* p_Stream, int p_Width, int p_Height, float* p_Gain, const float* p_Input, float* p_Output);
#endif

// Implementation of setProjectorParams
void ImagePrintMean::setProjectorParams(
    bool useCustomNegative, bool useCustomPrint, int negativeStock, int printStock, int layerMode,
    float negMean, float negLeft, float negRight, float negMax, float negMin,
    float sNegMean, float sNegLeft, float sNegRight, float sNegMax, float sNegMin, float sNegRatio,
    float printMean, float printLeft, float printRight, float printMax, float printMin,
    float sPrintMean, float sPrintLeft, float sPrintRight, float sPrintMax, float sPrintMin, float sPrintRatio,
    bool alphaPassThru, float alphaMin, float alphaMax, float linearAdjustment)  // ADD THESE
{
    // Store all parameters
    _useCustomNegative = useCustomNegative;
    _useCustomPrint = useCustomPrint;
    _negativeStock = negativeStock;
    _printStock = printStock;
    _layerMode = layerMode;
    
    _negMean = negMean;
    _negLeft = negLeft;
    _negRight = negRight;
    _negMax = negMax;
    _negMin = negMin;
    
    _sNegMean = sNegMean;
    _sNegLeft = sNegLeft;
    _sNegRight = sNegRight;
    _sNegMax = sNegMax;
    _sNegMin = sNegMin;
    _sNegRatio = sNegRatio;
    
    _printMean = printMean;
    _printLeft = printLeft;
    _printRight = printRight;
    _printMax = printMax;
    _printMin = printMin;
    
    _sPrintMean = sPrintMean;
    _sPrintLeft = sPrintLeft;
    _sPrintRight = sPrintRight;
    _sPrintMax = sPrintMax;
    _sPrintMin = sPrintMin;
    _sPrintRatio = sPrintRatio;
    
    // ADD THESE LINES
    _alphaPassThru = alphaPassThru;
    _alphaMin = alphaMin;
    _alphaMax = alphaMax;
    _linearAdjustment = linearAdjustment;
    
    // Build the arrays once
    buildPresetArrays();
}

// Method to build both preset arrays
void ImagePrintMean::buildPresetArrays()
{
    // Build negative array
    if (_useCustomNegative) {
        // Use inline conditionals to place custom values in correct layer position
        _negativePreset[0] = (_layerMode == RED_ONLY) ? _negMean : 650.0f;     // cyan_mean
        _negativePreset[1] = (_layerMode == GREEN_ONLY) ? _negMean : 525.0f;   // magenta_mean  
        _negativePreset[2] = (_layerMode == BLUE_ONLY) ? _negMean : 476.4f;    // yellow_mean
        _negativePreset[3] = _sNegMean;                                         // silver_mean
        
        _negativePreset[4] = (_layerMode == RED_ONLY) ? _negLeft : 48.1f;      // cyan_left_std
        _negativePreset[5] = (_layerMode == GREEN_ONLY) ? _negLeft : 35.9f;    // magenta_left_std
        _negativePreset[6] = (_layerMode == BLUE_ONLY) ? _negLeft : 26.9f;     // yellow_left_std
        _negativePreset[7] = _sNegLeft;                                         // silver_left_std
        
        _negativePreset[8] = (_layerMode == RED_ONLY) ? _negRight : 40.5f;     // cyan_right_std
        _negativePreset[9] = (_layerMode == GREEN_ONLY) ? _negRight : 28.4f;   // magenta_right_std
        _negativePreset[10] = (_layerMode == BLUE_ONLY) ? _negRight : 26.9f;   // yellow_right_std
        _negativePreset[11] = _sNegRight;                                       // silver_right_std
        
        _negativePreset[12] = (_layerMode == RED_ONLY) ? _negMax : 0.0f;       // cyan_max_value
        _negativePreset[13] = (_layerMode == GREEN_ONLY) ? _negMax : 0.0f;     // magenta_max_value
        _negativePreset[14] = (_layerMode == BLUE_ONLY) ? _negMax : 0.0f;      // yellow_max_value
        _negativePreset[15] = _sNegMax;                                         // silver_max_value
        
        _negativePreset[16] = (_layerMode == RED_ONLY) ? _negMin : 0.0f;       // cyan_min_value
        _negativePreset[17] = (_layerMode == GREEN_ONLY) ? _negMin : 0.0f;     // magenta_min_value
        _negativePreset[18] = (_layerMode == BLUE_ONLY) ? _negMin : 0.0f;      // yellow_min_value
        _negativePreset[19] = _sNegMin;                                         // silver_min_value
        
        _negativePreset[20] = _sNegRatio;                                       // silver_ratio
        _negativePreset[21] = 1.0f;                                             // input_gain
    } else {
        // Use film stock presets from dropdown selection
        if (_negativeStock == 0) { // NEG_5247 (Kodak 5247)
            if (_layerMode == RED_ONLY) {
                float preset[22] = {
                    650.0f, 525.0f, 476.4f, 511.5f,    // means
                    48.1f, 35.9f, 26.9f, 110.6f,       // left stds
                    40.5f, 28.4f, 26.9f, 19.2f,        // right stds
                    0.95f, 0.0f, 0.0f, 1.575f,         // max values
                    0.102f, 0.0f, 0.0f, 0.126f,        // min values
                    0.228f, 1.0f                        // silver ratio, input gain
                };
                for (int i = 0; i < 22; i++) _negativePreset[i] = preset[i];
            } else if (_layerMode == GREEN_ONLY) {
                float preset[22] = {
                    650.0f, 525.0f, 476.4f, 511.5f,    // means
                    48.1f, 35.9f, 26.9f, 110.6f,       // left stds
                    40.5f, 28.4f, 26.9f, 19.2f,        // right stds
                    0.0f, 0.95f, 0.0f, 1.481f,         // max values
                    0.0f, 0.083f, 0.0f, 0.084f,        // min values
                    0.007f, 1.0f                        // silver ratio, input gain
                };
                for (int i = 0; i < 22; i++) _negativePreset[i] = preset[i];
            } else if (_layerMode == BLUE_ONLY) {
                float preset[22] = {
                    650.0f, 525.0f, 476.4f, 511.5f,    // means
                    48.1f, 35.9f, 26.9f, 110.6f,       // left stds
                    40.5f, 28.4f, 26.9f, 19.2f,        // right stds
                    0.0f, 0.0f, 0.95f, 1.481f,         // max values
                    0.0f, 0.0f, 0.083f, 0.084f,        // min values
                    0.007f, 1.0f                        // silver ratio, input gain
                };
                for (int i = 0; i < 22; i++) _negativePreset[i] = preset[i];
            }
        } else if (_negativeStock == 1) { // NEG_5213 (Kodak 5213)
            if (_layerMode == RED_ONLY) {
                float preset[22] = {
                    650.0f, 550.0f, 450.0f, 510.0f,    // means
                    50.0f, 40.0f, 30.0f, 120.0f,       // left stds
                    45.0f, 35.0f, 25.0f, 20.0f,        // right stds
                    1.4f, 0.0f, 0.0f, 1.5f,            // max values
                    0.1f, 0.0f, 0.0f, 0.1f,            // min values
                    0.25f, 1.0f                         // silver ratio, input gain
                };
                for (int i = 0; i < 22; i++) _negativePreset[i] = preset[i];
            } else if (_layerMode == GREEN_ONLY) {
                float preset[22] = {
                    650.0f, 550.0f, 450.0f, 510.0f,    // means
                    50.0f, 40.0f, 30.0f, 120.0f,       // left stds
                    45.0f, 35.0f, 25.0f, 20.0f,        // right stds
                    0.0f, 1.2f, 0.0f, 1.4f,            // max values
                    0.0f, 0.1f, 0.0f, 0.1f,            // min values
                    0.20f, 1.0f                         // silver ratio, input gain
                };
                for (int i = 0; i < 22; i++) _negativePreset[i] = preset[i];
            } else if (_layerMode == BLUE_ONLY) {
                float preset[22] = {
                    650.0f, 550.0f, 450.0f, 510.0f,    // means
                    50.0f, 40.0f, 30.0f, 120.0f,       // left stds
                    45.0f, 35.0f, 25.0f, 20.0f,        // right stds
                    0.0f, 0.0f, 1.6f, 1.3f,            // max values
                    0.0f, 0.0f, 0.1f, 0.1f,            // min values
                    0.15f, 1.0f                         // silver ratio, input gain
                };
                for (int i = 0; i < 22; i++) _negativePreset[i] = preset[i];
            }
        }
    }
    
    // Build print array
    if (_useCustomPrint) {
        // Use inline conditionals to place custom values in correct layer position
        _printPreset[0] = (_layerMode == RED_ONLY) ? _printMean : 650.0f;      // cyan_mean
        _printPreset[1] = (_layerMode == GREEN_ONLY) ? _printMean : 525.0f;    // magenta_mean
        _printPreset[2] = (_layerMode == BLUE_ONLY) ? _printMean : 476.4f;     // yellow_mean
        _printPreset[3] = _sPrintMean;                                          // silver_mean
        
        _printPreset[4] = (_layerMode == RED_ONLY) ? _printLeft : 48.1f;       // cyan_left_std
        _printPreset[5] = (_layerMode == GREEN_ONLY) ? _printLeft : 35.9f;     // magenta_left_std
        _printPreset[6] = (_layerMode == BLUE_ONLY) ? _printLeft : 26.9f;      // yellow_left_std
        _printPreset[7] = _sPrintLeft;                                          // silver_left_std
        
        _printPreset[8] = (_layerMode == RED_ONLY) ? _printRight : 40.5f;      // cyan_right_std
        _printPreset[9] = (_layerMode == GREEN_ONLY) ? _printRight : 28.4f;    // magenta_right_std
        _printPreset[10] = (_layerMode == BLUE_ONLY) ? _printRight : 26.9f;    // yellow_right_std
        _printPreset[11] = _sPrintRight;                                        // silver_right_std
        
        _printPreset[12] = (_layerMode == RED_ONLY) ? _printMax : 0.0f;        // cyan_max_value
        _printPreset[13] = (_layerMode == GREEN_ONLY) ? _printMax : 0.0f;      // magenta_max_value
        _printPreset[14] = (_layerMode == BLUE_ONLY) ? _printMax : 0.0f;       // yellow_max_value
        _printPreset[15] = _sPrintMax;                                          // silver_max_value
        
        _printPreset[16] = (_layerMode == RED_ONLY) ? _printMin : 0.0f;        // cyan_min_value
        _printPreset[17] = (_layerMode == GREEN_ONLY) ? _printMin : 0.0f;      // magenta_min_value
        _printPreset[18] = (_layerMode == BLUE_ONLY) ? _printMin : 0.0f;       // yellow_min_value
        _printPreset[19] = _sPrintMin;                                          // silver_min_value
        
        _printPreset[20] = _sPrintRatio;                                        // silver_ratio
        _printPreset[21] = 1.0f;                                                // input_gain
    } else {
        // Use film stock presets from dropdown selection
        if (_printStock == 0) { // PRINT_2383 (Kodak 2383)
            if (_layerMode == RED_ONLY) {
                float preset[22] = {
                    660.0f, 545.0f, 446.0f, 500.0f,    // means
                    43.0f, 45.4f, 50.3f, 139.0f,       // left stds
                    57.6f, 32.0f, 24.8f, 200.9f,       // right stds
                    1.087f, 0.0f, 0.0f, 1.891f,        // max values
                    0.063f, 0.0f, 0.0f, 0.024f,        // min values
                    0.008f, 1.0f                        // silver ratio, input gain
                };
                for (int i = 0; i < 22; i++) _printPreset[i] = preset[i];
            } else if (_layerMode == GREEN_ONLY) {
                float preset[22] = {
                    660.0f, 545.0f, 446.0f, 500.0f,    // means
                    43.0f, 45.4f, 50.3f, 139.0f,       // left stds
                    57.6f, 32.0f, 24.8f, 200.9f,       // right stds
                    0.0f, 0.880f, 0.0f, 1.891f,        // max values
                    0.0f, 0.009f, 0.0f, 0.024f,        // min values
                    0.043f, 1.0f                        // silver ratio, input gain
                };
                for (int i = 0; i < 22; i++) _printPreset[i] = preset[i];
            } else if (_layerMode == BLUE_ONLY) {
                float preset[22] = {
                    660.0f, 545.0f, 446.0f, 500.0f,    // means
                    43.0f, 45.4f, 50.3f, 139.0f,       // left stds
                    57.6f, 32.0f, 24.8f, 200.9f,       // right stds
                    0.0f, 0.0f, 0.813f, 1.891f,        // max values
                    0.03f, 0.0f, 0.0f, 0.024f,         // min values
                    0.0043f, 1.0f                       // silver ratio, input gain
                };
                for (int i = 0; i < 22; i++) _printPreset[i] = preset[i];
            }
        } else if (_printStock == 1) { // PRINT_5384 (Kodak 5384)
            if (_layerMode == RED_ONLY) {
                float preset[22] = {
                    660.0f, 545.0f, 450.0f, 500.0f,    // means
                    50.0f, 45.0f, 40.0f, 120.0f,       // left stds
                    60.0f, 50.0f, 45.0f, 150.0f,       // right stds
                    1.20f, 0.0f, 0.0f, 1.80f,          // max values
                    0.05f, 0.0f, 0.0f, 0.02f,          // min values
                    0.01f, 1.0f                         // silver ratio, input gain
                };
                for (int i = 0; i < 22; i++) _printPreset[i] = preset[i];
            } else if (_layerMode == GREEN_ONLY) {
                float preset[22] = {
                    660.0f, 545.0f, 450.0f, 500.0f,    // means
                    50.0f, 45.0f, 40.0f, 120.0f,       // left stds
                    60.0f, 50.0f, 45.0f, 150.0f,       // right stds
                    0.0f, 1.10f, 0.0f, 1.80f,          // max values
                    0.0f, 0.04f, 0.0f, 0.02f,          // min values
                    0.02f, 1.0f                         // silver ratio, input gain
                };
                for (int i = 0; i < 22; i++) _printPreset[i] = preset[i];
            } else if (_layerMode == BLUE_ONLY) {
                float preset[22] = {
                    660.0f, 545.0f, 450.0f, 500.0f,    // means
                    50.0f, 45.0f, 40.0f, 120.0f,       // left stds
                    60.0f, 50.0f, 45.0f, 150.0f,       // right stds
                    0.0f, 0.0f, 1.00f, 1.80f,          // max values
                    0.0f, 0.0f, 0.03f, 0.02f,          // min values
                    0.015f, 1.0f                        // silver ratio, input gain
                };
                for (int i = 0; i < 22; i++) _printPreset[i] = preset[i];
            }
        }
    }
}

void ImagePrintMean::processImagesCUDA()
{
#ifndef __APPLE__
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    // ACTUALLY CALL THE KERNEL!
    RunProjectorCudaKernel(_pCudaStream, width, height,
                          _negativePreset, _printPreset,
                          input, output);
#endif
}

void ImagePrintMean::processImagesMetal()
{
#ifdef __APPLE__
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    // UPDATED CALL WITH ALPHA PARAMETERS!
    RunProjectorMetalKernel(_pMetalCmdQ, width, height,
                           _negativePreset, _printPreset,
                           _alphaPassThru, _alphaMin, _alphaMax, _linearAdjustment,
                           input, output);
#endif
}

extern void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float* p_Gain, const float* p_Input, float* p_Output);

void ImagePrintMean::processImagesOpenCL()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunOpenCLKernel(_pOpenCLCmdQ, width, height, _scales, input, output);
}

void ImagePrintMean::multiThreadProcessImages(OfxRectI p_ProcWindow)
{
    // Simple CPU fallback - just copy input to output for now
    // TODO: Implement full film simulation on CPU
    for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y)
    {
        if (_effect.abort()) break;

        float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));

        for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x)
        {
            float* srcPix = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(x, y) : 0);

            if (srcPix)
            {
                // For now, just pass through with a simple test modification
                dstPix[0] = srcPix[0] * 0.5f; // Darken red channel to see if it's working
                dstPix[1] = srcPix[1] * 0.5f; // Darken green channel  
                dstPix[2] = srcPix[2] * 0.5f; // Darken blue channel
                dstPix[3] = srcPix[3];        // Pass through alpha
            }
            else
            {
                // no src pixel here, be black and transparent
                for (int c = 0; c < 4; ++c)
                {
                    dstPix[c] = 0;
                }
            }

            // increment the dst pixel
            dstPix += 4;
        }
    }
}

void ImagePrintMean::setSrcImg(OFX::Image* p_SrcImg)
{
    _srcImg = p_SrcImg;
}

void ImagePrintMean::setScales(float p_PrintMean, float p_printleftstd, float p_printrightstd, float p_ScaleA)
{
    _scales[0] = p_PrintMean;
    _scales[1] = p_printleftstd;
    _scales[2] = p_printrightstd;
    _scales[3] = p_ScaleA;
}

////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin that does our work */
class FilmProjector : public OFX::ImageEffect
{
public:
    explicit FilmProjector(OfxImageEffectHandle p_Handle);

    /* Override the render */
    virtual void render(const OFX::RenderArguments& p_Args);

    /* Override is identity */
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);

    /* Override changedParam */
    virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);

    /* Override changed clip */
    virtual void changedClip(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ClipName);

    /* Set the enabledness of the component scale params depending on the type of input image and the state of the AlphaPassThru param */
    void setEnabledness();

    /* Set up and run a processor */
    void setupAndProcess(ImagePrintMean &p_ImagePrintMean, const OFX::RenderArguments& p_Args);

private:
    // Does not own the following pointers
    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;

    OFX::DoubleParam* m_Scale;
    OFX::DoubleParam* m_PrintMean;
    OFX::DoubleParam* m_printleftstd;
    OFX::DoubleParam* m_printrightstd;
    OFX::DoubleParam* m_ScaleA;
    OFX::BooleanParam* m_ComponentScalesEnabled;

    // New boolean parameters for custom stock selection
    OFX::BooleanParam* m_CustomNegativeSelect;
    OFX::BooleanParam* m_CustomPrintSelect;
    
    // Stock selection dropdowns
    OFX::ChoiceParam* m_NegativeStockSelect;
    OFX::ChoiceParam* m_PrintStockSelect;
    
    // Group parameters (to control group enablement)
    OFX::GroupParam* m_NegativeGroup;
    OFX::GroupParam* m_PrintGroup;
    
    // All the negative film parameters
    OFX::DoubleParam* m_NegMean;
    OFX::DoubleParam* m_NegLeftStd;
    OFX::DoubleParam* m_NegRightStd;
    OFX::DoubleParam* m_NegMax;
    OFX::DoubleParam* m_NegMin;
    
    // All the negative silver parameters
    OFX::DoubleParam* m_SNegMean;
    OFX::DoubleParam* m_SNegLeftStd;
    OFX::DoubleParam* m_SNegRightStd;
    OFX::DoubleParam* m_SNegMax;
    OFX::DoubleParam* m_SNegMin;
    OFX::DoubleParam* m_SNegRatio;
    
    // All the print parameters (existing ones you already have)
    OFX::DoubleParam* m_PrintMin;
    OFX::DoubleParam* m_SPrintMean;
    OFX::DoubleParam* m_SPrintLeftStd;
    OFX::DoubleParam* m_SPrintRightStd;
    OFX::DoubleParam* m_SPrintMax;
    OFX::DoubleParam* m_SPrintMin;
    OFX::DoubleParam* m_SPrintRatio;
  OFX::DoubleParam* m_LinearAdjustment;
    OFX::ChoiceParam* m_LayerModeSelect; // ADD THIS LINE
};

FilmProjector::FilmProjector(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

    // Existing parameters
    m_Scale = fetchDoubleParam("scale");
    m_PrintMean = fetchDoubleParam("PrintMean");
    m_printleftstd = fetchDoubleParam("printleftstd");
    m_printrightstd = fetchDoubleParam("printrightstd");
    m_ScaleA = fetchDoubleParam("printmax");
    m_ComponentScalesEnabled = fetchBooleanParam("AlphaPassThru");

    // New boolean controls
    m_CustomNegativeSelect = fetchBooleanParam("customenegative");
    m_CustomPrintSelect = fetchBooleanParam("customeprint");
    
    // Stock selection dropdowns
    m_NegativeStockSelect = fetchChoiceParam("negativestockselection");
    m_PrintStockSelect = fetchChoiceParam("printstockselection");
    
    // Fetch all the negative parameters
    m_NegMean = fetchDoubleParam("NegMean");
    m_NegLeftStd = fetchDoubleParam("negleftstd");
    m_NegRightStd = fetchDoubleParam("negrightstd");
    m_NegMax = fetchDoubleParam("negmax");
    m_NegMin = fetchDoubleParam("negmin");
    
    m_SNegMean = fetchDoubleParam("SNegMean");
    m_SNegLeftStd = fetchDoubleParam("Snegleftstd");
    m_SNegRightStd = fetchDoubleParam("Snegrightstd");
    m_SNegMax = fetchDoubleParam("Snegmax");
    m_SNegMin = fetchDoubleParam("Snegmin");
    m_SNegRatio = fetchDoubleParam("Sration");

    // Add the missing print parameter fetches:
    m_PrintMin = fetchDoubleParam("printmin");
    m_SPrintMean = fetchDoubleParam("SPrintMean");
    m_SPrintLeftStd = fetchDoubleParam("Sprintleftstd");
    m_SPrintRightStd = fetchDoubleParam("Sprintrightstd");
    m_SPrintMax = fetchDoubleParam("Sprintmax");
    m_SPrintMin = fetchDoubleParam("Sprintin");
    m_SPrintRatio = fetchDoubleParam("Sprintratio");
    
    // Fetch the group parameters
    m_NegativeGroup = fetchGroupParam("NegativeGroup");
    m_PrintGroup = fetchGroupParam("PrintGroup");
    m_LayerModeSelect = fetchChoiceParam("layermodeselection");
    m_LinearAdjustment = fetchDoubleParam("LinearAdjustment");
    // Set the initial enableness
    setEnabledness();
}

void FilmProjector::render(const OFX::RenderArguments& p_Args)
{
    if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
    {
        ImagePrintMean imagePrintMean(*this);
        setupAndProcess(imagePrintMean, p_Args);
    }
    else
    {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

bool FilmProjector::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
    double rScale = 1.0, gScale = 1.0, bScale = 1.0, aScale = 1.0;

    if (m_ComponentScalesEnabled->getValueAtTime(p_Args.time))
    {
        rScale = m_PrintMean->getValueAtTime(p_Args.time);
        gScale = m_printleftstd->getValueAtTime(p_Args.time);
        bScale = m_printrightstd->getValueAtTime(p_Args.time);
        aScale = m_ScaleA->getValueAtTime(p_Args.time);
    }

    const double scale = m_Scale->getValueAtTime(p_Args.time);
    rScale *= scale;
    gScale *= scale;
    bScale *= scale;

    if ((rScale == 1.0) && (gScale == 1.0) && (bScale == 1.0) && (aScale == 1.0))
    {
        p_IdentityClip = m_SrcClip;
        p_IdentityTime = p_Args.time;
        return true;
    }

    return false;
}

void FilmProjector::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
    // REMOVED "AlphaPassThru" from this condition
    // AlphaPassThru should not affect parameter enablement anymore
    if (p_ParamName == "customenegative" || 
        p_ParamName == "customeprint")
    {
        setEnabledness();
    }
}

void FilmProjector::changedClip(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ClipName)
{
    if (p_ClipName == kOfxImageEffectSimpleSourceClipName)
    {
        setEnabledness();
    }
}

void FilmProjector::setEnabledness()
{
    // Get the current values of the boolean switches
    const bool useCustomNegative = m_CustomNegativeSelect->getValue();
    const bool useCustomPrint = m_CustomPrintSelect->getValue();
    
    // NEGATIVE STOCK CONTROL LOGIC
    // When "Use Custom Negative" is OFF: Enable dropdown, disable all custom negative parameters
    // When "Use Custom Negative" is ON: Disable dropdown, enable all custom negative parameters
    m_NegativeStockSelect->setEnabled(!useCustomNegative);
    
    m_NegMean->setEnabled(useCustomNegative);
    m_NegLeftStd->setEnabled(useCustomNegative);
    m_NegRightStd->setEnabled(useCustomNegative);
    m_NegMax->setEnabled(useCustomNegative);
    m_NegMin->setEnabled(useCustomNegative);
    
    m_SNegMean->setEnabled(useCustomNegative);
    m_SNegLeftStd->setEnabled(useCustomNegative);
    m_SNegRightStd->setEnabled(useCustomNegative);
    m_SNegMax->setEnabled(useCustomNegative);
    m_SNegMin->setEnabled(useCustomNegative);
    m_SNegRatio->setEnabled(useCustomNegative);

    // PRINT STOCK CONTROL LOGIC  
    // When "Use Custom Print" is OFF: Enable dropdown, disable all custom print parameters
    // When "Use Custom Print" is ON: Disable dropdown, enable all custom print parameters
    m_PrintStockSelect->setEnabled(!useCustomPrint);
    
    // Enable ALL print parameters when useCustomPrint is true
    // Basic print parameters
    m_PrintMean->setEnabled(useCustomPrint);
    m_printleftstd->setEnabled(useCustomPrint);
    m_printrightstd->setEnabled(useCustomPrint);
    m_ScaleA->setEnabled(useCustomPrint);  // This is printmax
    
    // YOU WERE MISSING THESE! Add all the print parameters you fetch in constructor:
    m_PrintMin->setEnabled(useCustomPrint);
    
    // Silver print parameters
    m_SPrintMean->setEnabled(useCustomPrint);
    m_SPrintLeftStd->setEnabled(useCustomPrint);
    m_SPrintRightStd->setEnabled(useCustomPrint);
    m_SPrintMax->setEnabled(useCustomPrint);
    m_SPrintMin->setEnabled(useCustomPrint);
    m_SPrintRatio->setEnabled(useCustomPrint);
}

void FilmProjector::setupAndProcess(ImagePrintMean& p_ImagePrintMean, const OFX::RenderArguments& p_Args)
{
    // Get the dst image
    std::unique_ptr<OFX::Image> dst(m_DstClip->fetchImage(p_Args.time));
    OFX::BitDepthEnum dstBitDepth = dst->getPixelDepth();
    OFX::PixelComponentEnum dstComponents = dst->getPixelComponents();

    // Get the src image
    std::unique_ptr<OFX::Image> src(m_SrcClip->fetchImage(p_Args.time));
    OFX::BitDepthEnum srcBitDepth = src->getPixelDepth();
    OFX::PixelComponentEnum srcComponents = src->getPixelComponents();

    // Check to see if the bit depth and number of components are the same
    if ((srcBitDepth != dstBitDepth) || (srcComponents != dstComponents))
    {
        OFX::throwSuiteStatusException(kOfxStatErrValue);
    }

    // Get all projector simulation parameters
    bool useCustomNegative = m_CustomNegativeSelect->getValueAtTime(p_Args.time);
    bool useCustomPrint = m_CustomPrintSelect->getValueAtTime(p_Args.time);
    
    // FIX THESE LINES - getValueAtTime for ChoiceParam needs two parameters
    int negativeStock;
    m_NegativeStockSelect->getValueAtTime(p_Args.time, negativeStock);
    
    int printStock;
    m_PrintStockSelect->getValueAtTime(p_Args.time, printStock);
    
    // Get all negative parameters
    float negMean = m_NegMean->getValueAtTime(p_Args.time);
    float negLeft = m_NegLeftStd->getValueAtTime(p_Args.time);
    float negRight = m_NegRightStd->getValueAtTime(p_Args.time);
    float negMax = m_NegMax->getValueAtTime(p_Args.time);
    float negMin = m_NegMin->getValueAtTime(p_Args.time);
    
    float sNegMean = m_SNegMean->getValueAtTime(p_Args.time);
    float sNegLeft = m_SNegLeftStd->getValueAtTime(p_Args.time);
    float sNegRight = m_SNegRightStd->getValueAtTime(p_Args.time);
    float sNegMax = m_SNegMax->getValueAtTime(p_Args.time);
    float sNegMin = m_SNegMin->getValueAtTime(p_Args.time);
    float sNegRatio = m_SNegRatio->getValueAtTime(p_Args.time);
    
    // Get all print parameters
    float printMean = m_PrintMean->getValueAtTime(p_Args.time);
    float printLeft = m_printleftstd->getValueAtTime(p_Args.time);
    float printRight = m_printrightstd->getValueAtTime(p_Args.time);
    float printMax = m_ScaleA->getValueAtTime(p_Args.time);  // This is printmax
    float printMin = m_PrintMin->getValueAtTime(p_Args.time);
    
    float sPrintMean = m_SPrintMean->getValueAtTime(p_Args.time);
    float sPrintLeft = m_SPrintLeftStd->getValueAtTime(p_Args.time);
    float sPrintRight = m_SPrintRightStd->getValueAtTime(p_Args.time);
    float sPrintMax = m_SPrintMax->getValueAtTime(p_Args.time);
    float sPrintMin = m_SPrintMin->getValueAtTime(p_Args.time);
    float sPrintRatio = m_SPrintRatio->getValueAtTime(p_Args.time);

    // ADD THESE LINES - Get alpha parameters
    bool alphaPassThru = m_ComponentScalesEnabled->getValueAtTime(p_Args.time);
    
    // YOU NEED TO ADD THESE PARAMETERS TO YOUR PLUGIN:
    // For now, use defaults until you add the actual UI parameters
    float alphaMin = 0.0f;        // TODO: Add m_AlphaMin parameter
    float alphaMax = 1.0f;        // TODO: Add m_AlphaMax parameter  
    float linearAdjustment = m_LinearAdjustment->getValueAtTime(p_Args.time);  

    // Set the images
    p_ImagePrintMean.setDstImg(dst.get());
    p_ImagePrintMean.setSrcImg(src.get());

    // Setup OpenCL and CUDA Render arguments
    p_ImagePrintMean.setGPURenderArgs(p_Args);

    // Set the render window
    p_ImagePrintMean.setRenderWindow(p_Args.renderWindow);

    // Pass all projector parameters to the processor
    int layerMode;
    m_LayerModeSelect->getValueAtTime(p_Args.time, layerMode);
    p_ImagePrintMean.setProjectorParams(
        useCustomNegative, useCustomPrint, negativeStock, printStock, layerMode,
        negMean, negLeft, negRight, negMax, negMin,
        sNegMean, sNegLeft, sNegRight, sNegMax, sNegMin, sNegRatio,
        printMean, printLeft, printRight, printMax, printMin,
        sPrintMean, sPrintLeft, sPrintRight, sPrintMax, sPrintMin, sPrintRatio,
        // ADD THESE LINES
        alphaPassThru, alphaMin, alphaMax, linearAdjustment
    );

    // Call the base class process member, this will call the derived templated process code
    p_ImagePrintMean.process();
}

////////////////////////////////////////////////////////////////////////////////

class FilmProjectorInteract : public OFX::OverlayInteract
{
public:
    FilmProjectorInteract(OfxInteractHandle p_Handle, OFX::ImageEffect* /*p_Effect*/)
        : OFX::OverlayInteract(p_Handle)
    { }

    virtual bool draw(const OFX::DrawArgs& p_Args)
    {
        OfxDrawContextHandle contextHandle = OFX::Private::gDrawSuite ? p_Args.context : nullptr;
        if (!contextHandle) return false;

        const OfxRGBAColourF color = { 1.0f, 0.3f, 0.3f, 1.0f };
        OFX::Private::gDrawSuite->setColour(contextHandle, &color);
        {
            const OfxPointD points[] = { {0.0f, 0.0f}, {100.0f, 100.0f} };
            OFX::Private::gDrawSuite->draw(contextHandle, kOfxDrawPrimitiveLines, points, 2);
        }
        {
            const OfxPointD points[] = { {100.0f, 100.0f}, {200.0f, 200.0f} };
            OFX::Private::gDrawSuite->draw(contextHandle, kOfxDrawPrimitiveEllipse, points, 2);
        }
        {
            const OfxPointD points[] = { {200.0f, 200.0f} };
            OFX::Private::gDrawSuite->drawText(contextHandle, "FilmProjector", points, kOfxDrawTextAlignmentLeft);
        }

        return true;
    }
};

class GainOverlayInteractDescriptor : public OFX::DefaultEffectOverlayDescriptor<GainOverlayInteractDescriptor, FilmProjectorInteract>
{
};

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

FilmProjectorFactory::FilmProjectorFactory()
    : OFX::PluginFactoryHelper<FilmProjectorFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void FilmProjectorFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
{
    // Basic labels
    p_Desc.setLabels(kPluginName, kPluginName, kPluginName);
    p_Desc.setPluginGrouping(kPluginGrouping);
    p_Desc.setPluginDescription(kPluginDescription);

    // Add the supported contexts, only filter at the moment
    p_Desc.addSupportedContext(eContextFilter);
    p_Desc.addSupportedContext(eContextGeneral);

    // Add supported pixel depths
    p_Desc.addSupportedBitDepth(eBitDepthFloat);

    // Set a few flags
    p_Desc.setSingleInstance(false);
    p_Desc.setHostFrameThreading(false);
    p_Desc.setSupportsMultiResolution(kSupportsMultiResolution);
    p_Desc.setSupportsTiles(kSupportsTiles);
    p_Desc.setTemporalClipAccess(false);
    p_Desc.setRenderTwiceAlways(false);
    p_Desc.setSupportsMultipleClipPARs(kSupportsMultipleClipPARs);

    // Setup OpenCL render capability flags
    p_Desc.setSupportsOpenCLRender(true);

    // Setup CUDA render capability flags on non-Apple system
#ifndef __APPLE__
    p_Desc.setSupportsCudaRender(true);
    p_Desc.setSupportsCudaStream(true);
#endif

    // Setup Metal render capability flags only on Apple system
#ifdef __APPLE__
    p_Desc.setSupportsMetalRender(true);
#endif

    // Indicates that the plugin output does not depend on location or neighbours of a given pixel.
    // Therefore, this plugin could be executed during LUT generation.
    p_Desc.setNoSpatialAwareness(true);

    // Attach overlay interact
    p_Desc.setOverlayInteractDescriptor(new GainOverlayInteractDescriptor());
}

static DoubleParamDescriptor* defineScaleParam(OFX::ImageEffectDescriptor& p_Desc, const std::string& p_Name, const std::string& p_Label,
                                               const std::string& p_Hint, GroupParamDescriptor* p_Parent)
{
    DoubleParamDescriptor* param = p_Desc.defineDoubleParam(p_Name);
    param->setLabels(p_Label, p_Label, p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    param->setDefault(1);
    param->setRange(0, 10);
    param->setIncrement(0.1);
    param->setDisplayRange(0, 10);
    param->setDoubleType(eDoubleTypePlain);

    if (p_Parent)
    {
        param->setParent(*p_Parent);
    }

    return param;
}
static DoubleParamDescriptor* defineMeanParam(OFX::ImageEffectDescriptor& p_Desc, const std::string& p_Name, const std::string& p_Label,
                                               const std::string& p_Hint, GroupParamDescriptor* p_Parent)
{
    DoubleParamDescriptor* param = p_Desc.defineDoubleParam(p_Name);
    param->setLabels(p_Label, p_Label, p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    param->setDefault(1);
    param->setRange(0, 10);
    param->setIncrement(0.1);
    param->setDisplayRange(0, 10);
    param->setDoubleType(eDoubleTypePlain);

    if (p_Parent)
    {
        param->setParent(*p_Parent);
    }

    return param;
}

// Enhanced parameter definition function with specific defaults
static DoubleParamDescriptor* defineFilmParam(OFX::ImageEffectDescriptor& p_Desc, 
                                             const std::string& p_Name, 
                                             const std::string& p_Label,
                                             const std::string& p_Hint, 
                                             GroupParamDescriptor* p_Parent,
                                             double defaultValue = 1.0,
                                             double minValue = 0.0,
                                             double maxValue = 10.0,
                                             double increment = 0.1)
{
    DoubleParamDescriptor* param = p_Desc.defineDoubleParam(p_Name);
    param->setLabels(p_Label, p_Label, p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    param->setDefault(defaultValue);        // CUSTOM DEFAULT
    param->setRange(minValue, maxValue);
    param->setIncrement(increment);
    param->setDisplayRange(minValue, maxValue);
    param->setDoubleType(eDoubleTypePlain);

    if (p_Parent) {
        param->setParent(*p_Parent);
    }
    return param;
}

void FilmProjectorFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
{
    // Source clip only in the filter context
    // Create the mandated source clip
    ClipDescriptor* srcClip = p_Desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    srcClip->addSupportedComponent(ePixelComponentRGBA);
    srcClip->setTemporalClipAccess(false);
    srcClip->setSupportsTiles(kSupportsTiles);
    srcClip->setIsMask(false);

    // Create the mandated output clip
    ClipDescriptor* dstClip = p_Desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(ePixelComponentRGBA);
    dstClip->addSupportedComponent(ePixelComponentAlpha);
    dstClip->setSupportsTiles(kSupportsTiles);

    // Make some pages and to things in
    PageParamDescriptor* page = p_Desc.definePageParam("Controls");
  GroupParamDescriptor* Config = p_Desc.defineGroupParam("Config");
    Config->setHint("Scales on the individual component");
    Config->setLabels("Config", "Config", "Config");

        // Negative Group
    GroupParamDescriptor* negativeGroup = p_Desc.defineGroupParam("NegativeGroup");
    negativeGroup->setHint("Negative film characteristics");
    negativeGroup->setLabels("Negative Stock", "Negative Stock", "Negative Stock");
    // Group param to group the scales
    GroupParamDescriptor* PrintGroup = p_Desc.defineGroupParam("PrintGroup");
    PrintGroup->setHint("Scales on the individual component");
    PrintGroup->setLabels("Print Stock", "Print Stock", "Print Stock");
    GroupParamDescriptor* LegacyGroup = p_Desc.defineGroupParam("LegacyOptions");
    LegacyGroup->setHint("Scales on the individual component");
    LegacyGroup->setLabels("OldParams", "OldParams", "OldParams");
    
    
    // Make overall scale params
    DoubleParamDescriptor* param = defineScaleParam(p_Desc, "scale", "scale", "Scales all component in the image", LegacyGroup);
    page->addChild(*param);

  // Blend Mode dropdown - FIRST in config group
    ChoiceParamDescriptor* NegativeStockSelect = p_Desc.defineChoiceParam("negativestockselection");
    NegativeStockSelect->setLabel("Negative Stock");
    NegativeStockSelect->setHint("Choose Your Negative Stock modes");
    NegativeStockSelect->appendOption("5247");     // Index 0
    NegativeStockSelect->appendOption("5283");   // Index 1  
    NegativeStockSelect->setDefault(0); // Default to "Blend Control"
    NegativeStockSelect->setAnimates(true);
    NegativeStockSelect->setParent(*Config);
    page->addChild(*NegativeStockSelect);  // Blend Mode dropdown - FIRST in config group
   
   ChoiceParamDescriptor* PrintStockSelect = p_Desc.defineChoiceParam("printstockselection");
    PrintStockSelect->setLabel("Print Stock");
    PrintStockSelect->setHint("Choose Your Print Stock modes");
    PrintStockSelect->appendOption("2383");     // Index 0
    PrintStockSelect->appendOption("2384");   // Index 1  
    PrintStockSelect->setDefault(0); // Default to "Blend Control"
    PrintStockSelect->setAnimates(true);
    PrintStockSelect->setParent(*Config);
    page->addChild(*PrintStockSelect);  // Blend Mode dropdown - FIRST in config group

       ChoiceParamDescriptor* OperativeModeSelect = p_Desc.defineChoiceParam("OperativeModeSelct");
    OperativeModeSelect->setLabel("Operative Mode");
    OperativeModeSelect->setHint("Choose Your modes");
    OperativeModeSelect->appendOption("Negative Mode");     // Index 0
    OperativeModeSelect->appendOption("Print Mode");        // Index 1  
    OperativeModeSelect->appendOption("Pass Thru Mode");    // Index 2 
    OperativeModeSelect->setDefault(2); // Default to "Pass Thru Mode"
    OperativeModeSelect->setAnimates(true);
    OperativeModeSelect->setParent(*Config);
    page->addChild(*OperativeModeSelect);  // Operative Mode dropdown in config group
   
ChoiceParamDescriptor* LayerModeSelect = p_Desc.defineChoiceParam("layermodeselection");
LayerModeSelect->setLabel("Layer Mode");
LayerModeSelect->setHint("Select the color layer mode");
LayerModeSelect->appendOption("Red/Cyan");      // Index 0
LayerModeSelect->appendOption("Green/Magenta"); // Index 1
LayerModeSelect->appendOption("Blue/Yellow");   // Index 2
LayerModeSelect->setDefault(0); // Default to "Red/Cyan"
LayerModeSelect->setAnimates(true);
LayerModeSelect->setParent(*Config);
page->addChild(*LayerModeSelect);  // Layer Mode dropdown in config group


    // Use Custom Negative
    BooleanParamDescriptor* CustomNegativeSelect = p_Desc.defineBooleanParam("customenegative");
    CustomNegativeSelect->setDefault(false);
    CustomNegativeSelect->setHint("Work Custom Negative Values");
    CustomNegativeSelect->setLabels("Use Custom Negative", "Use Custom Negative", "Use Custom Negative");
    CustomNegativeSelect->setParent(*Config);
    page->addChild(*CustomNegativeSelect);
   // Use Custom Print
    BooleanParamDescriptor* CustomPrintSelect = p_Desc.defineBooleanParam("customeprint");
    CustomPrintSelect->setDefault(false);
    CustomPrintSelect->setHint("Work Custom Print Values");
    CustomPrintSelect->setLabels("Use Custom Print", "Use Custom Print", "Use Custom Print");
    CustomPrintSelect->setParent(*Config);
    page->addChild(*CustomPrintSelect);

    // Add a boolean to enable the component scale
    BooleanParamDescriptor* AlphaSelect = p_Desc.defineBooleanParam("AlphaPassThru");
    AlphaSelect->setDefault(true);
    AlphaSelect->setHint("Calculate Alpha and Pass it On... or not");
    AlphaSelect->setLabels("Alpha Pass Thru", "Alpha Pass Thru", "Alpha Pass Thru");
    AlphaSelect->setParent(*Config);
    page->addChild(*AlphaSelect);

param = defineFilmParam(p_Desc, "LinearAdjustment", "Printer Light ADJ", "Linear adjustment between passes", 
                       Config, 1.0, 0.1, 10.0, 0.1);
page->addChild(*param);
////////////////////////////////////////////////////////////////////////////////
    // PRINT FILM PARAMETERS - WITH PROPER DEFAULTS
    ////////////////////////////////////////////////////////////////////////////////
    
    // Based on Kodak 2383 defaults for RED_ONLY mode
    param = defineFilmParam(p_Desc, "PrintMean", "Print Mean", "Print film mean wavelength", 
                           PrintGroup, 660.0, 400.0, 800.0, 5.0);
    page->addChild(*param);

    param = defineFilmParam(p_Desc, "printleftstd", "Print Left Std", "Print film left standard deviation", 
                           PrintGroup, 43.0, 10.0, 200.0, 1.0);
    page->addChild(*param);

    param = defineFilmParam(p_Desc, "printrightstd", "Print Right Std", "Print film right standard deviation", 
                           PrintGroup, 57.6, 10.0, 200.0, 1.0);
    page->addChild(*param);

    param = defineFilmParam(p_Desc, "printmax", "Print Max Density", "Maximum print film density", 
                           PrintGroup, 1.087, 0.0, 5.0, 0.01);
    page->addChild(*param);

    param = defineFilmParam(p_Desc, "printmin", "Print Min Density", "Minimum print film density", 
                           PrintGroup, 0.063, 0.0, 1.0, 0.001);
    page->addChild(*param);

    // Silver print parameters with proper defaults
    param = defineFilmParam(p_Desc, "SPrintMean", "Silver Print Mean", "Silver print mean wavelength", 
                           PrintGroup, 500.0, 400.0, 700.0, 5.0);
    page->addChild(*param);

    param = defineFilmParam(p_Desc, "Sprintleftstd", "Silver Print Left Std", "Silver print left standard deviation", 
                           PrintGroup, 139.0, 50.0, 300.0, 1.0);
    page->addChild(*param);

    param = defineFilmParam(p_Desc, "Sprintrightstd", "Silver Print Right Std", "Silver print right standard deviation", 
                           PrintGroup, 200.9, 50.0, 400.0, 1.0);
    page->addChild(*param);

    param = defineFilmParam(p_Desc, "Sprintmax", "Silver Print Max", "Silver print maximum density", 
                           PrintGroup, 1.891, 0.0, 5.0, 0.01);
    page->addChild(*param);

    param = defineFilmParam(p_Desc, "Sprintin", "Silver Print Min", "Silver print minimum density", 
                           PrintGroup, 0.024, 0.0, 1.0, 0.001);
    page->addChild(*param);

    param = defineFilmParam(p_Desc, "Sprintratio", "Silver Print Ratio", "Silver to dye ratio", 
                           PrintGroup, 0.008, 0.0, 1.0, 0.001);
    page->addChild(*param);

    ////////////////////////////////////////////////////////////////////////////////
    // NEGATIVE FILM PARAMETERS - WITH PROPER DEFAULTS  
    ////////////////////////////////////////////////////////////////////////////////
    
    // Based on Kodak 5247 defaults for RED_ONLY mode
    param = defineFilmParam(p_Desc, "NegMean", "Negative Mean", "Negative film mean wavelength", 
                           negativeGroup, 650.0, 400.0, 800.0, 5.0);
    page->addChild(*param);

    param = defineFilmParam(p_Desc, "negleftstd", "Negative Left Std", "Negative film left standard deviation", 
                           negativeGroup, 48.1, 10.0, 200.0, 1.0);
    page->addChild(*param);

    param = defineFilmParam(p_Desc, "negrightstd", "Negative Right Std", "Negative film right standard deviation", 
                           negativeGroup, 40.5, 10.0, 200.0, 1.0);
    page->addChild(*param);

    param = defineFilmParam(p_Desc, "negmax", "Negative Max Density", "Maximum negative film density", 
                           negativeGroup, 0.95, 0.0, 5.0, 0.01);
    page->addChild(*param);

    param = defineFilmParam(p_Desc, "negmin", "Negative Min Density", "Minimum negative film density", 
                           negativeGroup, 0.102, 0.0, 1.0, 0.001);
    page->addChild(*param);

    // Silver negative parameters with proper defaults
    param = defineFilmParam(p_Desc, "SNegMean", "Silver Negative Mean", "Silver negative mean wavelength", 
                           negativeGroup, 511.5, 400.0, 700.0, 5.0);
    page->addChild(*param);

    param = defineFilmParam(p_Desc, "Snegleftstd", "Silver Negative Left Std", "Silver negative left standard deviation", 
                           negativeGroup, 110.6, 50.0, 300.0, 1.0);
    page->addChild(*param);

    param = defineFilmParam(p_Desc, "Snegrightstd", "Silver Negative Right Std", "Silver negative right standard deviation", 
                           negativeGroup, 19.2, 10.0, 200.0, 1.0);
    page->addChild(*param);

    param = defineFilmParam(p_Desc, "Snegmax", "Silver Negative Max", "Silver negative maximum density", 
                           negativeGroup, 1.575, 0.0, 5.0, 0.01);
    page->addChild(*param);

    param = defineFilmParam(p_Desc, "Snegmin", "Silver Negative Min", "Silver negative minimum density", 
                           negativeGroup, 0.126, 0.0, 1.0, 0.001);
    page->addChild(*param);

    param = defineFilmParam(p_Desc, "Sration", "Silver Negative Ratio", "Silver to dye ratio", 
                           negativeGroup, 0.228, 0.0, 1.0, 0.001);
    page->addChild(*param);

}

ImageEffect* FilmProjectorFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new FilmProjector(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static FilmProjectorFactory FilmProjector;
    p_FactoryArray.push_back(&FilmProjector);
}
