#import <Metal/Metal.h>

#include <unordered_map>
#include <mutex>

const char* kernelSource =  \
"#include <metal_stdlib>\n" \
"using namespace metal; \n" \
"kernel void ContrastSatVolumeKernel(constant int& p_Width [[buffer (11)]], constant int& p_Height [[buffer (12)]],                                  \n" \
"                                    constant float* p_RgbGammas [[buffer (13)]], constant float* p_CymGammas [[buffer (14)]],                      \n" \
"                                    constant float* p_RgbMidgreys [[buffer (15)]], constant float* p_CymMidgreys [[buffer (16)]],                  \n" \
"                                    constant float* p_Tilts [[buffer (17)]],                                                                         \n" \
"                                    const device float* p_Input [[buffer (0)]], device float* p_Output [[buffer (8)]],                             \n" \
"                                    uint2 id [[ thread_position_in_grid ]])                                                                          \n" \
"{                                                                                                                                                   \n" \
"   if ((id.x < p_Width) && (id.y < p_Height))                                                                                                       \n" \
"   {                                                                                                                                                \n" \
"       const int index = ((id.y * p_Width) + id.x) * 4;                                                                                             \n" \
"                                                                                                                                                    \n" \
"       // Extract RGBA values and clamp to safe range                                                                                              \n" \
"       float r = clamp(p_Input[index + 0], 0.001f, 0.999f);                                                                                        \n" \
"       float g = clamp(p_Input[index + 1], 0.001f, 0.999f);                                                                                        \n" \
"       float b = clamp(p_Input[index + 2], 0.001f, 0.999f);                                                                                        \n" \
"       float a = p_Input[index + 3];                                                                                                                \n" \
"                                                                                                                                                    \n" \
"       // Extract parameters from arrays                                                                                                            \n" \
"       float gammaR = p_RgbGammas[0], gammaG = p_RgbGammas[1], gammaB = p_RgbGammas[2];                                                            \n" \
"       float gammaC = p_CymGammas[0], gammaM = p_CymGammas[1], gammaY = p_CymGammas[2];                                                            \n" \
"       float midgreyR = p_RgbMidgreys[0], midgreyG = p_RgbMidgreys[1], midgreyB = p_RgbMidgreys[2];                                               \n" \
"       float midgreyC = p_CymMidgreys[0], midgreyM = p_CymMidgreys[1], midgreyY = p_CymMidgreys[2];                                               \n" \
"       float tiltCR = p_Tilts[0], tiltGM = p_Tilts[1], tiltBY = p_Tilts[2];                                                                        \n" \
"                                                                                                                                                    \n" \
"       // Apply RGB gamma adjustments                                                                                                               \n" \
"       float recipMidgreyR = 1.0f / clamp(midgreyR, 0.001f, 0.999f);                                                                               \n" \
"       float recipMidgreyG = 1.0f / clamp(midgreyG, 0.001f, 0.999f);                                                                               \n" \
"       float recipMidgreyB = 1.0f / clamp(midgreyB, 0.001f, 0.999f);                                                                               \n" \
"       float rAdjusted = pow(r * recipMidgreyR, gammaR) * midgreyR;                                                                                 \n" \
"       float gAdjusted = pow(g * recipMidgreyG, gammaG) * midgreyG;                                                                                 \n" \
"       float bAdjusted = pow(b * recipMidgreyB, gammaB) * midgreyB;                                                                                 \n" \
"                                                                                                                                                    \n" \
"       // Apply CYM gamma adjustments                                                                                                               \n" \
"       float recipMidgreyC = 1.0f / clamp(midgreyC, 0.001f, 0.999f);                                                                               \n" \
"       float recipMidgreyM = 1.0f / clamp(midgreyM, 0.001f, 0.999f);                                                                               \n" \
"       float recipMidgreyY = 1.0f / clamp(midgreyY, 0.001f, 0.999f);                                                                               \n" \
"       // True complement processing                                                                                                               \n" \
"       float cAdjusted = 1.0f - pow((1.0f - r) * recipMidgreyC, gammaC) * midgreyC;  // Process complement of Red                                \n" \
"       float mAdjusted = 1.0f - pow((1.0f - g) * recipMidgreyM, gammaM) * midgreyM;  // Process complement of Green                              \n" \
"       float yAdjusted = 1.0f - pow((1.0f - b) * recipMidgreyY, gammaY) * midgreyY;  // Process complement of Blue                                \n" \
"                                                                                                                                                    \n" \
"       // Blend between RGB and CYM using tilt parameters                                                                                           \n" \
"       float finalR = cAdjusted + (rAdjusted - cAdjusted) * tiltCR;                                                                                 \n" \
"       float finalG = mAdjusted + (gAdjusted - mAdjusted) * tiltGM;                                                                                 \n" \
"       float finalB = yAdjusted + (bAdjusted - yAdjusted) * tiltBY;                                                                                 \n" \
"                                                                                                                                                    \n" \
"       // Clamp and output                                                                                                                          \n" \
"       p_Output[index + 0] = clamp(finalR, 0.0f, 1.0f);                                                                                             \n" \
"       p_Output[index + 1] = clamp(finalG, 0.0f, 1.0f);                                                                                             \n" \
"       p_Output[index + 2] = clamp(finalB, 0.0f, 1.0f);                                                                                             \n" \
"       p_Output[index + 3] = a;                                                                                                                      \n" \
"   }                                                                                                                                                \n" \
"}                                                                                                                                                   \n";

std::mutex s_PipelineQueueMutex;
typedef std::unordered_map<id<MTLCommandQueue>, id<MTLComputePipelineState>> PipelineQueueMap;
PipelineQueueMap s_PipelineQueueMap;

void RunComplexMetalKernel(void* p_CmdQ, int p_Width, int p_Height, 
                          float* p_RgbGammas, float* p_CymGammas, 
                          float* p_RgbMidgreys, float* p_CymMidgreys, 
                          float* p_Tilts,
                          const float* p_Input, float* p_Output)
{
    const char* kernelName = "ContrastSatVolumeKernel";

    id<MTLCommandQueue>            queue = static_cast<id<MTLCommandQueue> >(p_CmdQ);
    id<MTLDevice>                  device = queue.device;
    id<MTLLibrary>                 metalLibrary;     // Metal library
    id<MTLFunction>                kernelFunction;   // Compute kernel
    id<MTLComputePipelineState>    pipelineState;    // Metal pipeline
    NSError* err;

    std::unique_lock<std::mutex> lock(s_PipelineQueueMutex);

    const auto it = s_PipelineQueueMap.find(queue);
    if (it == s_PipelineQueueMap.end())
    {
        MTLCompileOptions* options = [MTLCompileOptions new];

        // Replace the deprecated fastMathEnabled
        #if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
            // macOS 15.0+ - use the new mathMode property
            options.mathMode = MTLMathModeFast;  // Fixed: was MTLCompileMathModeFast
        #else
            // Earlier macOS versions - use the deprecated property
            options.fastMathEnabled = YES;
        #endif

        if (!(metalLibrary = [device newLibraryWithSource:@(kernelSource) options:options error:&err]))
        {
            fprintf(stderr, "Failed to load metal library, %s\n", err.localizedDescription.UTF8String);
            return;
        }
        [options release];
        if (!(kernelFunction  = [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:kernelName]]))
        {
            fprintf(stderr, "Failed to retrieve kernel\n");
            [metalLibrary release];
            return;
        }
        if (!(pipelineState   = [device newComputePipelineStateWithFunction:kernelFunction error:&err]))
        {
            fprintf(stderr, "Unable to compile, %s\n", err.localizedDescription.UTF8String);
            [metalLibrary release];
            [kernelFunction release];
            return;
        }

        s_PipelineQueueMap[queue] = pipelineState;

        //Release resources
        [metalLibrary release];
        [kernelFunction release];
    }
    else
    {
        pipelineState = it->second;
    }

    // Create Metal buffers for the parameter arrays
    id<MTLBuffer> rgbGammasBuf = [device newBufferWithBytes:p_RgbGammas length:3*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> cymGammasBuf = [device newBufferWithBytes:p_CymGammas length:3*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> rgbMidgreysBuf = [device newBufferWithBytes:p_RgbMidgreys length:3*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> cymMidgreysBuf = [device newBufferWithBytes:p_CymMidgreys length:3*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> tiltsBuf = [device newBufferWithBytes:p_Tilts length:3*sizeof(float) options:MTLResourceStorageModeShared];

    id<MTLBuffer> srcDeviceBuf = reinterpret_cast<id<MTLBuffer> >(const_cast<float *>(p_Input));
    id<MTLBuffer> dstDeviceBuf = reinterpret_cast<id<MTLBuffer> >(p_Output);

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    commandBuffer.label = [NSString stringWithFormat:@"ContrastSatVolumeKernel"];

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:pipelineState];

    int exeWidth = [pipelineState threadExecutionWidth];
    MTLSize threadGroupCount = MTLSizeMake(exeWidth, 1, 1);
    MTLSize threadGroups     = MTLSizeMake((p_Width + exeWidth - 1)/exeWidth, p_Height, 1);

    // Set input/output buffers
    [computeEncoder setBuffer:srcDeviceBuf offset: 0 atIndex: 0];
    [computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 8];
    
    // Set dimension parameters
    [computeEncoder setBytes:&p_Width length:sizeof(int) atIndex:11];
    [computeEncoder setBytes:&p_Height length:sizeof(int) atIndex:12];
    
    // Set parameter array buffers
    [computeEncoder setBuffer:rgbGammasBuf offset:0 atIndex:13];
    [computeEncoder setBuffer:cymGammasBuf offset:0 atIndex:14];
    [computeEncoder setBuffer:rgbMidgreysBuf offset:0 atIndex:15];
    [computeEncoder setBuffer:cymMidgreysBuf offset:0 atIndex:16];
    [computeEncoder setBuffer:tiltsBuf offset:0 atIndex:17];

    [computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

    [computeEncoder endEncoding];
    [commandBuffer commit];
    
    // Wait for completion before releasing buffers
    [commandBuffer waitUntilCompleted];
    
    // Release the temporary buffers
    [rgbGammasBuf release];
    [cymGammasBuf release];
    [rgbMidgreysBuf release];
    [cymMidgreysBuf release];
    [tiltsBuf release];
}

// Keep the original simple function for backward compatibility
void RunMetalKernel(void* p_CmdQ, int p_Width, int p_Height, float* p_Gain, const float* p_Input, float* p_Output)
{
    // Convert simple gain to complex parameters for backward compatibility
    float rgbGammas[3] = {p_Gain[0], p_Gain[1], p_Gain[2]};
    float cymGammas[3] = {1.0f, 1.0f, 1.0f};
    float rgbMidgreys[3] = {0.18f, 0.18f, 0.18f};
    float cymMidgreys[3] = {0.18f, 0.18f, 0.18f};
    float tilts[3] = {1.0f, 1.0f, 1.0f}; // Pure RGB mode
    
    RunComplexMetalKernel(p_CmdQ, p_Width, p_Height, 
                         rgbGammas, cymGammas, rgbMidgreys, cymMidgreys, tilts,
                         p_Input, p_Output);
}
