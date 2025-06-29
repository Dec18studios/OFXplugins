#import <Metal/Metal.h>

#include <unordered_map>
#include <mutex>
/* * This file is part of the Dummy Template OFX plugin.
 * It contains the Metal kernel code and the function to run it.
 * The kernel processes an image by extracting RGBA values and outputting them.
 * Make sure the kernel call here has the proper Parameters and indices set up correctly. in both the call and the kernel code
 */
const char* kernelSource =  \
"#include <metal_stdlib>\n" \
"using namespace metal;\n" \
"\n" \
"kernel void TemplateOFXKernel(constant int& p_Width [[buffer(11)]], constant int& p_Height [[buffer(12)]],\n" \
"                                const device float* p_Input [[buffer(0)]], device float* p_Output [[buffer(8)]],\n" \
"                                uint2 id [[thread_position_in_grid]])\n" \
"{\n" \
"    if ((id.x < p_Width) && (id.y < p_Height))\n" \
"    {\n" \
"        const int index = ((id.y * p_Width) + id.x) * 4;\n" \
"        \n" \
"        // Extract RGBA values\n" \
"        float r = p_Input[index + 0];\n" \
"        float g = p_Input[index + 1];\n" \
"        float b = p_Input[index + 2];\n" \
"        float a = p_Input[index + 3];\n" \

"        // Output to buffer\n" \
"        p_Output[index + 0] = r;\n" \
"        p_Output[index + 1] = g;\n" \
"        p_Output[index + 2] = b.z;\n" \
"        p_Output[index + 3] = a;\n" \
"    }\n" \
"}\n";

std::mutex s_PipelineQueueMutex;
typedef std::unordered_map<id<MTLCommandQueue>, id<MTLComputePipelineState>> PipelineQueueMap;
PipelineQueueMap s_PipelineQueueMap;

void RunMetalKernel(void* p_CmdQ, int p_Width, int p_Height,
                            const float* p_Input, float* p_Output)
{
    const char* kernelName = "TemplateOFXKernel";

    id<MTLCommandQueue> queue = static_cast<id<MTLCommandQueue>>(p_CmdQ);
    id<MTLDevice> device = queue.device;
    id<MTLLibrary> metalLibrary = nil;
    id<MTLFunction> kernelFunction = nil;
    id<MTLComputePipelineState> pipelineState = nil;
    NSError* err = nil;

    std::unique_lock<std::mutex> lock(s_PipelineQueueMutex);

    auto it = s_PipelineQueueMap.find(queue);
    if (it == s_PipelineQueueMap.end()) {
        MTLCompileOptions* options = [MTLCompileOptions new];
        #if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
            options.mathMode = MTLMathModeFast;
        #else
            options.fastMathEnabled = YES;
        #endif

        metalLibrary = [device newLibraryWithSource:@(kernelSource) options:options error:&err];
        [options release];
        if (!metalLibrary) {
            fprintf(stderr, "Failed to compile Metal library: %s\n", err.localizedDescription.UTF8String);
            return;
        }

        kernelFunction = [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:kernelName]];
        if (!kernelFunction) {
            fprintf(stderr, "Failed to get Metal kernel function.\n");
            [metalLibrary release];
            return;
        }

        pipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&err];
        if (!pipelineState) {
            fprintf(stderr, "Failed to create Metal pipeline: %s\n", err.localizedDescription.UTF8String);
            [metalLibrary release];
            [kernelFunction release];
            return;
        }

        s_PipelineQueueMap[queue] = pipelineState;
        [metalLibrary release];
        [kernelFunction release];
    } else {
        pipelineState = it->second;
    }

    lock.unlock();

    // Create Metal buffers for the preset arrays (exactly like Gamma Contrast approach)
    id<MTLBuffer> negativePresetBuf = [device newBufferWithBytes:negativePreset 
                                                          length:22*sizeof(float) 
                                                         options:MTLResourceStorageModeShared];
    id<MTLBuffer> printPresetBuf = [device newBufferWithBytes:printPreset 
                                                       length:22*sizeof(float) 
                                                      options:MTLResourceStorageModeShared];

    // Use the EXACT SAME pattern as Gamma Contrast for input/output buffers
    id<MTLBuffer> srcDeviceBuf = reinterpret_cast<id<MTLBuffer>>(const_cast<float*>(p_Input));
    id<MTLBuffer> dstDeviceBuf = reinterpret_cast<id<MTLBuffer>>(p_Output);

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    commandBuffer.label = @"TemplateOFXKernel";

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:pipelineState];

    int exeWidth = [pipelineState threadExecutionWidth];
    MTLSize threadGroupCount = MTLSizeMake(exeWidth, 1, 1);
    MTLSize threadGroups = MTLSizeMake((p_Width + exeWidth - 1)/exeWidth, p_Height, 1);
    int alphaPassThruInt = alphaPassThru ? 1 : 0;
    // Set buffers using the EXACT SAME indices as
    [computeEncoder setBuffer:srcDeviceBuf offset:0 atIndex:0];
    [computeEncoder setBuffer:dstDeviceBuf offset:0 atIndex:8];
    
    // Set dimension parameters
    [computeEncoder setBytes:&p_Width length:sizeof(int) atIndex:11];
    [computeEncoder setBytes:&p_Height length:sizeof(int) atIndex:12];
     
    ////////////////////////
    //Set your arrays and parameters to buffers  
      /* 
    // Set your preset array buffers
    [computeEncoder setBuffer:negativePresetBuf offset:0 atIndex:13];
    [computeEncoder setBuffer:printPresetBuf offset:0 atIndex:14];
  // Set additional parameters
 [computeEncoder setBytes:&alphaPassThruInt length:sizeof(int) atIndex:15];
    [computeEncoder setBytes:&alphaMin length:sizeof(float) atIndex:16];
    [computeEncoder setBytes:&alphaMax length:sizeof(float) atIndex:17];
    [computeEncoder setBytes:&linearAdjustment length:sizeof(float) atIndex:18];
    */
    [computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup:threadGroupCount];
    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // Release the temporary buffers
    [negativePresetBuf release];
    [printPresetBuf release];
}