#ifdef __APPLE__

#include "FilmProjectorMetal.h"
#include "FilmProjectorPresets.h"

void RunFilmProjectorMetalKernel(void* p_CmdQ, int p_Width, int p_Height,
                                float* p_NegativePreset, float* p_PrintPreset,
                                int p_Observer, int p_Mode, int p_LayerMode,
                                bool p_AlphaPassThru, float p_LinearAdjustment,
                                float p_GrainProbability, float p_GrainStrength, float p_GrainSeeth,
                                float p_HaloCutoff, float p_HaloRange, float p_HaloPressure,
                                const float* p_Input, float* p_Output)
{
    id<MTLCommandQueue> queue = (id<MTLCommandQueue>)p_CmdQ;
    id<MTLDevice> device = queue.device;
    
    // Create library and kernel
    NSError* error = nil;
    id<MTLLibrary> library = [device newDefaultLibrary];
    id<MTLFunction> kernelFunction = [library newFunctionWithName:@"FilmProjectorKernel"];
    id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
    
    // Create textures
    MTLTextureDescriptor* textureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                                   width:p_Width
                                                                                                  height:p_Height
                                                                                               mipmapped:NO];
    textureDescriptor.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> inputTexture = [device newTextureWithDescriptor:textureDescriptor];
    
    textureDescriptor.usage = MTLTextureUsageShaderWrite;
    id<MTLTexture> outputTexture = [device newTextureWithDescriptor:textureDescriptor];
    
    // Upload input data
    MTLRegion region = MTLRegionMake2D(0, 0, p_Width, p_Height);
    [inputTexture replaceRegion:region mipmapLevel:0 withBytes:p_Input bytesPerRow:p_Width * 4 * sizeof(float)];
    
    // Create buffers for presets and parameters
    id<MTLBuffer> negativeBuffer = [device newBufferWithBytes:p_NegativePreset length:22 * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> printBuffer = [device newBufferWithBytes:p_PrintPreset length:22 * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> observerBuffer = [device newBufferWithBytes:&p_Observer length:sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> modeBuffer = [device newBufferWithBytes:&p_Mode length:sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> layerModeBuffer = [device newBufferWithBytes:&p_LayerMode length:sizeof(int) options:MTLResourceStorageModeShared];
    id<MTLBuffer> alphaPassThruBuffer = [device newBufferWithBytes:&p_AlphaPassThru length:sizeof(bool) options:MTLResourceStorageModeShared];
    id<MTLBuffer> linearAdjustmentBuffer = [device newBufferWithBytes:&p_LinearAdjustment length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> grainProbabilityBuffer = [device newBufferWithBytes:&p_GrainProbability length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> grainStrengthBuffer = [device newBufferWithBytes:&p_GrainStrength length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> grainSeethBuffer = [device newBufferWithBytes:&p_GrainSeeth length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> haloCutoffBuffer = [device newBufferWithBytes:&p_HaloCutoff length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> haloRangeBuffer = [device newBufferWithBytes:&p_HaloRange length:sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> haloPressureBuffer = [device newBufferWithBytes:&p_HaloPressure length:sizeof(float) options:MTLResourceStorageModeShared];
    
    // Create command buffer and encoder
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    
    [encoder setComputePipelineState:pipelineState];
    [encoder setTexture:inputTexture atIndex:0];
    [encoder setTexture:outputTexture atIndex:1];
    [encoder setBuffer:negativeBuffer offset:0 atIndex:0];
    [encoder setBuffer:printBuffer offset:0 atIndex:1];
    [encoder setBuffer:observerBuffer offset:0 atIndex:2];
    [encoder setBuffer:modeBuffer offset:0 atIndex:3];
    [encoder setBuffer:layerModeBuffer offset:0 atIndex:4];
    [encoder setBuffer:alphaPassThruBuffer offset:0 atIndex:5];
    [encoder setBuffer:linearAdjustmentBuffer offset:0 atIndex:6];
    [encoder setBuffer:grainProbabilityBuffer offset:0 atIndex:7];
    [encoder setBuffer:grainStrengthBuffer offset:0 atIndex:8];
    [encoder setBuffer:grainSeethBuffer offset:0 atIndex:9];
    [encoder setBuffer:haloCutoffBuffer offset:0 atIndex:10];
    [encoder setBuffer:haloRangeBuffer offset:0 atIndex:11];
    [encoder setBuffer:haloPressureBuffer offset:0 atIndex:12];
    
    // Configure thread groups
    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    MTLSize threadgroupCount = MTLSizeMake((p_Width + threadgroupSize.width - 1) / threadgroupSize.width,
                                          (p_Height + threadgroupSize.height - 1) / threadgroupSize.height, 1);
    
    [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];
    
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    
    // Download result
    [outputTexture getBytes:p_Output bytesPerRow:p_Width * 4 * sizeof(float) fromRegion:region mipmapLevel:0];
}

#endif // __APPLE__