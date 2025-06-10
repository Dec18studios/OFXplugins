#import <Metal/Metal.h>

#include <unordered_map>
#include <mutex>

// Add extern "C" declaration
extern "C" {
    void RunMetalKernel(void* p_CmdQ, int p_Width, int p_Height,
                       float p_Range, float p_TargetHueRange, float p_PushStrength, float p_Bend,
                       float* p_OriginalColor, float* p_TargetColor,
                       const float* p_Input, float* p_Output);
                       
    void RunMultiMetalKernel(void* p_CmdQ, int p_Width, int p_Height,
                            int p_MappingCount, float* p_MappingData,
                            const float* p_Input, float* p_Output);
}

const char* kernelSource =  \
"#include <metal_stdlib>\n" \
"using namespace metal;\n" \
"\n" \
"// RGB to HSV conversion\n" \
"float3 rgb_to_hsv(float3 rgb) {\n" \
"    float cmax = max(max(rgb.r, rgb.g), rgb.b);\n" \
"    float cmin = min(min(rgb.r, rgb.g), rgb.b);\n" \
"    float delta = cmax - cmin;\n" \
"    \n" \
"    float3 hsv;\n" \
"    \n" \
"    // Hue\n" \
"    if (delta == 0.0) {\n" \
"        hsv.x = 0.0;\n" \
"    } else if (cmax == rgb.r) {\n" \
"        hsv.x = fmod((rgb.g - rgb.b) / delta + 6.0, 6.0) / 6.0;\n" \
"    } else if (cmax == rgb.g) {\n" \
"        hsv.x = ((rgb.b - rgb.r) / delta + 2.0) / 6.0;\n" \
"    } else {\n" \
"        hsv.x = ((rgb.r - rgb.g) / delta + 4.0) / 6.0;\n" \
"    }\n" \
"    \n" \
"    // Saturation\n" \
"    hsv.y = (cmax == 0.0) ? 0.0 : delta / cmax;\n" \
"    \n" \
"    // Value\n" \
"    hsv.z = cmax;\n" \
"    \n" \
"    return hsv;\n" \
"}\n" \
"\n" \
"// HSV to RGB conversion\n" \
"float3 hsv_to_rgb(float3 hsv) {\n" \
"    float c = hsv.z * hsv.y;\n" \
"    float x = c * (1.0 - abs(fmod(hsv.x * 6.0, 2.0) - 1.0));\n" \
"    float m = hsv.z - c;\n" \
"    \n" \
"    float3 rgb;\n" \
"    float h_sector = hsv.x * 6.0;\n" \
"    \n" \
"    if (h_sector < 1.0) {\n" \
"        rgb = float3(c, x, 0.0);\n" \
"    } else if (h_sector < 2.0) {\n" \
"        rgb = float3(x, c, 0.0);\n" \
"    } else if (h_sector < 3.0) {\n" \
"        rgb = float3(0.0, c, x);\n" \
"    } else if (h_sector < 4.0) {\n" \
"        rgb = float3(0.0, x, c);\n" \
"    } else if (h_sector < 5.0) {\n" \
"        rgb = float3(x, 0.0, c);\n" \
"    } else {\n" \
"        rgb = float3(c, 0.0, x);\n" \
"    }\n" \
"    \n" \
"    return rgb + m;\n" \
"}\n" \
"\n" \
"// Calculate shortest angular distance between two hues (0-1 range)\n" \
"float hue_distance(float h1, float h2) {\n" \
"    float diff = abs(h1 - h2);\n" \
"    return min(diff, 1.0 - diff);\n" \
"}\n" \
"\n" \
"// Calculate shortest angular direction from h1 to h2\n" \
"float hue_direction(float from_hue, float to_hue) {\n" \
"    float diff = to_hue - from_hue;\n" \
"    if (diff > 0.5) {\n" \
"        return diff - 1.0;\n" \
"    } else if (diff < -0.5) {\n" \
"        return diff + 1.0;\n" \
"    }\n" \
"    return diff;\n" \
"}\n" \
"\n" \
"// Smooth falloff function (smoothstep with customizable range)\n" \
"float smooth_falloff(float distance, float range) {\n" \
"    if (range <= 0.0) return (distance == 0.0) ? 1.0 : 0.0;\n" \
"    float normalized = clamp(distance / range, 0.0, 1.0);\n" \
"    return 1.0 - smoothstep(0.0, 1.0, normalized);\n" \
"}\n" \
"\n" \
"// Alternative Gaussian falloff (smoother but more expensive)\n" \
"float gaussian_falloff(float distance, float range) {\n" \
"    if (range <= 0.0) return (distance == 0.0) ? 1.0 : 0.0;\n" \
"    float sigma = range * 0.5;\n" \
"    return exp(-(distance * distance) / (2.0 * sigma * sigma));\n" \
"}\n" \
"\n" \
"// Enhanced hue mapping with target softness\n" \
"float apply_hue_mapping_with_target_softness(float currentHue, float srcHue, float srcRange, \n" \
"                                            float targetHue, float targetRange, \n" \
"                                            float strength, float bend) {\n" \
"    // Calculate hue distance from source hue\n" \
"    float hueDistance = hue_distance(currentHue, srcHue);\n" \
"    \n" \
"    // Calculate falloff weight based on distance from source and range\n" \
"    float sourceFalloff = gaussian_falloff(hueDistance, srcRange);\n" \
"    \n" \
"    if (sourceFalloff > 0.001) {\n" \
"        // Calculate the shortest direction to target hue\n" \
"        float hueShift = hue_direction(currentHue, targetHue);\n" \
"        \n" \
"        // Apply strength and bend curve to source falloff\n" \
"        float effectiveStrength = sourceFalloff * strength;\n" \
"        \n" \
"        // Apply bend curve (power function)\n" \
"        if (bend != 1.0) {\n" \
"            effectiveStrength = pow(effectiveStrength, bend);\n" \
"        }\n" \
"        \n" \
"        // Calculate intermediate hue position\n" \
"        float intermediateHue = currentHue + (hueShift * effectiveStrength);\n" \
"        intermediateHue = fmod(intermediateHue + 1.0, 1.0);\n" \
"        \n" \
"        // Apply target range softness if specified\n" \
"        if (targetRange > 0.0) {\n" \
"            // Calculate distance from the target hue\n" \
"            float targetDistance = hue_distance(intermediateHue, targetHue);\n" \
"            \n" \
"            // Create a soft transition zone around the target\n" \
"            float targetSoftness = gaussian_falloff(targetDistance, targetRange);\n" \
"            \n" \
"            // Blend between the intermediate result and exact target based on softness\n" \
"            // When close to target (high softness), use more of the intermediate result\n" \
"            // When far from target (low softness), push more toward exact target\n" \
"            float blendFactor = 0.5 + (targetSoftness * 0.5); // Range: 0.5 to 1.0\n" \
"            \n" \
"            // Calculate the exact target shift\n" \
"            float exactTargetShift = hue_direction(currentHue, targetHue);\n" \
"            float exactTargetHue = currentHue + (exactTargetShift * effectiveStrength);\n" \
"            exactTargetHue = fmod(exactTargetHue + 1.0, 1.0);\n" \
"            \n" \
"            // Blend between soft transition and exact target\n" \
"            float finalShift = hue_direction(currentHue, intermediateHue) * blendFactor + \n" \
"                              hue_direction(currentHue, exactTargetHue) * (1.0 - blendFactor);\n" \
"            \n" \
"            return currentHue + finalShift;\n" \
"        } else {\n" \
"            // No target softness, return intermediate result\n" \
"            return intermediateHue;\n" \
"        }\n" \
"    }\n" \
"    \n" \
"    return currentHue; // No change\n" \
"}\n" \
"\n" \
"// Single hue mapping kernel (legacy support)\n" \
"kernel void HueWarpKernel(constant int& p_Width [[buffer(11)]], constant int& p_Height [[buffer(12)]],\n" \
"                          constant float& p_Range [[buffer(13)]], constant float& p_TargetHueRange [[buffer(14)]],\n" \
"                          constant float& p_PushStrength [[buffer(15)]], constant float& p_Bend [[buffer(16)]],\n" \
"                          constant float* p_OriginalColor [[buffer(17)]], constant float* p_TargetColor [[buffer(18)]],\n" \
"                          const device float* p_Input [[buffer(0)]], device float* p_Output [[buffer(8)]],\n" \
"                          uint2 id [[thread_position_in_grid]])\n" \
"{\n" \
"    if ((id.x < p_Width) && (id.y < p_Height))\n" \
"    {\n" \
"        const int index = ((id.y * p_Width) + id.x) * 4;\n" \
"        \n" \
"        // Extract RGBA values\n" \
"        float3 inputColor = float3(p_Input[index + 0], p_Input[index + 1], p_Input[index + 2]);\n" \
"        float alpha = p_Input[index + 3];\n" \
"        \n" \
"        // Extract source and target hues from color arrays\n" \
"        float3 srcRGB = float3(p_OriginalColor[0], p_OriginalColor[1], p_OriginalColor[2]);\n" \
"        float3 targetRGB = float3(p_TargetColor[0], p_TargetColor[1], p_TargetColor[2]);\n" \
"        \n" \
"        // Convert source and target colors to get their hues\n" \
"        float3 srcHSV = rgb_to_hsv(srcRGB);\n" \
"        float3 targetHSV = rgb_to_hsv(targetRGB);\n" \
"        float srcHue = srcHSV.x;\n" \
"        float targetHue = targetHSV.x;\n" \
"        \n" \
"        // Convert input pixel to HSV\n" \
"        float3 hsv = rgb_to_hsv(inputColor);\n" \
"        \n" \
"        // Skip processing if pixel has no saturation (grayscale)\n" \
"        if (hsv.y > 0.001) {\n" \
"            // Apply hue mapping with target softness\n" \
"            float newHue = apply_hue_mapping_with_target_softness(hsv.x, srcHue, p_Range, \n" \
"                                                                 targetHue, p_TargetHueRange, \n" \
"                                                                 p_PushStrength, p_Bend);\n" \
"            \n" \
"            // Wrap hue to [0, 1] range\n" \
"            hsv.x = fmod(newHue + 1.0, 1.0);\n" \
"        }\n" \
"        \n" \
"        // Convert back to RGB\n" \
"        float3 outputColor = hsv_to_rgb(hsv);\n" \
"        \n" \
"        // Clamp values to valid range\n" \
"        outputColor = clamp(outputColor, 0.0, 1.0);\n" \
"        \n" \
"        // Output processed values\n" \
"        p_Output[index + 0] = outputColor.r;\n" \
"        p_Output[index + 1] = outputColor.g;\n" \
"        p_Output[index + 2] = outputColor.b;\n" \
"        p_Output[index + 3] = alpha;\n" \
"    }\n" \
"}\n" \
"\n" \
"// Multi-hue mapping kernel with anchor support\n" \
"kernel void MultiHueWarpKernel(constant int& p_Width [[buffer(11)]], constant int& p_Height [[buffer(12)]],\n" \
"                               constant int& p_MappingCount [[buffer(13)]],\n" \
"                               constant float* p_MappingData [[buffer(14)]],\n" \
"                               const device float* p_Input [[buffer(0)]], device float* p_Output [[buffer(8)]],\n" \
"                               uint2 id [[thread_position_in_grid]])\n" \
"{\n" \
"    if ((id.x < p_Width) && (id.y < p_Height))\n" \
"    {\n" \
"        const int index = ((id.y * p_Width) + id.x) * 4;\n" \
"        \n" \
"        // Extract RGBA values\n" \
"        float3 inputColor = float3(p_Input[index + 0], p_Input[index + 1], p_Input[index + 2]);\n" \
"        float alpha = p_Input[index + 3];\n" \
"        \n" \
"        // Convert input pixel to HSV\n" \
"        float3 hsv = rgb_to_hsv(inputColor);\n" \
"        \n" \
"        // Skip processing if pixel has no saturation (grayscale)\n" \
"        if (hsv.y > 0.001) {\n" \
"            // STEP 1: Calculate total protection from anchored hue zones\n" \
"            float totalProtection = 0.0;\n" \
"            \n" \
"            for (int i = 0; i < p_MappingCount; ++i) {\n" \
"                int offset = i * 8; // 8 floats per mapping (added anchor)\n" \
"                \n" \
"                float srcHue = p_MappingData[offset + 0];\n" \
"                float srcRange = p_MappingData[offset + 1];\n" \
"                float targetHue = p_MappingData[offset + 2];\n" \
"                float targetRange = p_MappingData[offset + 3];\n" \
"                float strength = p_MappingData[offset + 4];\n" \
"                float bend = p_MappingData[offset + 5];\n" \
"                float enabled = p_MappingData[offset + 6];\n" \
"                float anchorHue = p_MappingData[offset + 7]; // NEW: Anchor flag\n" \
"                \n" \
"                // Skip if mapping is disabled\n" \
"                if (enabled < 0.5) continue;\n" \
"                \n" \
"                // Check if this is an anchor mapping\n" \
"                if (anchorHue > 0.5) {\n" \
"                    // Calculate distance from input hue to anchor's source hue\n" \
"                    float dist = hue_distance(hsv.x, srcHue);\n" \
"                    \n" \
"                    // Compute soft protection using smoothstep falloff\n" \
"                    // Protection is strongest at the center and falls off to the edge of srcRange\n" \
"                    float protection = 1.0 - smoothstep(0.0, srcRange, dist);\n" \
"                    \n" \
"                    // Scale protection by anchor strength\n" \
"                    totalProtection += protection * strength;\n" \
"                }\n" \
"            }\n" \
"            \n" \
"            // Clamp total protection to [0, 1] range\n" \
"            totalProtection = clamp(totalProtection, 0.0, 1.0);\n" \
"            \n" \
"            // STEP 2: Apply regular (non-anchor) mappings with protection scaling\n" \
"            for (int i = 0; i < p_MappingCount; ++i) {\n" \
"                int offset = i * 8; // 8 floats per mapping\n" \
"                \n" \
"                float srcHue = p_MappingData[offset + 0];\n" \
"                float srcRange = p_MappingData[offset + 1];\n" \
"                float targetHue = p_MappingData[offset + 2];\n" \
"                float targetRange = p_MappingData[offset + 3];\n" \
"                float strength = p_MappingData[offset + 4];\n" \
"                float bend = p_MappingData[offset + 5];\n" \
"                float enabled = p_MappingData[offset + 6];\n" \
"                float anchorHue = p_MappingData[offset + 7];\n" \
"                \n" \
"                // Skip if mapping is disabled\n" \
"                if (enabled < 0.5) continue;\n" \
"                \n" \
"                // FIXED: Skip anchor mappings in the application phase\n" \
"                // Anchors only provide protection, they don't move hues\n" \
"                if (anchorHue > 0.5) continue;\n" \
"                \n" \
"                // For regular mappings, reduce strength based on total protection\n" \
"                float adjustedStrength = strength * (1.0 - totalProtection);\n" \
"                \n" \
"                // Skip if adjusted strength is negligible\n" \
"                if (adjustedStrength < 0.001) continue;\n" \
"                \n" \
"                // Apply hue mapping with adjusted strength\n" \
"                float newHue = apply_hue_mapping_with_target_softness(hsv.x, srcHue, srcRange, \n" \
"                                                                     targetHue, targetRange, \n" \
"                                                                     adjustedStrength, bend);\n" \
"                \n" \
"                // Wrap hue to [0, 1] range and update for next iteration\n" \
"                hsv.x = fmod(newHue + 1.0, 1.0);\n" \
"            }\n" \
"        }\n" \
"        \n" \
"        // Convert back to RGB\n" \
"        float3 outputColor = hsv_to_rgb(hsv);\n" \
"        \n" \
"        // Clamp values to valid range\n" \
"        outputColor = clamp(outputColor, 0.0, 1.0);\n" \
"        \n" \
"        // Output processed values\n" \
"        p_Output[index + 0] = outputColor.r;\n" \
"        p_Output[index + 1] = outputColor.g;\n" \
"        p_Output[index + 2] = outputColor.b;\n" \
"        p_Output[index + 3] = alpha;\n" \
"    }\n" \
"}\n";

std::mutex s_PipelineQueueMutex;
typedef std::unordered_map<id<MTLCommandQueue>, id<MTLComputePipelineState>> PipelineQueueMap;
PipelineQueueMap s_PipelineQueueMap;

void RunMetalKernel(void* p_CmdQ, int p_Width, int p_Height,
                   float p_Range, float p_TargetHueRange, float p_PushStrength, float p_Bend,
                   float* p_OriginalColor, float* p_TargetColor,
                   const float* p_Input, float* p_Output)
{
    const char* kernelName = "HueWarpKernel";

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
            options.mathMode = MTLMathModeFast;
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

    lock.unlock();

    // Create Metal buffers for the parameter arrays
    id<MTLBuffer> originalColorBuf = [device newBufferWithBytes:p_OriginalColor length:3*sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> targetColorBuf = [device newBufferWithBytes:p_TargetColor length:3*sizeof(float) options:MTLResourceStorageModeShared];

    id<MTLBuffer> srcDeviceBuf = reinterpret_cast<id<MTLBuffer> >(const_cast<float *>(p_Input));
    id<MTLBuffer> dstDeviceBuf = reinterpret_cast<id<MTLBuffer> >(p_Output);

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    commandBuffer.label = [NSString stringWithFormat:@"HueWarpKernel"];

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
    
    // Set scalar parameters
    [computeEncoder setBytes:&p_Range length:sizeof(float) atIndex:13];
    [computeEncoder setBytes:&p_TargetHueRange length:sizeof(float) atIndex:14];
    [computeEncoder setBytes:&p_PushStrength length:sizeof(float) atIndex:15];
    [computeEncoder setBytes:&p_Bend length:sizeof(float) atIndex:16];
    
    // Set color array buffers
    [computeEncoder setBuffer:originalColorBuf offset:0 atIndex:17];
    [computeEncoder setBuffer:targetColorBuf offset:0 atIndex:18];

    [computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

    [computeEncoder endEncoding];
    [commandBuffer commit];
    
    // Release the temporary buffers
    [originalColorBuf release];
    [targetColorBuf release];
}

void RunMultiMetalKernel(void* p_CmdQ, int p_Width, int p_Height,
                        int p_MappingCount, float* p_MappingData,
                        const float* p_Input, float* p_Output)
{
    const char* kernelName = "MultiHueWarpKernel";

    id<MTLCommandQueue>            queue = static_cast<id<MTLCommandQueue> >(p_CmdQ);
    id<MTLDevice>                  device = queue.device;
    id<MTLLibrary>                 metalLibrary;     // Metal library
    id<MTLFunction>                kernelFunction;   // Compute kernel
    id<MTLComputePipelineState>    pipelineState;    // Metal pipeline
    NSError* err;

    std::unique_lock<std::mutex> lock(s_PipelineQueueMutex);

    // Use a different key for multi-kernel pipeline
    void* multiKey = (void*)((uintptr_t)p_CmdQ + 1);
    const auto it = s_PipelineQueueMap.find(static_cast<id<MTLCommandQueue>>(multiKey));
    if (it == s_PipelineQueueMap.end())
    {
        MTLCompileOptions* options = [MTLCompileOptions new];

        #if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
            options.mathMode = MTLMathModeFast;
        #else
            options.fastMathEnabled = YES;
        #endif

        if (!(metalLibrary = [device newLibraryWithSource:@(kernelSource) options:options error:&err]))
        {
            fprintf(stderr, "Failed to load metal library for multi-kernel, %s\n", err.localizedDescription.UTF8String);
            return;
        }
        [options release];
        if (!(kernelFunction  = [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:kernelName]]))
        {
            fprintf(stderr, "Failed to retrieve multi-kernel\n");
            [metalLibrary release];
            return;
        }
        if (!(pipelineState   = [device newComputePipelineStateWithFunction:kernelFunction error:&err]))
        {
            fprintf(stderr, "Unable to compile multi-kernel, %s\n", err.localizedDescription.UTF8String);
            [metalLibrary release];
            [kernelFunction release];
            return;
        }

        s_PipelineQueueMap[static_cast<id<MTLCommandQueue>>(multiKey)] = pipelineState;

        //Release resources
        [metalLibrary release];
        [kernelFunction release];
    }
    else
    {
        pipelineState = it->second;
    }

    lock.unlock();

    // Create Metal buffer for the mapping data (now 8 floats per mapping)
    size_t mappingDataSize = p_MappingCount * 8 * sizeof(float);  // Updated from 7 to 8
    id<MTLBuffer> mappingDataBuf = [device newBufferWithBytes:p_MappingData length:mappingDataSize options:MTLResourceStorageModeShared];

    id<MTLBuffer> srcDeviceBuf = reinterpret_cast<id<MTLBuffer> >(const_cast<float *>(p_Input));
    id<MTLBuffer> dstDeviceBuf = reinterpret_cast<id<MTLBuffer> >(p_Output);

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    commandBuffer.label = [NSString stringWithFormat:@"MultiHueWarpKernel"];

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:pipelineState];

    int exeWidth = [pipelineState threadExecutionWidth];
    MTLSize threadGroupCount = MTLSizeMake(exeWidth, 1, 1);
    MTLSize threadGroups     = MTLSizeMake((p_Width + exeWidth - 1)/exeWidth, p_Height, 1);

    // Set input/output buffers
    [computeEncoder setBuffer:srcDeviceBuf offset: 0 atIndex: 0];
    [computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 8];
    
    // Set parameters for multi-mapping kernel
    [computeEncoder setBytes:&p_Width length:sizeof(int) atIndex:11];
    [computeEncoder setBytes:&p_Height length:sizeof(int) atIndex:12];
    [computeEncoder setBytes:&p_MappingCount length:sizeof(int) atIndex:13];
    [computeEncoder setBuffer:mappingDataBuf offset:0 atIndex:14];

    [computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

    [computeEncoder endEncoding];
    [commandBuffer commit];
    
    // Release the temporary buffer
    [mappingDataBuf release];
}


