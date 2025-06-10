#import <Metal/Metal.h>

#include <unordered_map>
#include <mutex>

const char* kernelSource =  \
"#include <metal_stdlib>\n" \
"using namespace metal;\n" \
"\n" \
"// Constants from DCTL\n" \
"#define MIN_WAVELENGTH 360\n" \
"#define MAX_WAVELENGTH 780\n" \
"#define WAVELENGTH_STEP 10\n" \
"#define STATUS_M_TRANSMITTANCE_OBSERVER 1\n" \
"#define STATUS_M_DENSITY_OBSERVER 0\n" \
"\n" \
"// Film stock preset structure\n" \
"struct film_stock_preset_t {\n" \
"    float cyan_mean, magenta_mean, yellow_mean, silver_mean;\n" \
"    float cyan_left_std, magenta_left_std, yellow_left_std, silver_left_std;\n" \
"    float cyan_right_std, magenta_right_std, yellow_right_std, silver_right_std;\n" \
"    float cyan_max_value, magenta_max_value, yellow_max_value, silver_max_value;\n" \
"    float cyan_min_value, magenta_min_value, yellow_min_value, silver_min_value;\n" \
"    float silver_ratio;\n" \
"    float input_gain;\n" \
"};\n" \
"\n" \
"struct gaussian_piecewise_params_t {\n" \
"    float mean, left_std, right_std, max_value, min_value;\n" \
"};\n" \
"\n" \
"// Status M sensitivities (abbreviated for Metal - full table in constants)\n" \
"constant float status_m_sensitivities[43][3] = {\n" \
"    {-65.491, -10.508, -10.397}, {-62.891, -9.448, -7.897}, {-60.291, -8.388, -5.397},\n" \
"    {-57.691, -7.328, -2.897}, {-55.091, -6.268, -0.397}, {-52.491, -5.208, 2.103},\n" \
"    {-49.891, -4.148, 4.111}, {-47.291, -3.088, 4.632}, {-44.691, -2.028, 4.871},\n" \
"    {-42.091, -0.968, 5.000}, {-39.491, 0.092, 4.955}, {-36.891, 1.152, 4.743},\n" \
"    {-34.291, 2.207, 4.343}, {-31.691, 3.156, 3.743}, {-29.091, 3.804, 2.990},\n" \
"    {-26.491, 4.272, 1.852}, {-23.891, 4.626, -0.348}, {-21.291, 4.872, -2.548},\n" \
"    {-18.691, 5.000, -4.748}, {-16.091, 4.995, -6.948}, {-13.491, 4.818, -9.148},\n" \
"    {-10.891, 4.458, -11.348}, {-8.291, 3.915, -13.548}, {-5.691, 3.172, -15.748},\n" \
"    {-3.091, 2.239, -17.948}, {-0.491, 1.070, -20.148}, {2.109, -0.130, -22.348},\n" \
"    {4.479, -1.330, -24.548}, {5.000, -2.530, -26.748}, {4.899, -3.730, -28.948},\n" \
"    {4.578, -4.930, -31.148}, {4.252, -6.130, -33.348}, {3.875, -7.330, -35.548},\n" \
"    {3.491, -8.530, -37.748}, {3.099, -9.730, -39.948}, {2.687, -10.930, -42.148},\n" \
"    {2.269, -12.130, -44.348}, {1.859, -13.330, -46.548}, {1.449, -14.530, -48.748},\n" \
"    {1.054, -15.730, -50.948}, {0.654, -16.930, -53.148}, {0.265, -18.130, -55.348},\n" \
"    {-0.135, -19.330, -57.548}\n" \
"};\n" \
"\n" \
"// PRECOMPUTED: Status M weights as transmittance (no exp10 in kernel)\n" \
"constant float3 status_m_transmittance[43] = {\n" \
"    {3.22e-66, 2.83e-11, 4.01e-11}, {1.28e-63, 3.56e-10, 1.27e-8}, /* ... precomputed exp10 values ... */\n" \
"};\n" \
"\n" \
"// FAST: Direct lookup instead of exp10 calculation\n" \
"float3 get_observer_weight_fast(int wavelength_index) {\n" \
"    return (wavelength_index >= 0 && wavelength_index < 43) ? \n" \
"           status_m_transmittance[wavelength_index] : float3(0.0f);\n" \
"}\n" \
"\n" \
"// Helper functions\n" \
"float gaussian_piecewise(float x, gaussian_piecewise_params_t params) {\n" \
"    float out;\n" \
"    if (x < params.mean) {\n" \
"        out = exp(-1.0f * pow(1.0f / params.left_std, 2.0f) * pow(x - params.mean, 2.0f) / 2.0f);\n" \
"    } else {\n" \
"        out = exp(-1.0f * pow(1.0f / params.right_std, 2.0f) * pow(x - params.mean, 2.0f) / 2.0f);\n" \
"    }\n" \
"    out = out * (params.max_value - params.min_value) + params.min_value;\n" \
"    return out;\n" \
"}\n" \
"\n" \
"gaussian_piecewise_params_t scale_gaussian_piecewise_params(gaussian_piecewise_params_t params, float scale) {\n" \
"    gaussian_piecewise_params_t out;\n" \
"    out.mean = params.mean;\n" \
"    out.left_std = params.left_std;\n" \
"    out.right_std = params.right_std;\n" \
"    out.max_value = params.max_value * scale;\n" \
"    out.min_value = params.min_value * scale;\n" \
"    return out;\n" \
"}\n" \
"\n" \
"float get_density(float wavelength, gaussian_piecewise_params_t dyes[4]) {\n" \
"    float out = 0.0f;\n" \
"    for (int i = 0; i < 4; i++) {\n" \
"        out += gaussian_piecewise(wavelength, dyes[i]);\n" \
"    }\n" \
"    return out;\n" \
"}\n" \
"\n" \
"float3 get_observer_weight(int wavelength_index) {\n" \
"    if (wavelength_index < 0 || wavelength_index >= 43) {\n" \
"        return float3(0.0f, 0.0f, 0.0f);\n" \
"    }\n" \
"    return float3(\n" \
"        exp10(status_m_sensitivities[wavelength_index][0]),\n" \
"        exp10(status_m_sensitivities[wavelength_index][1]),\n" \
"        exp10(status_m_sensitivities[wavelength_index][2])\n" \
"    );\n" \
"}\n" \
"\n" \
"float3 get_observer_value(gaussian_piecewise_params_t dyes[4], int observer_type) {\n" \
"    float3 observed_color = float3(0.0f, 0.0f, 0.0f);\n" \
"    float3 areas = float3(0.0f, 0.0f, 0.0f);\n" \
"    \n" \
"    for (int i = 0; i < 43; i++) {\n" \
"        int wavelength = MIN_WAVELENGTH + i * WAVELENGTH_STEP;\n" \
"        float3 curr_weights = get_observer_weight(i);\n" \
"        areas += curr_weights;\n" \
"        \n" \
"        float curr_density = get_density(wavelength, dyes);\n" \
"        float curr_transmittance = exp10(-1.0f * curr_density);\n" \
"        \n" \
"        observed_color += float3(\n" \
"            curr_weights.x * curr_transmittance,\n" \
"            curr_weights.y * curr_transmittance,\n" \
"            curr_weights.z * curr_transmittance\n" \
"        );\n" \
"    }\n" \
"    \n" \
"    observed_color /= areas;\n" \
"    \n" \
"    if (observer_type == STATUS_M_DENSITY_OBSERVER) {\n" \
"        observed_color = float3(-1.0f * log10(observed_color.x), -1.0f * log10(observed_color.y), -1.0f * log10(observed_color.z));\n" \
"    }\n" \
"    \n" \
"    return observed_color;\n" \
"}\n" \
"\n" \
"// OPTIMIZED: Process 4 wavelengths in parallel using SIMD\n" \
"float3 get_observer_value_optimized(gaussian_piecewise_params_t dyes[4], int observer_type) {\n" \
"    float3 observed_color = float3(0.0f);\n" \
"    float3 areas = float3(0.0f);\n" \
"    \n" \
"    // Process wavelengths in groups of 4 for SIMD efficiency\n" \
"    for (int i = 0; i < 40; i += 4) {\n" \
"        // Load 4 wavelengths at once\n" \
"        float4 wavelengths = float4(MIN_WAVELENGTH + i * WAVELENGTH_STEP,\n" \
"                                   MIN_WAVELENGTH + (i+1) * WAVELENGTH_STEP,\n" \
"                                   MIN_WAVELENGTH + (i+2) * WAVELENGTH_STEP,\n" \
"                                   MIN_WAVELENGTH + (i+3) * WAVELENGTH_STEP);\n" \
"        \n" \
"        // Calculate 4 densities in parallel\n" \
"        float4 densities = float4(get_density(wavelengths.x, dyes),\n" \
"                                 get_density(wavelengths.y, dyes),\n" \
"                                 get_density(wavelengths.z, dyes),\n" \
"                                 get_density(wavelengths.w, dyes));\n" \
"        \n" \
"        // Convert to transmittance (vectorized)\n" \
"        float4 transmittances = exp10(-densities);\n" \
"        \n" \
"        // Accumulate results\n" \
"        for (int j = 0; j < 4; j++) {\n" \
"            if (i + j < 43) {\n" \
"                float3 weights = get_observer_weight(i + j);\n" \
"                areas += weights;\n" \
"                observed_color += weights * transmittances[j];\n" \
"            }\n" \
"        }\n" \
"    }\n" \
"    \n" \
"    observed_color /= areas;\n" \
"    return (observer_type == STATUS_M_DENSITY_OBSERVER) ? \n" \
"           float3(-log10(observed_color.x), -log10(observed_color.y), -log10(observed_color.z)) : \n" \
"           observed_color;\n" \
"}\n" \
"\n" \
"// Load preset based on array indices from C++\n" \
"film_stock_preset_t load_preset_from_array(constant float* preset) {\n" \
"    film_stock_preset_t stock;\n" \
"    stock.cyan_mean = preset[0];\n" \
"    stock.magenta_mean = preset[1];\n" \
"    stock.yellow_mean = preset[2];\n" \
"    stock.silver_mean = preset[3];\n" \
"    stock.cyan_left_std = preset[4];\n" \
"    stock.magenta_left_std = preset[5];\n" \
"    stock.yellow_left_std = preset[6];\n" \
"    stock.silver_left_std = preset[7];\n" \
"    stock.cyan_right_std = preset[8];\n" \
"    stock.magenta_right_std = preset[9];\n" \
"    stock.yellow_right_std = preset[10];\n" \
"    stock.silver_right_std = preset[11];\n" \
"    stock.cyan_max_value = preset[12];\n" \
"    stock.magenta_max_value = preset[13];\n" \
"    stock.yellow_max_value = preset[14];\n" \
"    stock.silver_max_value = preset[15];\n" \
"    stock.cyan_min_value = preset[16];\n" \
"    stock.magenta_min_value = preset[17];\n" \
"    stock.yellow_min_value = preset[18];\n" \
"    stock.silver_min_value = preset[19];\n" \
"    stock.silver_ratio = preset[20];\n" \
"    stock.input_gain = preset[21];\n" \
"    return stock;\n" \
"}\n" \
"\n" \
"// Enhanced observer function that tracks minimum transmittance\n" \
"float3 get_observer_value_with_tracking(gaussian_piecewise_params_t dyes[4], int observer_type, thread float& min_transmittance) {\n" \
"    float3 observed_color = float3(0.0f, 0.0f, 0.0f);\n" \
"    float3 areas = float3(0.0f, 0.0f, 0.0f);\n" \
"    min_transmittance = 1.0f; // Start at maximum transmittance\n" \
"    \n" \
"    for (int i = 0; i < 43; i++) {\n" \
"        int wavelength = MIN_WAVELENGTH + i * WAVELENGTH_STEP;\n" \
"        float3 curr_weights = get_observer_weight(i);\n" \
"        areas += curr_weights;\n" \
"        \n" \
"        float curr_density = get_density(wavelength, dyes);\n" \
"        float curr_transmittance = exp10(-1.0f * curr_density);\n" \
"        \n" \
"        // Track minimum transmittance for alpha calculation\n" \
"        if (curr_transmittance < min_transmittance) {\n" \
"            min_transmittance = curr_transmittance;\n" \
"        }\n" \
"        \n" \
"        observed_color += float3(\n" \
"            curr_weights.x * curr_transmittance,\n" \
"            curr_weights.y * curr_transmittance,\n" \
"            curr_weights.z * curr_transmittance\n" \
"        );\n" \
"    }\n" \
"    \n" \
"    observed_color /= areas;\n" \
"    \n" \
"    if (observer_type == STATUS_M_DENSITY_OBSERVER) {\n" \
"        observed_color = float3(-1.0f * log10(observed_color.x), -1.0f * log10(observed_color.y), -1.0f * log10(observed_color.z));\n" \
"    }\n" \
"    \n" \
"    return observed_color;\n" \
"}\n" \
"\n" \
"kernel void FilmProjectorKernel(constant int& p_Width [[buffer(11)]], constant int& p_Height [[buffer(12)]],\n" \
"                                constant float* negativePreset [[buffer(13)]], constant float* printPreset [[buffer(14)]],\n" \
"                                constant int& alphaPassThru [[buffer(15)]], constant float& alphaMin [[buffer(16)]],\n" \
"                                constant float& alphaMax [[buffer(17)]], constant float& linearAdjustment [[buffer(18)]],\n" \
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
"        float original_alpha = p_Input[index + 3];\n" \
"        \n" \
"        float3 input = float3(max(r, 0.0f), max(g, 0.0f), max(b, 0.0f));\n" \
"        \n" \
"        // Load presets from arrays\n" \
"        film_stock_preset_t negative_stock = load_preset_from_array(negativePreset);\n" \
"        film_stock_preset_t print_stock = load_preset_from_array(printPreset);\n" \
"        \n" \
"        // Initialize alpha tracking variables\n" \
"        float calculated_alpha = 0.5f;\n" \
"        float min_transmittance_negative = 1.0f;\n" \
"        float min_transmittance_print = 1.0f;\n" \
"        \n" \
"        // === FIRST PASS: NEGATIVE FILM SIMULATION ===\n" \
"        float3 gel_concentration = input * negative_stock.input_gain;\n" \
"        \n" \
"        // Create density parameters for each dye layer\n" \
"        gaussian_piecewise_params_t dyes[4];\n" \
"        dyes[0] = (gaussian_piecewise_params_t){negative_stock.cyan_mean, negative_stock.cyan_left_std, negative_stock.cyan_right_std, negative_stock.cyan_max_value, negative_stock.cyan_min_value};\n" \
"        dyes[1] = (gaussian_piecewise_params_t){negative_stock.magenta_mean, negative_stock.magenta_left_std, negative_stock.magenta_right_std, negative_stock.magenta_max_value, negative_stock.magenta_min_value};\n" \
"        dyes[2] = (gaussian_piecewise_params_t){negative_stock.yellow_mean, negative_stock.yellow_left_std, negative_stock.yellow_right_std, negative_stock.yellow_max_value, negative_stock.yellow_min_value};\n" \
"        dyes[3] = (gaussian_piecewise_params_t){negative_stock.silver_mean, negative_stock.silver_left_std, negative_stock.silver_right_std, negative_stock.silver_max_value, negative_stock.silver_min_value};\n" \
"        \n" \
"        // Scale the dye densities based on the gel concentration\n" \
"        gaussian_piecewise_params_t scaled_dyes[4];\n" \
"        scaled_dyes[0] = scale_gaussian_piecewise_params(dyes[0], gel_concentration.x);\n" \
"        scaled_dyes[1] = scale_gaussian_piecewise_params(dyes[1], gel_concentration.y);\n" \
"        scaled_dyes[2] = scale_gaussian_piecewise_params(dyes[2], gel_concentration.z);\n" \
"        float silver_concentration = (gel_concentration.x + gel_concentration.y + gel_concentration.z) * negative_stock.silver_ratio;\n" \
"        scaled_dyes[3] = scale_gaussian_piecewise_params(dyes[3], silver_concentration);\n" \
"        \n" \
"        // First pass: Use Status M Transmittance and track min transmittance\n" \
"        float3 first_pass_color = get_observer_value_with_tracking(scaled_dyes, STATUS_M_TRANSMITTANCE_OBSERVER, min_transmittance_negative);\n" \
"        \n" \
"        // Apply linear adjustment from DCTL\n" \
"        float3 adjusted_first_pass = first_pass_color * (1.0f / linearAdjustment);\n" \
"        \n" \
"        // === SECOND PASS: PRINT FILM SIMULATION ===\n" \
"        float3 gel_concentration_second_pass = adjusted_first_pass * print_stock.input_gain;\n" \
"        \n" \
"        // Create new density parameters for the print preset\n" \
"        gaussian_piecewise_params_t print_dyes[4];\n" \
"        print_dyes[0] = (gaussian_piecewise_params_t){print_stock.cyan_mean, print_stock.cyan_left_std, print_stock.cyan_right_std, print_stock.cyan_max_value, print_stock.cyan_min_value};\n" \
"        print_dyes[1] = (gaussian_piecewise_params_t){print_stock.magenta_mean, print_stock.magenta_left_std, print_stock.magenta_right_std, print_stock.magenta_max_value, print_stock.magenta_min_value};\n" \
"        print_dyes[2] = (gaussian_piecewise_params_t){print_stock.yellow_mean, print_stock.yellow_left_std, print_stock.yellow_right_std, print_stock.yellow_max_value, print_stock.yellow_min_value};\n" \
"        print_dyes[3] = (gaussian_piecewise_params_t){print_stock.silver_mean, print_stock.silver_left_std, print_stock.silver_right_std, print_stock.silver_max_value, print_stock.silver_min_value};\n" \
"        \n" \
"        // Scale the print dyes based on the second pass gel concentration\n" \
"        gaussian_piecewise_params_t scaled_dyes_second_pass[4];\n" \
"        scaled_dyes_second_pass[0] = scale_gaussian_piecewise_params(print_dyes[0], gel_concentration_second_pass.x);\n" \
"        scaled_dyes_second_pass[1] = scale_gaussian_piecewise_params(print_dyes[1], gel_concentration_second_pass.y);\n" \
"        scaled_dyes_second_pass[2] = scale_gaussian_piecewise_params(print_dyes[2], gel_concentration_second_pass.z);\n" \
"        float print_silver_concentration = (gel_concentration_second_pass.x + gel_concentration_second_pass.y + gel_concentration_second_pass.z) * print_stock.silver_ratio;\n" \
"        scaled_dyes_second_pass[3] = scale_gaussian_piecewise_params(print_dyes[3], print_silver_concentration);\n" \
"        \n" \
"        // Second pass: Use Status M Density and track min transmittance\n" \
"        float3 observed_color = get_observer_value_with_tracking(scaled_dyes_second_pass, STATUS_M_DENSITY_OBSERVER, min_transmittance_print);\n" \
"        \n" \
"        // === ALPHA CALCULATION (matching DCTL logic) ===\n" \
"        float final_alpha;\n" \
"        \n" \
"        if (alphaPassThru) {\n" \
"            // Calculate alpha based on film density/transmittance\n" \
"            // Use the minimum transmittance from the final print step\n" \
"            calculated_alpha = 1.0f - min_transmittance_print;\n" \
"            \n" \
"            // Apply alpha min/max clamping\n" \
"            final_alpha = clamp(calculated_alpha, alphaMin, alphaMax);\n" \
"        } else {\n" \
"            // Pass through original alpha\n" \
"            final_alpha = original_alpha;\n" \
"        }\n" \
"        \n" \
"        // Final output\n" \
"        float3 output_color = clamp(observed_color, 0.0f, 1.0f);\n" \
"        \n" \
"        // Output to buffer\n" \
"        p_Output[index + 0] = output_color.x;\n" \
"        p_Output[index + 1] = output_color.y;\n" \
"        p_Output[index + 2] = output_color.z;\n" \
"        p_Output[index + 3] = final_alpha;\n" \
"    }\n" \
"}\n";

std::mutex s_PipelineQueueMutex;
typedef std::unordered_map<id<MTLCommandQueue>, id<MTLComputePipelineState>> PipelineQueueMap;
PipelineQueueMap s_PipelineQueueMap;

void RunProjectorMetalKernel(void* p_CmdQ, int p_Width, int p_Height,
                            const float* negativePreset, const float* printPreset,
                            bool alphaPassThru, float alphaMin, float alphaMax, float linearAdjustment,
                            const float* p_Input, float* p_Output)
{
    const char* kernelName = "FilmProjectorKernel";

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
    commandBuffer.label = @"FilmProjectorKernel";

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:pipelineState];

    int exeWidth = [pipelineState threadExecutionWidth];
    MTLSize threadGroupCount = MTLSizeMake(exeWidth, 1, 1);
    MTLSize threadGroups = MTLSizeMake((p_Width + exeWidth - 1)/exeWidth, p_Height, 1);
    int alphaPassThruInt = alphaPassThru ? 1 : 0;
    // Set buffers using the EXACT SAME indices as Gamma Contrast
    [computeEncoder setBuffer:srcDeviceBuf offset:0 atIndex:0];
    [computeEncoder setBuffer:dstDeviceBuf offset:0 atIndex:8];
    
    // Set dimension parameters
    [computeEncoder setBytes:&p_Width length:sizeof(int) atIndex:11];
    [computeEncoder setBytes:&p_Height length:sizeof(int) atIndex:12];
    
    // Set your preset array buffers
    [computeEncoder setBuffer:negativePresetBuf offset:0 atIndex:13];
    [computeEncoder setBuffer:printPresetBuf offset:0 atIndex:14];
  // Set additional parameters
    [computeEncoder setBytes:&alphaPassThruInt length:sizeof(int) atIndex:15];
    [computeEncoder setBytes:&alphaMin length:sizeof(float) atIndex:16];
    [computeEncoder setBytes:&alphaMax length:sizeof(float) atIndex:17];
    [computeEncoder setBytes:&linearAdjustment length:sizeof(float) atIndex:18];
    
    [computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup:threadGroupCount];
    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // Release the temporary buffers
    [negativePresetBuf release];
    [printPresetBuf release];
}