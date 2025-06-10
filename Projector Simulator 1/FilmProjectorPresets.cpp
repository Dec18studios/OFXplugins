#include "FilmProjectorPresets.h"

FilmStockPreset FilmStockPresets::loadPreset(int stockType, int layerMode, bool isNegative) {
    if (isNegative) {
        // Negative stocks
        if (stockType == NEG_5247) {
            if (layerMode == RED_ONLY) {
                return {
                    650.0f, 525.0f, 476.4f, 511.5f,    // Means
                    48.1f, 35.9f, 26.9f, 110.6f,       // Left std
                    40.5f, 28.4f, 26.9f, 19.2f,        // Right std
                    0.95f, 0.0f, 0.0f, 1.575f,         // Max values
                    0.102f, 0.0f, 0.0f, 0.126f,        // Min values
                    0.228f,                             // Silver ratio
                    1.0f                                // Input gain
                };
            } else if (layerMode == GREEN_ONLY) {
                return {
                    650.0f, 525.0f, 476.4f, 511.5f,    // Means
                    48.1f, 35.9f, 26.9f, 110.6f,       // Left std
                    40.5f, 28.4f, 26.9f, 19.2f,        // Right std
                    0.0f, 0.95f, 0.0f, 1.481f,         // Max values
                    0.0f, 0.083f, 0.0f, 0.084f,        // Min values
                    0.007f,                             // Silver ratio
                    1.0f                                // Input gain
                };
            } else if (layerMode == BLUE_ONLY) {
                return {
                    650.0f, 525.0f, 476.4f, 511.5f,    // Means
                    48.1f, 35.9f, 26.9f, 110.6f,       // Left std
                    40.5f, 28.4f, 26.9f, 19.2f,        // Right std
                    0.0f, 0.0f, 0.95f, 1.481f,         // Max values
                    0.0f, 0.0f, 0.083f, 0.084f,        // Min values
                    0.007f,                             // Silver ratio
                    1.0f                                // Input gain
                };
            }
        } else if (stockType == NEG_5213) {
            if (layerMode == RED_ONLY) {
                return {
                    650.0f, 550.0f, 450.0f, 510.0f,    // Means
                    50.0f, 40.0f, 30.0f, 120.0f,       // Left std
                    45.0f, 35.0f, 25.0f, 20.0f,        // Right std
                    1.4f, 0.0f, 0.0f, 1.5f,            // Max values
                    0.1f, 0.0f, 0.0f, 0.1f,            // Min values
                    0.25f,                              // Silver ratio
                    1.0f                                // Input gain
                };
            } else if (layerMode == GREEN_ONLY) {
                return {
                    650.0f, 550.0f, 450.0f, 510.0f,    // Means
                    50.0f, 40.0f, 30.0f, 120.0f,       // Left std
                    45.0f, 35.0f, 25.0f, 20.0f,        // Right std
                    0.0f, 1.2f, 0.0f, 1.4f,            // Max values
                    0.0f, 0.1f, 0.0f, 0.1f,            // Min values
                    0.20f,                              // Silver ratio
                    1.0f                                // Input gain
                };
            } else if (layerMode == BLUE_ONLY) {
                return {
                    650.0f, 550.0f, 450.0f, 510.0f,    // Means
                    50.0f, 40.0f, 30.0f, 120.0f,       // Left std
                    45.0f, 35.0f, 25.0f, 20.0f,        // Right std
                    0.0f, 0.0f, 1.6f, 1.3f,            // Max values
                    0.0f, 0.0f, 0.1f, 0.1f,            // Min values
                    0.15f,                              // Silver ratio
                    1.0f                                // Input gain
                };
            }
        }
    } else {
        // Print stocks
        if (stockType == PRINT_2383) {
            if (layerMode == RED_ONLY) {
                return {
                    660.0f, 545.0f, 446.0f, 500.0f,    // Means
                    43.0f, 45.4f, 50.3f, 139.0f,       // Left std
                    57.6f, 32.0f, 24.8f, 200.9f,       // Right std
                    1.087f, 0.0f, 0.0f, 1.891f,        // Max values
                    0.063f, 0.0f, 0.0f, 0.024f,        // Min values
                    0.008f,                             // Silver ratio
                    1.0f                                // Input gain
                };
            } else if (layerMode == GREEN_ONLY) {
                return {
                    660.0f, 545.0f, 446.0f, 500.0f,    // Means
                    43.0f, 45.4f, 50.3f, 139.0f,       // Left std
                    57.6f, 32.0f, 24.8f, 200.9f,       // Right std
                    0.0f, 0.880f, 0.0f, 1.891f,        // Max values
                    0.0f, 0.009f, 0.0f, 0.024f,        // Min values
                    0.043f,                             // Silver ratio
                    1.0f                                // Input gain
                };
            } else if (layerMode == BLUE_ONLY) {
                return {
                    660.0f, 545.0f, 446.0f, 500.0f,    // Means
                    43.0f, 45.4f, 50.3f, 139.0f,       // Left std
                    57.6f, 32.0f, 24.8f, 200.9f,       // Right std
                    0.0f, 0.0f, 0.813f, 1.891f,        // Max values
                    0.0f, 0.0f, 0.03f, 0.024f,         // Min values
                    0.0043f,                            // Silver ratio
                    1.0f                                // Input gain
                };
            }
        } else if (stockType == PRINT_5384) {
            if (layerMode == RED_ONLY) {
                return {
                    660.0f, 545.0f, 450.0f, 500.0f,    // Means
                    50.0f, 45.0f, 40.0f, 120.0f,       // Left std
                    60.0f, 50.0f, 45.0f, 150.0f,       // Right std
                    1.20f, 0.0f, 0.0f, 1.80f,          // Max values
                    0.05f, 0.0f, 0.0f, 0.02f,          // Min values
                    0.01f,                              // Silver ratio
                    1.0f                                // Input gain
                };
            } else if (layerMode == GREEN_ONLY) {
                return {
                    660.0f, 545.0f, 450.0f, 500.0f,    // Means
                    50.0f, 45.0f, 40.0f, 120.0f,       // Left std
                    60.0f, 50.0f, 45.0f, 150.0f,       // Right std
                    0.0f, 1.10f, 0.0f, 1.80f,          // Max values
                    0.0f, 0.04f, 0.0f, 0.02f,          // Min values
                    0.02f,                              // Silver ratio
                    1.0f                                // Input gain
                };
            } else if (layerMode == BLUE_ONLY) {
                return {
                    660.0f, 545.0f, 450.0f, 500.0f,    // Means
                    50.0f, 45.0f, 40.0f, 120.0f,       // Left std
                    60.0f, 50.0f, 45.0f, 150.0f,       // Right std
                    0.0f, 0.0f, 1.00f, 1.80f,          // Max values
                    0.0f, 0.0f, 0.03f, 0.02f,          // Min values
                    0.015f,                             // Silver ratio
                    1.0f                                // Input gain
                };
            }
        }
    }
    
    // Default preset
    return {
        660.0f, 525.0f, 476.4f, 511.5f,    // Means
        48.1f, 35.9f, 26.9f, 110.6f,       // Left std
        40.5f, 28.4f, 26.9f, 19.2f,        // Right std
        0.95f, 0.0f, 0.0f, 1.575f,         // Max values
        0.079f, 0.0f, 0.0f, 0.126f,        // Min values
        0.236f,                             // Silver ratio
        1.0f                                // Input gain
    };
}

void FilmStockPresets::presetToArray(const FilmStockPreset& preset, float* array) {
    array[0] = preset.cyan_mean;
    array[1] = preset.magenta_mean;
    array[2] = preset.yellow_mean;
    array[3] = preset.silver_mean;
    
    array[4] = preset.cyan_left_std;
    array[5] = preset.magenta_left_std;
    array[6] = preset.yellow_left_std;
    array[7] = preset.silver_left_std;
    
    array[8] = preset.cyan_right_std;
    array[9] = preset.magenta_right_std;
    array[10] = preset.yellow_right_std;
    array[11] = preset.silver_right_std;
    
    array[12] = preset.cyan_max_value;
    array[13] = preset.magenta_max_value;
    array[14] = preset.yellow_max_value;
    array[15] = preset.silver_max_value;
    
    array[16] = preset.cyan_min_value;
    array[17] = preset.magenta_min_value;
    array[18] = preset.yellow_min_value;
    array[19] = preset.silver_min_value;
    
    array[20] = preset.silver_ratio;
    array[21] = preset.input_gain;
}