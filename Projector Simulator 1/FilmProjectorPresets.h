#ifndef FILM_PROJECTOR_PRESETS_H
#define FILM_PROJECTOR_PRESETS_H

// Film stock identifiers
enum FilmStockType {
    NEG_5247 = 0,
    NEG_5213 = 1,
    PRINT_2383 = 0,
    PRINT_5384 = 1
};

enum LayerMode {
    RED_ONLY = 0,
    GREEN_ONLY = 1,
    BLUE_ONLY = 2
};

// Structure matching the DCTL film_stock_preset_t
struct FilmStockPreset {
    float cyan_mean, magenta_mean, yellow_mean, silver_mean;           // 4 values
    float cyan_left_std, magenta_left_std, yellow_left_std, silver_left_std;     // 4 values
    float cyan_right_std, magenta_right_std, yellow_right_std, silver_right_std; // 4 values
    float cyan_max_value, magenta_max_value, yellow_max_value, silver_max_value; // 4 values
    float cyan_min_value, magenta_min_value, yellow_min_value, silver_min_value; // 4 values
    float silver_ratio;                                                // 1 value
    float input_gain;                                                  // 1 value
    // Total: 22 float values
};

// Preset storage class
class FilmStockPresets {
public:
    // Load preset based on stock type, layer mode, and negative/print
    static FilmStockPreset loadPreset(int stockType, int layerMode, bool isNegative);
    
    // Convert preset to array for GPU kernel
    static void presetToArray(const FilmStockPreset& preset, float* array);
    
    // Get all preset data as arrays
    static void getAllPresets(float negativePresets[2][3][22], float printPresets[2][3][22]);
};

#endif // FILM_PROJECTOR_PRESETS_H