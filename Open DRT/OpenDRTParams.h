#pragma once

// OpenDRT Parameters Structure - shared between C++ and Metal
struct OpenDRTParams {
    // Input/Output Settings
    int inGamut;
    int inOetf;
    
    // Tonescale Parameters
    float tnLp;           // Display Peak Luminance
    float tnGb;           // HDR Grey Boost
    float ptHdr;          // HDR Purity
    
    // Clamp Parameters
    int clamp;            // bool converted to int for Metal compatibility
    float tnLg;           // Grey Luminance
    float tnCon;          // Contrast
    float tnSh;           // Shoulder Clip
    float tnToe;          // Toe
    float tnOff;          // Offset
    
    // High Contrast Parameters
    float tnHcon;         // Contrast High
    float tnHconPv;       // Contrast High Pivot
    float tnHconSt;       // Contrast High Strength
    
    // Low Contrast Parameters
    float tnLcon;         // Contrast Low
    float tnLconW;        // Contrast Low Width
    float tnLconPc;       // Contrast Low Per-Channel
    
    // Creative White Parameters
    int cwp;              // Creative White
    float cwpRng;         // Creative White Range
    
    // Render Space Parameters
    float rsSa;           // Render Space Strength
    float rsRw;           // Render Space Red Weight
    float rsBw;           // Render Space Blue Weight
    
    // Purity Compress Parameters
    float ptR, ptG, ptB;  // Purity Compress RGB
    float ptRngLow;       // Purity Range Low
    float ptRngHigh;      // Purity Range High
    
    // Mid Purity Parameters
    float ptmLow;         // Mid Purity Low
    float ptmLowSt;       // Mid Purity Low Strength
    float ptmHigh;        // Mid Purity High
    float ptmHighSt;      // Mid Purity High Strength
    
    // Brilliance Parameters
    float brlR, brlG, brlB; // Brilliance RGB
    float brlC, brlM, brlY; // Brilliance CMY
    float brlRng;         // Brilliance Range
    
    // Hueshift RGB Parameters
    float hsR, hsG, hsB;  // Hueshift RGB
    float hsRgbRng;       // Hueshift RGB Range
    
    // Hueshift CMY Parameters
    float hsC, hsM, hsY;  // Hueshift CMY
    
    // Hue Contrast Parameters
    float hcR;            // Hue Contrast R
    
    // NEW PARAMETERS - Filmic Mode and Advanced Controls
    int filmicMode;       // Enable Filmic Mode (bool converted to int)
    float filmicDynamicRange; // Filmic Dynamic Range
    int filmicProjectorSim;   // Filmic Projector Simulation
    float filmicSourceStops;  // Original Camera Range
    float filmicTargetStops;  // Target Film Range
    float filmicStrength;     // Filmic Strength
    int advHueContrast;   // Advanced Hue Contrast (bool converted to int)
    int tonescaleMap;     // Tonescale Map (bool converted to int)
    int diagnosticsMode;  // Diagnostics Mode (bool converted to int)
    int rgbChipsMode;     // RGB Chips Mode (bool converted to int)
    int betaFeaturesEnable; // Beta Features Enable (bool converted to int)
    
    // Display Parameters
    int displayGamut;     // Display Gamut
    int eotf;             // Display EOTF
    
    // Matrix data (3x3 = 9 floats each)
    float inputMatrix[9];
    float outputMatrix[9];
    float cwpMatrix[9];
    float xyzToP3Matrix[9];     // XYZ to P3-D65 working space
    float p3ToRec709Matrix[9];  // P3 to Rec.709 D65 (for display_gamut==0)
    
    
    // Transfer Function Parameters (NEW)
    float oetfParams[8];    // OETF parameters for linearization
    float eotfParams[8];    // EOTF parameters for display encoding
    int oetfType;           // OETF function type
    int eotfType;           // EOTF function type
    
    // PRECALCULATED TONESCALE CONSTANTS (NEW)
    float ts_x1;        // pow(2.0f, 6.0f*tn_sh + 4.0f)
    float ts_y1;        // tn_Lp/100.0f
    float ts_x0;        // 0.18f + tn_off
    float ts_y0;        // tn_Lg/100.0f*(1.0f + tn_gb*log2(ts_y1))
    float ts_s0;        // compress_toe_quadratic(ts_y0, tn_toe, 1)
    float ts_s10;       // ts_x0*(pow(ts_s0, -1.0f/tn_con) - 1.0f)
    float ts_m1;        // ts_y1/pow(ts_x1/(ts_x1 + ts_s10), tn_con)
    float ts_m2;        // compress_toe_quadratic(ts_m1, tn_toe, 1)
    float ts_s;         // ts_x0*(pow(ts_s0/ts_m2, -1.0f/tn_con) - 1.0f)
    float ts_dsc;       // Display scale factor based on EOTF
    float pt_cmp_Lf;    // Lerped purity compression factor
    float s_Lp100;      // Scene-linear scale at 100 nits
    float ts_s1;        // Final tonescale factor
    
    // MODULE ENABLE FLAGS
    int tnHconEnable;
    int tnLconEnable;
    int ptlEnable;
    int ptmEnable;
    int brlEnable;
    int hsRgbEnable;
    int hsCmyEnable;
    int hcEnable;

    // UI ENABLE FLAGS (for UI visibility)
    int tnHconUIEnable;
    int tnLconUIEnable;
    int ptlUIEnable;
    int ptmUIEnable;
    int brlUIEnable;
    int hsRgbUIEnable;
    int hsCmyUIEnable;
    int hcUIEnable;
    
    // PRESET ENABLE FLAGS (for execution)
    int tnHconPresetEnable;
    int tnLconPresetEnable;
    int ptlPresetEnable;
    int ptmPresetEnable;
    int brlPresetEnable;
    int hsRgbPresetEnable;
    int hsCmyPresetEnable;
    int hcPresetEnable;
    
};