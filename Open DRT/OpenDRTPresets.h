#pragma once

#include <cmath>

// OpenDRT Preset Values
// Based on the DCTL preset tables

namespace OpenDRTPresets {

    // Matrix structure for 3x3 color matrices
    struct ColorMatrix3x3 {
        float m[9]; // Row-major order: [0,1,2] = row1, [3,4,5] = row2, [6,7,8] = row3
        
        ColorMatrix3x3(float m00, float m01, float m02,
                       float m10, float m11, float m12,
                       float m20, float m21, float m22) {
            m[0] = m00; m[1] = m01; m[2] = m02;
            m[3] = m10; m[4] = m11; m[5] = m12;
            m[6] = m20; m[7] = m21; m[8] = m22;
        }
    };

    // Input Gamut Enums (matching DCTL order)
    enum InputGamutType {
        IN_GAMUT_XYZ = 0,
        IN_GAMUT_AP0,           // ACES 2065-1
        IN_GAMUT_AP1,           // ACEScg
        IN_GAMUT_P3D65,         // P3-D65
        IN_GAMUT_REC2020,       // Rec.2020
        IN_GAMUT_REC709,        // Rec.709
        IN_GAMUT_AWG3,          // Arri Wide Gamut 3
        IN_GAMUT_AWG4,          // Arri Wide Gamut 4
        IN_GAMUT_RWG,           // Red Wide Gamut RGB
        IN_GAMUT_SGAMUT3,       // Sony SGamut3
        IN_GAMUT_SGAMUT3CINE,   // Sony SGamut3Cine
        IN_GAMUT_VGAMUT,        // Panasonic V-Gamut
        IN_GAMUT_BMDWG,         // Blackmagic Wide Gamut
        IN_GAMUT_EGAMUT,        // Filmlight E-Gamut
        IN_GAMUT_EGAMUT2,       // Filmlight E-Gamut2
        IN_GAMUT_DAVINCIWG      // DaVinci Wide Gamut
    };

    // Display Gamut Enums
    enum DisplayGamutType {
        DISPLAY_P3D65 = 0,
        DISPLAY_REC2020,
        DISPLAY_REC709
    };

    // Creative Whitepoint Enums
    enum CreativeWhitepointType {
        CWP_NONE = 0,
        CWP_D60,
        CWP_D55,
        CWP_D50
    };

    // CORRECTED INPUT GAMUT MATRICES
static const ColorMatrix3x3 INPUT_GAMUT_MATRICES[] = {
    // XYZ (Identity) - CORRECT
    ColorMatrix3x3(1.0f, 0.0f, 0.0f,
                   0.0f, 1.0f, 0.0f,
                   0.0f, 0.0f, 1.0f),
    
    // AP0 to XYZ - CORRECTED
    ColorMatrix3x3(0.93863094875f, -0.00574192055f, 0.017566898852f,
                   0.338093594922f, 0.727213902811f, -0.065307497733f,
                   0.000723121511f, 0.000818441849f, 1.0875161874f),
    
    // AP1 to XYZ - CORRECTED
    ColorMatrix3x3(0.652418717672f, 0.127179925538f, 0.170857283842f,
                   0.268064059194f, 0.672464478993f, 0.059471461813f,
                   -0.00546992851f, 0.005182799977f, 1.08934487929f),
    
    // P3-D65 to XYZ - CORRECT
    ColorMatrix3x3(0.486571133137f, 0.265667706728f, 0.198217317462f,
                   0.228974640369f, 0.691738605499f, 0.079286918044f,
                   0.0f, 0.045113388449f, 1.043944478035f),
    
    // Rec.2020 to XYZ - CORRECT
    ColorMatrix3x3(0.636958122253f, 0.144616916776f, 0.168880969286f,
                   0.262700229883f, 0.677998125553f, 0.059301715344f,
                   0.0f, 0.028072696179f, 1.060985088348f),
    
    // Rec.709 to XYZ - CORRECT
    ColorMatrix3x3(0.412390917540f, 0.357584357262f, 0.180480793118f,
                   0.212639078498f, 0.715168714523f, 0.072192311287f,
                   0.019330825657f, 0.119194783270f, 0.950532138348f),
    
    // Arri Wide Gamut 3 to XYZ - CORRECTED
    ColorMatrix3x3(0.638007619284f, 0.214703856337f, 0.097744451431f,
                   0.291953779f, 0.823841041511f, -0.11579482051f,
                   0.002798279032f, -0.067034235689f, 1.15329370742f),
    
    // Arri Wide Gamut 4 to XYZ - CORRECTED  
    ColorMatrix3x3(0.704858320407f, 0.12976029517f, 0.115837311474f,
                   0.254524176404f, 0.781477732712f, -0.036001909116f,
                   0.0f, 0.0f, 1.08905775076f),
    
    // Red Wide Gamut RGB to XYZ - CORRECTED
    ColorMatrix3x3(0.735275208950f, 0.068609409034f, 0.146571278572f,
                   0.286694079638f, 0.842979073524f, -0.129673242569f,
                   -0.079680845141f, -0.347343206406f, 1.516081929207f),
    
    // Sony S-Gamut3 to XYZ - CORRECTED
    ColorMatrix3x3(0.706482713192f, 0.128801049791f, 0.115172164069f,
                   0.270979670813f, 0.786606411221f, -0.057586082034f,
                   -0.009677845386f, 0.004600037493f, 1.09413555865f),
    
    // Sony S-Gamut3.Cine to XYZ - CORRECTED
    ColorMatrix3x3(0.599083920758f, 0.248925516115f, 0.102446490178f,
                   0.215075820116f, 0.885068501744f, -0.100144321859f,
                   -0.032065849545f, -0.027658390679f, 1.14878199098f),
    
    // Panasonic V-Gamut to XYZ - CORRECTED
    ColorMatrix3x3(0.679644469878f, 0.15221141244f, 0.118600044733f,
                   0.26068555009f, 0.77489446333f, -0.03558001342f,
                   -0.009310198218f, -0.004612467044f, 1.10298041602f),
    
    // Blackmagic Wide Gamut to XYZ - CORRECTED
    ColorMatrix3x3(0.606538414955f, 0.220412746072f, 0.123504832387f,
                   0.267992943525f, 0.832748472691f, -0.100741356611f,
                   -0.029442556202f, -0.086612440646f, 1.205112814903f),
    
    // Filmlight E-Gamut to XYZ - CORRECTED
    ColorMatrix3x3(0.705396831036f, 0.164041340351f, 0.081017754972f,
                   0.280130714178f, 0.820206701756f, -0.100337378681f,
                   -0.103781513870f, -0.072907261550f, 1.265746593475f),
    
    // Filmlight E-Gamut2 to XYZ - CORRECTED
    ColorMatrix3x3(0.736477700184f, 0.130739651087f, 0.083238575781f,
                   0.275069984406f, 0.828017790216f, -0.103087774621f,
                   -0.124225154248f, -0.087159767391f, 1.3004426724f),
    
    // DaVinci Wide Gamut to XYZ - CORRECTED  
    ColorMatrix3x3(0.700622320175f, 0.148774802685f, 0.101058728993f,
                   0.274118483067f, 0.873631775379f, -0.147750422359f,
                   -0.098962903023f, -0.137895315886f, 1.325916051865f)
};

// Fix your OUTPUT_GAMUT_MATRICES to match UI order:
static const ColorMatrix3x3 OUTPUT_GAMUT_MATRICES[] = {
    // Index 0: P3 to Rec.709 (for "Rec.709" selection)
    ColorMatrix3x3(1.224940181f, -0.2249402404f, 0.0f,
                   -0.04205697775f, 1.042057037f, -1.4901e-08f,
                   -0.01963755488f, -0.07863604277f, 1.098273635f),
    
    // Index 1: P3 to P3-D65 (identity - for "P3-D65" selection)
    ColorMatrix3x3(1.0f, 0.0f, 0.0f,
                   0.0f, 1.0f, 0.0f,
                   0.0f, 0.0f, 1.0f),
    
    // Index 2: P3 to Rec.2020 (for "Rec.2020 (P3 Limited)" selection)
    ColorMatrix3x3(0.7538330344f, 0.1985973691f, 0.04756959659f,
                   0.04574384897f, 0.9417772198f, 0.01247893122f,
                   -0.001210340355f, 0.0176017173f, 0.9836086231f)
};



    // CREATIVE WHITEPOINT MATRICES (P3 to P3-DXX)
    static const ColorMatrix3x3 CREATIVE_WHITEPOINT_MATRICES[][4] = {
        // P3 Display matrices
    {
        // None (Identity)
        ColorMatrix3x3(1.0f, 0.0f, 0.0f,
                       0.0f, 1.0f, 0.0f,
                       0.0f, 0.0f, 1.0f),
        // D60 - CORRECTED
        ColorMatrix3x3(0.979832881f, 0.01836378979f, 0.001803284786f,
                       -0.000805359793f, 0.9618000331f, 1.8876121e-05f,
                       -0.000338382322f, -0.003671835795f, 0.894139105f),
        // D55 - CORRECTED  
        ColorMatrix3x3(0.9559790976f, 0.0403850003f, 0.003639287409f,
                       -0.001771929896f, 0.9163058305f, 3.3300759e-05f,
                       -0.000674760809f, -0.0072466358f, 0.7831189153f),
        // D50 - CORRECTED
        ColorMatrix3x3(0.9287127388f, 0.06578032793f, 0.005506708345f,
                       -0.002887159176f, 0.8640709228f, 4.3593718e-05f,
                       -0.001009551548f, -0.01073503317f, 0.6672692039f)
        },
        // Rec.2020 Display matrices (same as P3 for now)
    {
        // None (Identity)
        ColorMatrix3x3(1.0f, 0.0f, 0.0f,
                       0.0f, 1.0f, 0.0f,
                       0.0f, 0.0f, 1.0f),
        // D60 - CORRECTED
        ColorMatrix3x3(0.979832881f, 0.01836378979f, 0.001803284786f,
                       -0.000805359793f, 0.9618000331f, 1.8876121e-05f,
                       -0.000338382322f, -0.003671835795f, 0.894139105f),
        // D55 - CORRECTED  
        ColorMatrix3x3(0.9559790976f, 0.0403850003f, 0.003639287409f,
                       -0.001771929896f, 0.9163058305f, 3.3300759e-05f,
                       -0.000674760809f, -0.0072466358f, 0.7831189153f),
        // D50 - CORRECTED
        ColorMatrix3x3(0.9287127388f, 0.06578032793f, 0.005506708345f,
                       -0.002887159176f, 0.8640709228f, 4.3593718e-05f,
                       -0.001009551548f, -0.01073503317f, 0.6672692039f)
        },
        // Rec.709 Display matrices (identity for now - can be expanded)
       {
        // None (Identity)
        ColorMatrix3x3(1.0f, 0.0f, 0.0f,
                       0.0f, 1.0f, 0.0f,
                       0.0f, 0.0f, 1.0f),
        // D60 - CORRECTED
        ColorMatrix3x3(1.189986856f, -0.192168414f, 0.002185496045f,
                       -0.04168263635f, 0.9927757018f, -5.5660878e-05f,
                       -0.01937995127f, -0.07933006919f, 0.9734397041f),
        // D55 - CORRECTED
        ColorMatrix3x3(1.149327514f, -0.1536910745f, 0.004366526746f,
                       -0.0412590771f, 0.9351717477f, -0.000116126221f,
                       -0.01900949528f, -0.07928282823f, 0.8437884317f),
        // D50 - CORRECTED  
        ColorMatrix3x3(1.103807322f, -0.1103425121f, 0.006531676079f,
                       -0.04079386701f, 0.8704694227f, -0.000180522628f,
                       -0.01854055914f, -0.07857582481f, 0.7105498861f)
    }
    };

    // Helper functions to get matrices
    inline const ColorMatrix3x3& getInputMatrix(int gamutIndex) {
        if (gamutIndex >= 0 && gamutIndex < sizeof(INPUT_GAMUT_MATRICES)/sizeof(INPUT_GAMUT_MATRICES[0])) {
            return INPUT_GAMUT_MATRICES[gamutIndex];
        }
        return INPUT_GAMUT_MATRICES[0]; // Default to XYZ identity
    }

    inline const ColorMatrix3x3& getOutputMatrix(int displayGamut) {
        if (displayGamut >= 0 && displayGamut < sizeof(OUTPUT_GAMUT_MATRICES)/sizeof(OUTPUT_GAMUT_MATRICES[0])) {
            return OUTPUT_GAMUT_MATRICES[displayGamut];
        }
        return OUTPUT_GAMUT_MATRICES[0]; // Default to P3-D65
    }

    inline const ColorMatrix3x3& getCreativeWhitepointMatrix(int displayGamut, int cwpIndex) {
        if (displayGamut >= 0 && displayGamut < 3 && cwpIndex >= 0 && cwpIndex < 4) {
            return CREATIVE_WHITEPOINT_MATRICES[displayGamut][cwpIndex];
        }
        return CREATIVE_WHITEPOINT_MATRICES[0][0]; // Default to identity
    }


    // OpenDRT Look Preset Structure
    struct OpenDRTLookPreset {
        // Tonescale parameters
        float tn_Lg, tn_con, tn_sh, tn_toe, tn_off;
        
        // High/Low Contrast
        bool tn_hcon_enable, tn_lcon_enable;
        float tn_hcon, tn_hcon_pv, tn_hcon_st;
        float tn_lcon, tn_lcon_w, tn_lcon_pc;
        
        // Creative White
        int cwp;
        float cwp_rng;
        
        // Render Space
        float rs_sa, rs_rw, rs_bw;
        
        // Purity
        float pt_r, pt_g, pt_b, pt_rng_low, pt_rng_high;
        bool ptl_enable, ptm_enable;
        float ptm_low, ptm_low_st, ptm_high, ptm_high_st;
        
        // Brilliance
        bool brl_enable;
        float brl_r, brl_g, brl_b, brl_c, brl_m, brl_y, brl_rng;
        
        // Hueshift RGB
        bool hs_rgb_enable;
        float hs_r, hs_g, hs_b, hs_rgb_rng;
        
        // Hueshift CMY
        bool hs_cmy_enable;
        float hs_c, hs_m, hs_y;
        
        // Hue Contrast
        bool hc_enable;
        float hc_r;
    };

    // OpenDRT Tonescale Preset Structure
    struct OpenDRTTonescalePreset {
        float tn_Lg, tn_con, tn_sh, tn_toe, tn_off;
        bool tn_hcon_enable, tn_lcon_enable;
        float tn_hcon, tn_hcon_pv, tn_hcon_st;
        float tn_lcon, tn_lcon_w, tn_lcon_pc;
    };

    // Look Presets Array (matches your DCTL data)
    static const OpenDRTLookPreset LOOK_PRESETS[] = {
        // Default
        {
            11.1f, 1.4f, 0.5f, 0.003f, 0.005f,  // tn_Lg, tn_con, tn_sh, tn_toe, tn_off
            false, true,  // tn_hcon_enable, tn_lcon_enable
            0.0f, 1.0f, 4.0f,  // tn_hcon, tn_hcon_pv, tn_hcon_st
            1.0f, 0.5f, 1.0f,  // tn_lcon, tn_lcon_w, tn_lcon_pc
            0, 0.5f,  // cwp, cwp_rng
            0.35f, 0.25f, 0.55f,  // rs_sa, rs_rw, rs_bw
            0.5f, 2.0f, 2.0f, 0.2f, 0.8f,  // pt_r, pt_g, pt_b, pt_rng_low, pt_rng_high
            true, true,  // ptl_enable, ptm_enable
            0.2f, 0.5f, -0.8f, 0.3f,  // ptm_low, ptm_low_st, ptm_high, ptm_high_st
            true,  // brl_enable
            -0.5f, -0.4f, -0.2f, 0.0f, 0.0f, 0.0f, 0.66f,  // brl_r, brl_g, brl_b, brl_c, brl_m, brl_y, brl_rng
            true,  // hs_rgb_enable
            0.35f, 0.25f, 0.5f, 0.6f,  // hs_r, hs_g, hs_b, hs_rgb_rng
            true,  // hs_cmy_enable
            0.2f, 0.2f, 0.2f,  // hs_c, hs_m, hs_y
            true,  // hc_enable
            0.6f   // hc_r
        },
        
        // Colorful
        {
            11.1f, 1.3f, 0.5f, 0.005f, 0.005f,
            false, true,
            0.0f, 1.0f, 4.0f,
            0.75f, 1.0f, 1.0f,
            0, 0.5f,
            0.35f, 0.15f, 0.55f,
            0.5f, 0.8f, 0.5f, 0.25f, 0.5f,
            true, true,
            0.5f, 0.5f, -0.8f, 0.3f,
            true,
            -0.55f, -0.5f, 0.0f, 0.0f, 0.0f, 0.1f, 0.5f,
            true,
            0.4f, 0.6f, 0.5f, 0.6f,
            true,
            0.2f, 0.1f, 0.2f,
            true,
            0.8f
        },
        
        // Umbra
        {
            6.0f, 1.8f, 0.5f, 0.001f, 0.015f,
            false, true,
            0.0f, 1.0f, 4.0f,
            1.0f, 1.0f, 1.0f,
            3, 0.8f,
            0.45f, 0.1f, 0.35f,
            0.1f, 0.4f, 2.5f, 0.2f, 0.8f,
            true, true,
            0.4f, 0.5f, -0.8f, 0.3f,
            true,
            -0.7f, -0.6f, -0.2f, 0.0f, -0.25f, 0.1f, 0.9f,
            true,
            0.4f, 0.8f, 0.4f, 1.0f,
            true,
            1.0f, 0.6f, 1.0f,
            true,
            0.8f
        },
        
        // Base
        {
            11.1f, 1.4f, 0.5f, 0.003f, 0.0f,
            false, false,
            0.0f, 1.0f, 4.0f,
            0.0f, 0.5f, 1.0f,
            0, 0.5f,
            0.35f, 0.25f, 0.5f,
            1.0f, 2.0f, 2.5f, 0.25f, 0.25f,
            true, false,
            0.0f, 0.5f, 0.0f, 0.3f,
            false,
            0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f,
            false,
            0.0f, 0.0f, 0.0f, 0.5f,
            false,
            0.0f, 0.0f, 0.0f,
            false,
            0.0f
        }
    };

    // Tonescale Presets Array
    static const OpenDRTTonescalePreset TONESCALE_PRESETS[] = {
        // High-Contrast
        { 11.1f, 1.4f, 0.5f, 0.003f, 0.005f, false, true, 0.0f, 1.0f, 4.0f, 1.0f, 0.5f, 1.0f },
        
        // Low-Contrast
        { 11.1f, 1.4f, 0.5f, 0.003f, 0.005f, false, false, 0.0f, 1.0f, 4.0f, 0.0f, 0.5f, 1.0f },
        
        // ACES-1.x
        { 10.0f, 1.0f, 0.245f, 0.02f, 0.0f, true, true, 0.55f, 0.0f, 2.0f, 1.13f, 1.0f, 1.0f },
        
        // ACES-2.0
        { 10.0f, 1.15f, 0.5f, 0.04f, 0.0f, false, false, 1.0f, 1.0f, 1.0f, 1.0f, 0.6f, 1.0f },
        
        // Marvelous Tonescape
        { 6.0f, 1.5f, 0.5f, 0.003f, 0.01f, true, true, 0.25f, 0.0f, 4.0f, 1.0f, 1.0f, 1.0f },
        
        // Arriba Tonecall
        { 11.1f, 1.05f, 0.5f, 0.1f, 0.015f, false, true, 0.0f, 0.0f, 2.0f, 2.0f, 0.2f, 1.0f },
        
        // DaGrinchi Tonegroan
        { 10.42f, 1.2f, 0.5f, 0.02f, 0.0f, false, false, 0.0f, 1.0f, 1.0f, 0.0f, 0.6f, 1.0f },
        
        // Aery Tonescale
        { 11.1f, 1.15f, 0.5f, 0.04f, 0.006f, false, true, 0.0f, 0.0f, 0.5f, 0.5f, 2.0f, 0.5f },
        
        // Umbra Tonescale
        { 6.0f, 1.8f, 0.5f, 0.001f, 0.015f, false, true, 0.0f, 1.0f, 4.0f, 1.0f, 1.0f, 1.0f }
    };

    // OETF/EOTF Transfer Function Enums
    enum InputOETFType {
        IOETF_LINEAR = 0,
        IOETF_DAVINCI_INTERMEDIATE,
        IOETF_FILMLIGHT_TLOG,
        IOETF_ACESCCT,
        IOETF_ARRI_LOGC3,
        IOETF_ARRI_LOGC4,
        IOETF_REDLOG3G10,
        IOETF_PANASONIC_VLOG,
        IOETF_SONY_SLOG3,
        IOETF_FUJI_FLOG2
    };

    enum DisplayEOTFType {
        EOTF_LINEAR = 0,
        EOTF_GAMMA_2_2,
        EOTF_GAMMA_2_4,
        EOTF_GAMMA_2_6,
        EOTF_PQ_ST2084,
        EOTF_HLG_BT2100
    };

    // Transfer Function Parameter Structure
    struct TransferFunctionParams {
        // OETF Parameters (for linearization)
        float oetf_params[8];  // Generic parameter array for different OETF types
        
        // EOTF Parameters (for display encoding)
        float eotf_params[8];  // Generic parameter array for different EOTF types
        
        // Function type identifiers
        int oetf_type;
        int eotf_type;
    };

    // OETF Preset Data (parameters for each transfer function)
    static const TransferFunctionParams OETF_PRESETS[] = {
        // Linear (no parameters needed)
        {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 
         {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 
         IOETF_LINEAR, EOTF_LINEAR},
        
        // DaVinci Intermediate
        {{10.44426855f, 0.07329248f, 7.0f, 0.0075f, 0.02740668f, 0.0f, 0.0f, 0.0f}, 
         {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 
         IOETF_DAVINCI_INTERMEDIATE, EOTF_LINEAR},
        
        // Filmlight T-Log
        {{16.184376489665897f, 0.09232902596577353f, 0.5520126568606655f, 0.0057048244042473785f, 0.075f, 0.0f, 0.0f, 0.0f}, 
         {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 
         IOETF_FILMLIGHT_TLOG, EOTF_LINEAR},
        
// ACEScct - CORRECTED
{{10.5402377416545f, 0.0729055341958355f, 17.52f, 9.72f, 0.155251141552511f, 0.0f, 0.0f, 0.0f}, 
 {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 
 IOETF_ACESCCT, EOTF_LINEAR},
        
        // Arri LogC3
        {{5.367655f, 0.010591f, 0.092809f, 0.385537f, 0.247190f, 0.052272f, 5.555556f, 0.0f}, 
         {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 
         IOETF_ARRI_LOGC3, EOTF_LINEAR},
        
// Arri LogC4 - CORRECTED PARAMETERS  
{{0.3033266726886969f, 0.7774983977293537f, 14.0f, 0.09286412512218964f, 0.9071358748778103f, 6.0f, 64.0f, 2231.8263090676883f}, 
 {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 
 IOETF_ARRI_LOGC4, EOTF_LINEAR},
        
        // RedLog3G10
        {{15.1927f, 0.01f, 0.224282f, 155.975327f, 0.0f, 0.0f, 0.0f, 0.0f}, 
         {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 
         IOETF_REDLOG3G10, EOTF_LINEAR},
        
        // Panasonic V-Log
        {{5.6f, 0.125f, 0.598206f, 0.241514f, 0.00873f, 0.181f, 0.0f, 0.0f}, 
         {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 
         IOETF_PANASONIC_VLOG, EOTF_LINEAR},
        
        // Sony S-Log3
        {{171.2102946929f, 1023.0f, 95.0f, 0.01125f, 420.0f, 261.5f, 0.18f, 0.01f}, 
         {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 
         IOETF_SONY_SLOG3, EOTF_LINEAR},
        
        // Fuji F-Log2
        {{8.799461f, 0.092864f, 0.384316f, 0.245281f, 5.555556f, 0.064829f, 0.100686685370811f, 0.0f}, 
         {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 
         IOETF_FUJI_FLOG2, EOTF_LINEAR}
    };

    // EOTF Preset Data
    static const TransferFunctionParams EOTF_PRESETS[] = {
        // Linear
        {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 
         {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 
         IOETF_LINEAR, EOTF_LINEAR},
        
        // Gamma 2.2
        {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 
         {2.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 
         IOETF_LINEAR, EOTF_GAMMA_2_2},
        
        // Gamma 2.4
        {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 
         {2.4f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 
         IOETF_LINEAR, EOTF_GAMMA_2_4},
        
        // Gamma 2.6
        {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 
         {2.6f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 
         IOETF_LINEAR, EOTF_GAMMA_2_6},
        
        // PQ ST.2084
        {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 
         {2610.0f/16384.0f, 2523.0f/32.0f, 107.0f/128.0f, 2413.0f/128.0f, 2392.0f/128.0f, 0.0f, 0.0f, 0.0f}, 
         IOETF_LINEAR, EOTF_PQ_ST2084},
        
        // HLG BT.2100
        {{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 
         {0.2627f, 0.6780f, 0.0593f, 1.2f, 0.17883277f, 0.28466892f, 0.55991073f, 0.5f}, 
         IOETF_LINEAR, EOTF_HLG_BT2100}
    };

    // Helper functions to get transfer function parameters
    inline const TransferFunctionParams& getOETFParams(int oetfIndex) {
        if (oetfIndex >= 0 && oetfIndex < sizeof(OETF_PRESETS)/sizeof(OETF_PRESETS[0])) {
            return OETF_PRESETS[oetfIndex];
        }
        return OETF_PRESETS[0]; // Default to linear
    }

    inline const TransferFunctionParams& getEOTFParams(int eotfIndex) {
        if (eotfIndex >= 0 && eotfIndex < sizeof(EOTF_PRESETS)/sizeof(EOTF_PRESETS[0])) {
            return EOTF_PRESETS[eotfIndex];
        }
        return EOTF_PRESETS[0]; // Default to linear
    }

    // Helper function for compress_toe_quadratic (from DCTL)
inline float compress_toe_quadratic(float x, float toe, int invert) {
    if (toe == 0.0f) return x;
    if (invert == 0) {
        return x*x / (x + toe);
    } else {
        return (x + sqrtf(x * (4.0f * toe + x))) / 2.0f;
    }
}

    // Tonescale constants calculation helper
    struct TonescaleConstants {
        float ts_x1;
        float ts_y1;
        float ts_x0;
        float ts_y0;
        float ts_s0;
        float ts_s10;
        float ts_m1;
        float ts_m2;
        float ts_s;
        float ts_dsc;
        float pt_cmp_Lf;
        float s_Lp100;
        float ts_s1;
    };

    // Function to precalculate all tonescale constants
inline TonescaleConstants calculateTonescaleConstants(
    float tn_Lp, float tn_gb, float pt_hdr, float tn_Lg, float tn_con, 
    float tn_sh, float tn_toe, float tn_off, int eotf) {
    
    TonescaleConstants tc = {};
    
    // EXACT DCTL calculations
    tc.ts_x1 = powf(2.0f, 6.0f * tn_sh + 4.0f);
    tc.ts_y1 = tn_Lp / 100.0f;
    tc.ts_x0 = 0.18f + tn_off;
    tc.ts_y0 = (tn_Lg / 100.0f) * (1.0f + tn_gb * log2f(tc.ts_y1));
    
    // Toe compression calculations - CORRECTED
    tc.ts_s0 = compress_toe_quadratic(tc.ts_y0, tn_toe, 1);
    tc.ts_s10 = tc.ts_x0 * (powf(tc.ts_s0, -1.0f / tn_con) - 1.0f);
    
    // Mid-tone calculations - CORRECTED
    tc.ts_m1 = tc.ts_y1 / powf(tc.ts_x1 / (tc.ts_x1 + tc.ts_s10), tn_con);
    tc.ts_m2 = compress_toe_quadratic(tc.ts_m1, tn_toe, 1);
    
    // EXACT DCTL formula for ts_s
    tc.ts_s = tc.ts_x0 * (powf(tc.ts_s0 / tc.ts_m2, -1.0f / tn_con) - 1.0f);
    
    // Display scale factor - EXACT DCTL
    tc.ts_dsc = (eotf == 4) ? 0.01f : (eotf == 5) ? 0.1f : 100.0f / tn_Lp;
    
    // Purity compression interpolation - EXACT DCTL
    tc.pt_cmp_Lf = pt_hdr * fminf(1.0f, (tn_Lp - 100.0f) / 900.0f);
    
    // Scene-linear scale at 100 nits - EXACT DCTL FORMULA
    tc.s_Lp100 = tc.ts_x0 * (powf(tn_Lg / 100.0f, -1.0f / tn_con) - 1.0f);
    
    // Final tonescale factor - EXACT DCTL FORMULA  
    tc.ts_s1 = tc.ts_s * tc.pt_cmp_Lf + tc.s_Lp100 * (1.0f - tc.pt_cmp_Lf);
    
    return tc;
}

    // XYZ to P3-D65 working space matrix
    static const ColorMatrix3x3& getXYZToP3Matrix() {
        static const ColorMatrix3x3 xyzToP3(
            2.4934969119f, -0.9313836179f, -0.4027107845f,
            -0.8294889696f, 1.7626640603f, 0.0236246858f,
            0.0358458302f, -0.0761723893f, 0.9568845240f
        );
        return xyzToP3;
    }
    
// P3-D65 to Rec.709 D65 matrix - CORRECTED
static const ColorMatrix3x3& getP3ToRec709D65Matrix() {
    static const ColorMatrix3x3 p3ToRec709D65(
        1.224940181f, -0.2249402404f, 0.0f,
        -0.04205697775f, 1.042057037f, -1.4901e-08f,
        -0.01963755488f, -0.07863604277f, 1.098273635f
    );
        return p3ToRec709D65;
    }
    
    // OETF Linearization Transfer Functions (matching DCTL exactly)
    inline float oetf_davinci_intermediate(float x) {
        return x <= 0.02740668f ? x/10.44426855f : exp2f(x/0.07329248f - 7.0f) - 0.0075f;
    }
    
    inline float oetf_filmlight_tlog(float x) {
        return x < 0.075f ? (x-0.075f)/16.184376489665897f : expf((x - 0.5520126568606655f)/0.09232902596577353f) - 0.0057048244042473785f;
    }
    
    inline float oetf_acescct(float x) {
        return x <= 0.155251141552511f ? (x - 0.0729055341958355f)/10.5402377416545f : exp2f(x*17.52f - 9.72f);
    }
    
    inline float oetf_arri_logc3(float x) {
        return x < 5.367655f*0.010591f + 0.092809f ? (x - 0.092809f)/5.367655f : (powf(10.0f, (x - 0.385537f)/0.247190f) - 0.052272f)/5.555556f;
    }
    
    inline float oetf_arri_logc4(float x) {
        return x < -0.7774983977293537f ? x*0.3033266726886969f - 0.7774983977293537f : (exp2f(14.0f*(x - 0.09286412512218964f)/0.9071358748778103f + 6.0f) - 64.0f)/2231.8263090676883f;
    }
    
    inline float oetf_red_log3g10(float x) {
        return x < 0.0f ? (x/15.1927f) - 0.01f : (powf(10.0f, x/0.224282f) - 1.0f)/155.975327f - 0.01f;
    }
    
    inline float oetf_panasonic_vlog(float x) {
        return x < 0.181f ? (x - 0.125f)/5.6f : powf(10.0f, (x - 0.598206f)/0.241514f) - 0.00873f;
    }
    
    inline float oetf_sony_slog3(float x) {
        return x < 171.2102946929f/1023.0f ? (x*1023.0f - 95.0f)*0.01125f/(171.2102946929f - 95.0f) : (powf(10.0f, ((x*1023.0f - 420.0f)/261.5f))*(0.18f + 0.01f) - 0.01f);
    }
    
    inline float oetf_fujifilm_flog2(float x) {
        return x < 0.100686685370811f ? (x - 0.092864f)/8.799461f : (powf(10.0f, ((x - 0.384316f)/0.245281f))/5.555556f - 0.064829f/5.555556f);
    }
    
    // EOTF Display Encoding Transfer Functions
    inline float eotf_gamma_power(float x, float gamma, bool inverse) {
        if (inverse) {
            return powf(fmaxf(0.0f, x), 1.0f/gamma);
        } else {
            return powf(fmaxf(0.0f, x), gamma);
        }
    }
    
    // HLG and PQ are more complex and handled separately in the Metal kernel
    
    // Dispatcher function for CPU-side calculations if needed
    inline float apply_oetf_cpu(float x, int type) {
        switch(type) {
            case 0: return x; // Linear
            case 1: return oetf_davinci_intermediate(x);
            case 2: return oetf_filmlight_tlog(x);
            case 3: return oetf_acescct(x);
            case 4: return oetf_arri_logc3(x);
            case 5: return oetf_arri_logc4(x);
            case 6: return oetf_red_log3g10(x);
            case 7: return oetf_panasonic_vlog(x);
            case 8: return oetf_sony_slog3(x);
            case 9: return oetf_fujifilm_flog2(x);
            default: return x;
        }
    }
    
    inline float apply_eotf_cpu(float x, int type, bool inverse) {
        switch(type) {
            case 0: return x; // Linear
            case 1: return eotf_gamma_power(x, 2.2f, inverse);
            case 2: return eotf_gamma_power(x, 2.4f, inverse);
            case 3: return eotf_gamma_power(x, 2.6f, inverse);
            // PQ and HLG need special handling - keep in Metal kernel
            default: return x;
        }
    }

} // namespace OpenDRTPresets
