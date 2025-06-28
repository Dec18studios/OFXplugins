#pragma once

const char* matrixFunctions = \
"// DYNAMIC MATRIX SELECTION FUNCTIONS\n" \
"float3x3 getInputMatrix(int gamut) {\n" \
"    switch(gamut) {\n" \
"        case 0: // DaVinci Wide Gamut to XYZ\n" \
"            return float3x3(\n" \
"                float3(0.700622320175f, 0.274118483067f, -0.098962903023f),\n" \
"                float3(0.148774802685f, 0.873631775379f, -0.137895315886f),\n" \
"                float3(0.101058728993f, -0.147750422359f, 1.325916051865f));\n" \
"        case 1: // Rec.709 to XYZ\n" \
"            return float3x3(\n" \
"                float3(0.4123907992659f, 0.2126390058715f, 0.0193308187155f),\n" \
"                float3(0.3575843393838f, 0.7151686787677f, 0.1191947797946f),\n" \
"                float3(0.1804807884018f, 0.0721923153608f, 0.9505321522496f));\n" \
"        case 2: // Rec.2020 to XYZ\n" \
"            return float3x3(\n" \
"                float3(0.6369580f, 0.2627401f, 0.0000000f),\n" \
"                float3(0.1446169f, 0.6780980f, 0.0280727f),\n" \
"                float3(0.1688809f, 0.0593017f, 1.0609851f));\n" \
"        case 3: // P3-D65 to XYZ\n" \
"            return float3x3(\n" \
"                float3(0.4865709f, 0.2289746f, 0.0000000f),\n" \
"                float3(0.2656677f, 0.6917385f, 0.0451134f),\n" \
"                float3(0.1982173f, 0.0792869f, 1.0439444f));\n" \
"        default:\n" \
"            return float3x3(1.0f); // Identity fallback\n" \
"    }\n" \
"}\n" \
"\n" \
"float3x3 getOutputMatrix(int gamut) {\n" \
"    switch(gamut) {\n" \
"        case 0: // P3 to Rec.709 D65\n" \
"            return float3x3(\n" \
"                float3( 1.224940181f,   -0.04205697775f, -0.01963755488f),\n" \
"                float3(-0.2249402404f,   1.042057037f,   -0.07863604277f),\n" \
"                float3( 0.0f,           -1.4901e-08f,     1.098273635f));\n" \
"        case 1: // P3 Identity (no conversion)\n" \
"            return float3x3(1.0f);\n" \
"        case 2: // P3 to Rec.2020\n" \
"            return float3x3(\n" \
"                float3(0.627403914928f, 0.069097289441f, 0.016391203574f),\n" \
"                float3(0.329283038616f, 0.919540429115f, 0.088013307851f),\n" \
"                float3(0.043313046456f, 0.011362281442f, 0.895595488575f));\n" \
"        case 3: // P3 to DCI-P3\n" \
"            return float3x3(\n" \
"                float3(0.9553970f, 0.0000000f, 0.0000985f),\n" \
"                float3(0.0230340f, 0.9846986f, 0.0157851f),\n" \
"                float3(0.0215690f, 0.0153014f, 0.9841164f));\n" \
"        default:\n" \
"            return float3x3(1.0f);\n" \
"    }\n" \
"}\n" \
"\n" \
"float3x3 getCreativeWhitepointMatrix(int displayGamut, int cwp) {\n" \
"    // Creative Whitepoint matrices for different display gamuts\n" \
"    if (displayGamut == 0) { // Rec.709 CWP matrices\n" \
"        switch(cwp) {\n" \
"            case 1: // D60 to Rec.709\n" \
"                return float3x3(\n" \
"                    float3(0.9872240f, -0.0061365f,  0.0159533f),\n" \
"                    float3(0.0047890f,  0.9989840f, -0.0037006f),\n" \
"                    float3(-0.0030181f, 0.0071525f,  0.9877316f));\n" \
"            case 2: // D55 to Rec.709\n" \
"                return float3x3(\n" \
"                    float3(0.9581476f, -0.0196350f,  0.0525814f),\n" \
"                    float3(0.0180057f,  0.9943132f, -0.0123044f),\n" \
"                    float3(-0.0119398f, 0.0253218f,  0.9597370f));\n" \
"            case 3: // D50 to Rec.709\n" \
"                return float3x3(\n" \
"                    float3(0.9555766f, -0.0282895f,  0.0756529f),\n" \
"                    float3(0.0267034f,  0.9908566f, -0.0175600f),\n" \
"                    float3(-0.0175095f, 0.0374329f,  0.9419071f));\n" \
"            default: // D65 - Identity\n" \
"                return float3x3(1.0f);\n" \
"        }\n" \
"    } else if (displayGamut == 1) { // P3-D65 CWP matrices\n" \
"        switch(cwp) {\n" \
"            case 1: // D60 to P3\n" \
"                return float3x3(\n" \
"                    float3(0.9923664f, -0.0070295f,  0.0146631f),\n" \
"                    float3(0.0048174f,  0.9991623f, -0.0039797f),\n" \
"                    float3(-0.0025838f, 0.0078672f,  0.9854166f));\n" \
"            case 2: // D55 to P3\n" \
"                return float3x3(\n" \
"                    float3(0.9692877f, -0.0216498f,  0.0523620f),\n" \
"                    float3(0.0166677f,  0.9958344f, -0.0125021f),\n" \
"                    float3(-0.0089554f, 0.0258154f,  0.9601380f));\n" \
"            case 3: // D50 to P3\n" \
"                return float3x3(\n" \
"                    float3(0.9619472f, -0.0308901f,  0.0689429f),\n" \
"                    float3(0.0245098f,  0.9937902f, -0.0183000f),\n" \
"                    float3(-0.0135570f, 0.0371000f,  0.9516570f));\n" \
"            default: // D65 - Identity\n" \
"                return float3x3(1.0f);\n" \
"        }\n" \
"    } else { // Rec.2020 or other gamuts\n" \
"        return float3x3(1.0f); // Identity for now\n" \
"    }\n" \
"}\n" \
"\n" \
"float3x3 getXYZToWorkingMatrix() {\n" \
"    // XYZ to P3-D65 working space (hardcoded for now)\n" \
"    return float3x3(\n" \
"        float3( 2.49349691194f, -0.829488694668f,  0.0358458302915f),\n" \
"        float3(-0.931383617919f, 1.76266097069f,  -0.0761723891287f),\n" \
"        float3(-0.402710784451f, 0.0236246771724f, 0.956884503364f));\n" \
"}\n" \
"\n";