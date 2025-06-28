#ifndef MATRIX_MANAGER_H
#define MATRIX_MANAGER_H

#include <string>
#include <map>
#include <array>

// Simple 3x3 matrix structure
struct Matrix3x3 {
    float m[3][3];
    
    Matrix3x3() {
        // Initialize to identity
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                m[i][j] = (i == j) ? 1.0f : 0.0f;
            }
        }
    }
    
    Matrix3x3(float m00, float m01, float m02,
              float m10, float m11, float m12,
              float m20, float m21, float m22) {
        m[0][0] = m00; m[0][1] = m01; m[0][2] = m02;
        m[1][0] = m10; m[1][1] = m11; m[1][2] = m12;
        m[2][0] = m20; m[2][1] = m21; m[2][2] = m22;
    }
};

class MatrixManager {
public:
    // Get matrix for input/output gamut conversion
    static Matrix3x3 getInputMatrix(int gamutIndex);
    static Matrix3x3 getOutputMatrix(int gamutIndex);
    static Matrix3x3 getCreativeWhitepointMatrix(int displayGamut, int cwpIndex);
    
    // Load matrices from JSON configuration file
    static bool loadMatrices(const std::string& configFilePath);
    
    // Generate Metal kernel source with only needed matrices
    static std::string generateMatrixConstants(int inputGamut, int outputGamut, int cwp = 0);
    
    // Check if matrices are loaded
    static bool isInitialized();
    
    // Get gamut name for debugging
    static std::string getGamutName(int gamutIndex, bool isOutput = false);
    
private:
    static std::map<int, Matrix3x3> inputMatrices;
    static std::map<int, Matrix3x3> outputMatrices;
    static std::map<std::string, Matrix3x3> creativeWhitepointMatrices;
    static bool initialized;
    static std::string configPath;
    
    // Helper functions
    static Matrix3x3 parseMatrixFromJson(const std::string& jsonArray);
    static std::string matrixToMetalString(const Matrix3x3& matrix, const std::string& name);
};

#endif // MATRIX_MANAGER_H
