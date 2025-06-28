#include "MatrixManager.h"
#include "SimpleJSON.h"
#include <fstream>
#include <sstream>
#include <iostream>

// Static member initialization
std::map<int, Matrix3x3> MatrixManager::inputMatrices;
std::map<int, Matrix3x3> MatrixManager::outputMatrices;
std::map<std::string, Matrix3x3> MatrixManager::creativeWhitepointMatrices;
bool MatrixManager::initialized = false;
std::string MatrixManager::configPath;

Matrix3x3 MatrixManager::getInputMatrix(int gamutIndex) {
    if (!initialized) {
        std::cerr << "MatrixManager not initialized! Call loadMatrices() first." << std::endl;
        return Matrix3x3(); // Return identity matrix
    }
    
    auto it = inputMatrices.find(gamutIndex);
    if (it != inputMatrices.end()) {
        return it->second;
    }
    
    std::cerr << "Input gamut " << gamutIndex << " not found, returning identity matrix." << std::endl;
    return Matrix3x3(); // Return identity matrix as fallback
}

Matrix3x3 MatrixManager::getOutputMatrix(int gamutIndex) {
    if (!initialized) {
        std::cerr << "MatrixManager not initialized! Call loadMatrices() first." << std::endl;
        return Matrix3x3(); // Return identity matrix
    }
    
    auto it = outputMatrices.find(gamutIndex);
    if (it != outputMatrices.end()) {
        return it->second;
    }
    
    std::cerr << "Output gamut " << gamutIndex << " not found, returning identity matrix." << std::endl;
    return Matrix3x3(); // Return identity matrix as fallback
}

Matrix3x3 MatrixManager::getCreativeWhitepointMatrix(int displayGamut, int cwpIndex) {
    if (!initialized) {
        return Matrix3x3(); // Return identity matrix
    }
    
    std::string key;
    if (displayGamut == 5) { // Rec.709
        switch (cwpIndex) {
            case 1: key = "rec709_d60"; break;
            case 2: key = "rec709_d55"; break;
            case 3: key = "rec709_d50"; break;
            default: return Matrix3x3(); // Identity for D65
        }
    } else if (displayGamut == 3) { // P3-D65
        switch (cwpIndex) {
            case 1: key = "p3d65_d60"; break;
            case 2: key = "p3d65_d55"; break;
            case 3: key = "p3d65_d50"; break;
            default: return Matrix3x3(); // Identity for D65
        }
    }
    
    auto it = creativeWhitepointMatrices.find(key);
    if (it != creativeWhitepointMatrices.end()) {
        return it->second;
    }
    
    return Matrix3x3(); // Return identity matrix as fallback
}

bool MatrixManager::loadMatrices(const std::string& configFilePath) {
    configPath = configFilePath;
    
    auto matrices = SimpleJSON::parseMatrixFile(configFilePath);
    if (matrices.empty()) {
        std::cerr << "Failed to load any matrices from: " << configFilePath << std::endl;
        return false;
    }
    
    try {
        // Load input gamut matrices
        for (int i = 0; i <= 5; i++) {
            std::string key = "input_" + std::to_string(i);
            auto it = matrices.find(key);
            if (it != matrices.end() && it->second.size() == 3 && it->second[0].size() == 3) {
                const auto& matrix = it->second;
                Matrix3x3 mat(
                    matrix[0][0], matrix[0][1], matrix[0][2],
                    matrix[1][0], matrix[1][1], matrix[1][2],
                    matrix[2][0], matrix[2][1], matrix[2][2]
                );
                inputMatrices[i] = mat;
            }
        }
        
        // Load output gamut matrices
        for (int i = 0; i <= 5; i++) {
            std::string key = "output_" + std::to_string(i);
            auto it = matrices.find(key);
            if (it != matrices.end() && it->second.size() == 3 && it->second[0].size() == 3) {
                const auto& matrix = it->second;
                Matrix3x3 mat(
                    matrix[0][0], matrix[0][1], matrix[0][2],
                    matrix[1][0], matrix[1][1], matrix[1][2],
                    matrix[2][0], matrix[2][1], matrix[2][2]
                );
                outputMatrices[i] = mat;
            }
        }
        
        initialized = true;
        std::cout << "Successfully loaded " << inputMatrices.size() << " input gamuts and " 
                  << outputMatrices.size() << " output gamuts." << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error parsing matrix configuration: " << e.what() << std::endl;
        return false;
    }
}

std::string MatrixManager::generateMatrixConstants(int inputGamut, int outputGamut, int cwp) {
    if (!initialized) {
        return "// MatrixManager not initialized\n";
    }
    
    std::stringstream ss;
    ss << "// Generated matrix constants for input gamut " << inputGamut 
       << " to output gamut " << outputGamut << "\n";
    
    // Add input matrix
    Matrix3x3 inputMatrix = getInputMatrix(inputGamut);
    ss << matrixToMetalString(inputMatrix, "input_matrix");
    
    // Add output matrix
    Matrix3x3 outputMatrix = getOutputMatrix(outputGamut);
    ss << matrixToMetalString(outputMatrix, "output_matrix");
    
    // Add creative whitepoint matrix if needed
    if (cwp > 0) {
        Matrix3x3 cwpMatrix = getCreativeWhitepointMatrix(outputGamut, cwp);
        ss << matrixToMetalString(cwpMatrix, "cwp_matrix");
    }
    
    return ss.str();
}

bool MatrixManager::isInitialized() {
    return initialized;
}

std::string MatrixManager::getGamutName(int gamutIndex, bool isOutput) {
    // Simple lookup - could be enhanced to read from JSON
    std::map<int, std::string> names = {
        {0, "XYZ"},
        {1, "ACES 2065-1 (AP0)"},
        {2, "ACEScg (AP1)"},
        {3, "P3-D65"},
        {4, "Rec.2020"},
        {5, "Rec.709"}
    };
    
    auto it = names.find(gamutIndex);
    if (it != names.end()) {
        return it->second;
    }
    
    return "Unknown Gamut";
}

std::string MatrixManager::matrixToMetalString(const Matrix3x3& matrix, const std::string& name) {
    std::stringstream ss;
    ss << "constant float3x3 " << name << " = float3x3(\n";
    ss << "    float3(" << matrix.m[0][0] << "f, " << matrix.m[0][1] << "f, " << matrix.m[0][2] << "f),\n";
    ss << "    float3(" << matrix.m[1][0] << "f, " << matrix.m[1][1] << "f, " << matrix.m[1][2] << "f),\n";
    ss << "    float3(" << matrix.m[2][0] << "f, " << matrix.m[2][1] << "f, " << matrix.m[2][2] << "f)\n";
    ss << ");\n\n";
    return ss.str();
}
