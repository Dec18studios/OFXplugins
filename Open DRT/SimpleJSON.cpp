#include "SimpleJSON.h"
#include <fstream>
#include <sstream>
#include <iostream>

std::map<std::string, std::vector<std::vector<float>>> SimpleJSON::parseMatrixFile(const std::string& filepath) {
    std::map<std::string, std::vector<std::vector<float>>> matrices;
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return matrices;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    
    try {
        // Parse input_gamuts
        std::string inputSection = extractValue(content, "input_gamuts");
        if (!inputSection.empty()) {
            for (int i = 0; i <= 5; i++) {
                std::string key = "\"" + std::to_string(i) + "\"";
                std::string gamutSection = extractValue(inputSection, key);
                if (!gamutSection.empty()) {
                    std::string matrixStr = extractValue(gamutSection, "matrix");
                    if (!matrixStr.empty()) {
                        matrices["input_" + std::to_string(i)] = parseMatrix(matrixStr);
                    }
                }
            }
        }
        
        // Parse output_gamuts
        std::string outputSection = extractValue(content, "output_gamuts");
        if (!outputSection.empty()) {
            for (int i = 0; i <= 5; i++) {
                std::string key = "\"" + std::to_string(i) + "\"";
                std::string gamutSection = extractValue(outputSection, key);
                if (!gamutSection.empty()) {
                    std::string matrixStr = extractValue(gamutSection, "matrix");
                    if (!matrixStr.empty()) {
                        matrices["output_" + std::to_string(i)] = parseMatrix(matrixStr);
                    }
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
    }
    
    return matrices;
}

std::vector<float> SimpleJSON::parseFloatArray(const std::string& arrayStr) {
    std::vector<float> values;
    std::string cleaned = trim(arrayStr);
    
    // Remove [ and ]
    if (cleaned.front() == '[') cleaned = cleaned.substr(1);
    if (cleaned.back() == ']') cleaned = cleaned.substr(0, cleaned.length()-1);
    
    std::istringstream iss(cleaned);
    std::string token;
    
    while (std::getline(iss, token, ',')) {
        try {
            values.push_back(std::stof(trim(token)));
        } catch (const std::exception& e) {
            std::cerr << "Failed to parse float: " << token << std::endl;
        }
    }
    
    return values;
}

std::vector<std::vector<float>> SimpleJSON::parseMatrix(const std::string& matrixStr) {
    std::vector<std::vector<float>> matrix;
    std::string cleaned = trim(matrixStr);
    
    // Remove outer [ and ]
    if (cleaned.front() == '[') cleaned = cleaned.substr(1);
    if (cleaned.back() == ']') cleaned = cleaned.substr(0, cleaned.length()-1);
    
    // Find each row (inner array)
    size_t pos = 0;
    while (pos < cleaned.length()) {
        size_t start = cleaned.find('[', pos);
        if (start == std::string::npos) break;
        
        size_t end = cleaned.find(']', start);
        if (end == std::string::npos) break;
        
        std::string rowStr = cleaned.substr(start, end - start + 1);
        matrix.push_back(parseFloatArray(rowStr));
        
        pos = end + 1;
    }
    
    return matrix;
}

std::string SimpleJSON::trim(const std::string& str) {
    size_t start = str.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    
    size_t end = str.find_last_not_of(" \t\n\r");
    return str.substr(start, end - start + 1);
}

std::string SimpleJSON::extractValue(const std::string& json, const std::string& key) {
    std::string searchKey = "\"" + key + "\"";
    size_t keyPos = json.find(searchKey);
    if (keyPos == std::string::npos) return "";
    
    size_t colonPos = json.find(':', keyPos);
    if (colonPos == std::string::npos) return "";
    
    size_t valueStart = colonPos + 1;
    
    // Skip whitespace
    while (valueStart < json.length() && (json[valueStart] == ' ' || json[valueStart] == '\t' || json[valueStart] == '\n')) {
        valueStart++;
    }
    
    if (valueStart >= json.length()) return "";
    
    size_t valueEnd;
    if (json[valueStart] == '{') {
        // Object value
        int braceCount = 1;
        valueEnd = valueStart + 1;
        while (valueEnd < json.length() && braceCount > 0) {
            if (json[valueEnd] == '{') braceCount++;
            else if (json[valueEnd] == '}') braceCount--;
            valueEnd++;
        }
        valueEnd--; // Include the closing brace
    } else if (json[valueStart] == '[') {
        // Array value
        int bracketCount = 1;
        valueEnd = valueStart + 1;
        while (valueEnd < json.length() && bracketCount > 0) {
            if (json[valueEnd] == '[') bracketCount++;
            else if (json[valueEnd] == ']') bracketCount--;
            valueEnd++;
        }
        valueEnd--; // Include the closing bracket
    } else {
        // Simple value (string, number, etc.)
        valueEnd = json.find_first_of(",}\n", valueStart);
        if (valueEnd == std::string::npos) valueEnd = json.length();
        valueEnd--; // Don't include the delimiter
    }
    
    return trim(json.substr(valueStart, valueEnd - valueStart + 1));
}
