#ifndef SIMPLE_JSON_H
#define SIMPLE_JSON_H

#include <string>
#include <map>
#include <vector>

// Simple lightweight JSON parser alternative to jsoncpp
class SimpleJSON {
public:
    static std::map<std::string, std::vector<std::vector<float>>> parseMatrixFile(const std::string& filepath);
    
private:
    static std::vector<float> parseFloatArray(const std::string& arrayStr);
    static std::vector<std::vector<float>> parseMatrix(const std::string& matrixStr);
    static std::string trim(const std::string& str);
    static std::string extractValue(const std::string& json, const std::string& key);
};

#endif // SIMPLE_JSON_H
