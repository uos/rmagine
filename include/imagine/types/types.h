#ifndef IMAGINE_TYPES_TYPES_HPP
#define IMAGINE_TYPES_TYPES_HPP

#include <cstdint>

namespace imagine {

struct Vector {
    float x;
    float y;
    float z;
};

using Point = Vector;

struct Quaternion
{
    float x;
    float y;
    float z;
    float w;
};

// 16*4 Byte Transform struct 
struct Transform {
    Quaternion R;
    Vector t;
    uint32_t stamp;
};

struct Matrix3x3 {
    float data[3][3];

    #ifdef __CUDA_ARCH__
    __host__ __device__ 
    #endif
    float* operator[](const unsigned int i) {
        return data[i];
    };

    #ifdef __CUDA_ARCH__
    __host__ __device__ 
    #endif
    const float* operator[](const unsigned int i) const {
        return data[i];
    };
};

// using Matrix2x2 = float[2][2];
// using Matrix3x3 = float[9];
// using Matrix4x4 = float[4][4];
// typedef float[3] Matrix3x3;


} // namespace imagine

#endif // IMAGINE_TYPES_TYPES_HPP