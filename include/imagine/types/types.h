#ifndef IMAGINE_TYPES_TYPES_HPP
#define IMAGINE_TYPES_TYPES_HPP

#include <cstdint>

namespace imagine {

struct Interval {
    float min;
    float max;
};

struct DiscreteInterval
{
    float min;
    float max;
    float step;
    uint32_t size;
};

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


// TODOs: 
// - check if Eigen::Matrix3f raw data is same
// - check if math is correct
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

} // namespace imagine

#endif // IMAGINE_TYPES_TYPES_HPP