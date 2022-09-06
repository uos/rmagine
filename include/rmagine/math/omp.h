#ifndef RMAGINE_MATH_OMP_H
#define RMAGINE_MATH_OMP_H

#include "types.h"

#pragma omp declare reduction( + : rmagine::Vector3 : omp_out += omp_in )
#pragma omp declare reduction( + : rmagine::Matrix3x3 : omp_out += omp_in )
#pragma omp declare reduction( + : rmagine::Matrix4x4 : omp_out += omp_in )


#endif // RMAGINE_MATH_OMP_H