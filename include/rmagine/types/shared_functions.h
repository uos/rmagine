#ifndef RMAGINE_TYPES_SHARED_FUNCTIONS_H
#define RMAGINE_TYPES_SHARED_FUNCTIONS_H

#ifdef __CUDA_ARCH__
#define RMAGINE_FUNCTION __host__ __device__
#define RMAGINE_INLINE_FUNCTION __inline__ __host__ __device__ 
#else
#define RMAGINE_FUNCTION
#define RMAGINE_INLINE_FUNCTION inline
#endif

#endif // RMAGINE_TYPES_SHARED_FUNCTIONS_H