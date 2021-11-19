#ifndef IMAGINE_UTIL_CUDA_FUNCTIONS_HPP
#define IMAGINE_UTIL_CUDA_FUNCTIONS_HPP


#ifdef __CUDA_ARCH__
#define IMAGINE_HOST_DEVICE __inline__ __host__ __device__ 
#else
#define IMAGINE_HOST_DEVICE inline
#endif

#endif // IMAGINE_UTIL_CUDA_FUNCTIONS_HPP