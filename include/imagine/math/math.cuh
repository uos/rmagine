#ifndef IMAGINE_MATH_MATH_CUH
#define IMAGINE_MATH_MATH_CUH

#include <imagine/math/types.h>
#include <cuda_runtime.h>
#include <imagine/types/MemoryCuda.hpp>

namespace imagine 
{

void multNxN(
    const Memory<Quaternion, VRAM_CUDA>& A,
    const Memory<Quaternion, VRAM_CUDA>& B,
    Memory<Quaternion, VRAM_CUDA>& C);

Memory<Quaternion, VRAM_CUDA> multNxN(
    const Memory<Quaternion, VRAM_CUDA>& A, 
    const Memory<Quaternion, VRAM_CUDA>& B);

void multNxN(
    const Memory<Quaternion, VRAM_CUDA>& A,
    const Memory<Vector, VRAM_CUDA>& b, 
    Memory<Vector, VRAM_CUDA>& c);

Memory<Vector, VRAM_CUDA> multNxN(
    const Memory<Quaternion, VRAM_CUDA>& A,
    const Memory<Vector, VRAM_CUDA>& b);

void multNxN(
    const Memory<Transform, VRAM_CUDA>& T,
    const Memory<Vector, VRAM_CUDA>& x,
    Memory<Vector, VRAM_CUDA>& c);

Memory<Vector, VRAM_CUDA> multNxN(
    const Memory<Transform, VRAM_CUDA>& T,
    const Memory<Vector, VRAM_CUDA>& x);

void multNxN(
    const Memory<Matrix3x3, VRAM_CUDA>& M,
    const Memory<Vector, VRAM_CUDA>& x,
    Memory<Vector, VRAM_CUDA>& c);

Memory<Vector, VRAM_CUDA> multNxN(
    const Memory<Matrix3x3, VRAM_CUDA>& M,
    const Memory<Vector, VRAM_CUDA>& x);

} // namespace imagine

#endif // IMAGINE_MATH_MATH_CUH