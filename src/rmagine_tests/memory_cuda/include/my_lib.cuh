#ifndef MYLIB_MY_LIB_CUH
#define MYLIB_MY_LIB_CUH

#include <rmagine/types/MemoryCuda.hpp>

namespace rm = rmagine;

// add and write result to existing memory
void add(const rm::MemoryView<float, rm::VRAM_CUDA>& a,
    const rm::MemoryView<float, rm::VRAM_CUDA>& b,
    rm::MemoryView<float, rm::VRAM_CUDA>& res);

// add and return new memory
rm::Memory<float, rm::VRAM_CUDA> add(
    const rm::MemoryView<float, rm::VRAM_CUDA>& a,
    const rm::MemoryView<float, rm::VRAM_CUDA>& b);

#endif // MYLIB_MY_LIB_CUH