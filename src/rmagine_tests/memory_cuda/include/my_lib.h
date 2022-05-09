#ifndef MYLIB_MY_LIB_H
#define MYLIB_MY_LIB_H

#include <rmagine/types/Memory.hpp>

namespace rm = rmagine;

// add and write result to existing memory
void add(const rm::MemoryView<float, rm::RAM>& a,
    const rm::MemoryView<float, rm::RAM>& b,
    rm::MemoryView<float, rm::RAM>& res);

// add and return new memory
rm::Memory<float, rm::RAM> add(
    const rm::MemoryView<float, rm::RAM>& a,
    const rm::MemoryView<float, rm::RAM>& b);

#endif // MYLIB_MY_LIB_H