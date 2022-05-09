#include "my_lib.h"

// add and write result to existing memory
void add(const rm::MemoryView<float, rm::RAM>& a,
    const rm::MemoryView<float, rm::RAM>& b,
    rm::MemoryView<float, rm::RAM>& res)
{
    // here you could add omp intructions
    for(size_t i=0; i<a.size(); i++)
    {
        res[i] = a[i] + b[i];
    }
}

// add and return new memory
rm::Memory<float, rm::RAM> add(
    const rm::MemoryView<float, rm::RAM>& a,
    const rm::MemoryView<float, rm::RAM>& b)
{
    rm::Memory<float, rm::RAM> res(a.size());
    add(a, b, res);
    return res;
}
