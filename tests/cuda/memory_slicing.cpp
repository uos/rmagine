#include <iostream>
#include <memory>
#include <type_traits>
#include <rmagine/types/Memory.hpp>
#include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/util/StopWatch.hpp>

#include <rmagine/math/math.cuh>

#include <rmagine/util/prints.h>

using namespace rmagine;

static int chapter_counter = 0;

template<typename DataT, typename MemT>
void fill(MemoryView<DataT, MemT>& a)
{
    for(size_t i=0; i<a.size(); i++)
    {
        a[i] = i;
    }
}

// HEADER .h
Memory<float, VRAM_CUDA> func_host(
    const MemoryView<float, VRAM_CUDA>& input);

// CODE: .cu

// kernel: 
// __global__ void func_kernel(const float* input, size_t Ninput, float* res)
// { ... }

Memory<float, VRAM_CUDA> func_host(
    const MemoryView<float, VRAM_CUDA>& input)
{
    Memory<float, VRAM_CUDA> res(input.size());
    // func_kernel<<<100, 1>>>(input.raw(), input.size(), res.raw());
    return res;
}

void lib_example()
{

}

void test_slicing()
{
    chapter_counter++;
    std::cout << chapter_counter << ". Test Slicing User" << std::endl;

    Memory<float> ah(100);
    fill(ah);

    Memory<float, VRAM_CUDA> ad;
    ad = ah(20, 30);
    // b = a(20, 30); // upload 
    std::cout << ad.size() << std::endl;

    ah(70, 75) = ad(0, 5);
    std::cout << ah[0] << std::endl;

    // copy memory into a new one
    // Memory<float> b = a(20, 30);

    // // shallow manipulate memory
    ad(5, 10) = ad(0, 5);
    ah(80, 90) = ad;
    std::cout << ah[80] << std::endl;


    // math

    Memory<Transform, RAM> T(1);
    T[0] = Transform::Identity();
    Memory<Transform, VRAM_CUDA> Td = T;

    Memory<Vector, RAM> V(1000);
    for(size_t i=0; i<V.size(); i++)
    {
        V[i] = {(float)i, (float)i, (float)i};
    }
    Memory<Vector, VRAM_CUDA> Vd = V;

    Memory<Vector, VRAM_CUDA> resd(1000);

    for(size_t i=0; i<10; i++)
    {
        auto slice = resd(i * 100, i * 100 + 100);
        mult1xN(Td, Vd(i * 100, i * 100 + 100), slice);
    }

    Memory<Vector, RAM> res = resd;
    std::cout << "Math Test 1: " << res[500] << " " << res[999] << std::endl;

    // clear
    for(size_t i=0; i<res.size(); i++)
    {
        res[i] = {0.0, 0.0, 0.0};
    }
    resd = res;


    for(size_t i=0; i<10; i++)
    {
        resd(i * 100, i * 100 + 100) = mult1xN(Td, Vd(i * 100, i * 100 + 100) );
    }

    res = resd;
    std::cout << "Math Test 2: " << res[500] << " " << res[999] << std::endl;



}

int main(int argc, char** argv)
{
    std::cout << "Rmagine Tests: Memory Slicing" << std::endl;

    test_slicing();


    return 0;
}