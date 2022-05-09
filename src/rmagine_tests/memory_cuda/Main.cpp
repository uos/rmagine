#include <iostream>
#include <memory>
#include <rmagine/types/MemoryCuda.hpp>

#include "my_lib.h"

using namespace rmagine;

void fill_sequence(MemoryView<float>& a)
{
    for(size_t i=0; i<a.size(); i++)
    {
        a[i] = i;
    }
}


void library_example()
{
    Memory<float> data(100);
    fill_sequence(data);

    // upload to device
    Memory<float, VRAM_CUDA> data1_d = data;
    Memory<float, VRAM_CUDA> data2_d = data;


    // use own cuda library
    
    // add version 1
    auto res_d = add(data1_d, data2_d);

    // or sliced version
    res_d(20, 30) = add(data1_d(0,10), data2_d(10,20));

    // or version without temorary memory
    add(data1_d, data2_d, res_d);

    // sliced without new memory
    auto res_d_ = res_d(20, 30);
    add(data1_d(0,10), data2_d(10,20), res_d_);
}

int main(int argc, char** argv)
{
    std::cout << "Rmagine Tests: Memory Cuda" << std::endl;

    return 0;
}