#include <iostream>
#include <memory>
#include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/math/memory_math.cuh>
#include <rmagine/util/exceptions.h>
#include <rmagine/util/prints.h>



using namespace rmagine;

void fill_sequence(MemoryView<int> a)
{
    for(size_t i=0; i<a.size(); i++)
    {
        a[i] = i;
    }
}

void fill_sequence(MemoryView<int, RAM_CUDA> a)
{
    for(size_t i=0; i<a.size(); i++)
    {
        a[i] = i;
    }
}


void fill_sequence(MemoryView<float> a)
{
    for(size_t i=0; i<a.size(); i++)
    {
        a[i] = i;
    }
}

void fill_sequence(MemoryView<float, RAM_CUDA> a)
{
    for(size_t i=0; i<a.size(); i++)
    {
        a[i] = i;
    }
}


void memory_basic()
{
    Memory<float> data(100);
    fill_sequence(data);

    // upload
    Memory<float, VRAM_CUDA> data_d = data;

    if(data_d.size() != data.size())
    {
        RM_THROW(CudaException, "Memory error");
    }


    Memory<float, RAM_CUDA> arr_cpu(10000);
    fill_sequence(arr_cpu);

    Memory<float, VRAM_CUDA> arr;
    arr.resize(5);
    arr = arr_cpu;

    std::cout << arr << std::endl;


    Memory<float, RAM> dest;
    dest = arr;

    std::cout << dest[8] << std::endl;
}

void error_if_not_equal(
    MemoryView<int, RAM> ram,
    MemoryView<int, RAM_CUDA> ram_cuda)
{
    for(size_t i=0; i<ram.size(); i++)
    {
        if(ram_cuda[i] != ram[i])
        {
            RM_THROW(CudaException, "Memory error, RAM != RAM_CUDA");
        }
    }
}

void memory_unified()
{
    std::cout << "MEM UNIFIED" << std::endl;
    Memory<int, RAM> ram;
    
    Memory<int, RAM_CUDA> ram_cuda;
    Memory<int, VRAM_CUDA> vram_cuda;
    Memory<int, UNIFIED_CUDA> unified_cuda;


    ram.resize(10);
    fill_sequence(ram);

    ram_cuda = ram;
    error_if_not_equal(ram, ram_cuda);
    vram_cuda = ram_cuda;
    unified_cuda = vram_cuda;
    
    for(size_t i=0; i<ram.size(); i++)
    {
        std::cout << unified_cuda[i] << std::endl;
    }
}

void memory_slicing()
{
    Memory<float> data(100);
    fill_sequence(data);

    // upload to device
    Memory<float, VRAM_CUDA> data1_d = data;
    Memory<float, VRAM_CUDA> data2_d = data;

    // use own cuda library: my_lib_cuda
    
    // add version 1
    auto res_d = addNxN(data1_d, data2_d);

    // or sliced version
    res_d(50, 60) = data1_d(10,20) + data2_d(20,30);


    data(0, 10) = res_d(50, 60);

    if(fabs(data[0] - 30.0) > 0.000001)
    {
        RM_THROW(CudaException, "Cuda Memory Slicing Error");
    }
}

int main(int argc, char** argv)
{
    std::cout << "Rmagine Tests: Memory Cuda" << std::endl;

    memory_basic();
    memory_slicing();
    memory_unified();
    

    return 0;
}