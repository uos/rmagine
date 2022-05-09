#include <iostream>
#include <memory>
#include <type_traits>
#include <rmagine/types/Memory.hpp>
#include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/util/StopWatch.hpp>


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

void test_slicing_2()
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

}

int main(int argc, char** argv)
{
    std::cout << "Rmagine Tests: Memory Slicing" << std::endl;

    test_slicing_2();


    return 0;
}