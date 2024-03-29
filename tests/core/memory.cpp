#include <iostream>
#include <memory>
#include <type_traits>
#include <rmagine/types/Memory.hpp>
#include <rmagine/types/sensor_models.h>
#include <rmagine/util/StopWatch.hpp>

#include <rmagine/util/prints.h>


using namespace rmagine;

void test_cpu()
{

    // flat
    std::cout << "1D Array" << std::endl;
    Memory<float, RAM> arr(10000);

    std::cout << "2D Array" << std::endl;

    Memory<Memory<float, RAM>, RAM> mat(100);
    mat[0].resize(10);
    mat[0][0] = 10.0;

    std::cout << "3D Array" << std::endl;
    Memory<Memory<Memory<float, RAM>, RAM>, RAM> arr3d;

    
    arr3d.resize(10);
    arr3d[0].resize(10);
    arr3d[0][0].resize(10);
    arr3d[0][0][0] = 2.0;

    std::cout << arr3d[0][0][0] << std::endl;
}


// void test_gpu()
// {
//     Memory<float, RAM_CUDA> arr_cpu(10000);

//     for(size_t i=0; i<10000; i++)
//     {
//         arr_cpu[i] = i;
//     }

//     Memory<float, VRAM_CUDA> arr;
//     arr.resize(5);
//     arr = arr_cpu;


//     Memory<float, RAM> dest;
//     dest = arr;

//     std::cout << dest[8] << std::endl;

// }


void test_sensor_models()
{
    O1DnModel_<RAM> model;
    model.height = 100;

    Memory<O1DnModel_<RAM>, RAM> model_mem(1);
    model_mem[0] = model;
}

void init(MemView<float>& a)
{
    for(size_t i=0; i<a.size(); i++)
    {
        a[i] = i;
    }
}

template<typename DataT>
void print(const MemView<DataT>& a)
{
    for(size_t i=0; i<a.size(); i++)
    {
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;
}

template<typename DataT>
Mem<DataT> add(const MemView<DataT>& a, const MemView<DataT>& b )
{
    Mem<DataT> c(a.size());
    #pragma omp parallel for
    for(size_t i=0; i<a.size(); i++)
    {
        c[i] = a[i] + b[i];
    }
    return c;
}

template<typename DataT>
void add(const MemView<DataT>& a, const MemView<DataT>& b, MemView<DataT>& c)
{
    #pragma omp parallel for
    for(size_t i=0; i<a.size(); i++)
    {
        c[i] = a[i] + b[i];
    }
}

#define yet int blaaaa=0

void problem_I_cannot_fix(yet)
{
    std::cout << "Try to cause some problems" << std::endl;
    std::shared_ptr<Memory<float> > bla;
    bla.reset(new Memory<float>(10));
    {
        std::cout << "Make MemoryView" << std::endl;
        MemoryView<float>& bla_ = *bla;
        // remove original memory but not the view. What to do here?
        bla.reset();
        std::cout << "bla_.size: " << bla_.size() << std::endl;

        // illegal
        bla_[3] = 5.0;
        std::cout << "bla_: ";
        print(bla_);
        // this is a problem I
    }
}

void test_slicing_small()
{
    std::cout << "Test Constructors" << std::endl;
    Memory<float> a(10);
    Memory<float> b(10);
    Memory<float> c(10);

    std::cout << a << std::endl;

    init(a);
    init(b);

    std::cout << "a: ";
    print(a);
    
    std::cout << "Test Copy constructors" << std::endl;
    // copy constructor
    // no op
    {
        MemoryView<float>& a_ = a;
        a_[2] = 5.0;
        std::cout << "a: ";
        print(a);
    }
    // copy
    {
        Memory<float> b = a;
    }

    // Slicing copy
    {
        a(0,5) = a(5,10);
        std::cout << "a: ";
        print(a);

        // TODO: check this
        Memory<float> d(5);
        init(d);
        a(0,5) = d;
        std::cout << "Assign Memory to Slice" << std::endl;
        std::cout << "a: ";
        print(a);

        std::cout << "Assing Slice to Memory" << std::endl;
        d = a(5,10);
        std::cout << "d: ";
        print(d);
    }

    // Slicing functions
    {
        c = add(a, b);
        std::cout << "c: ";
        print(c);
        c(0,2) = add(a(1,3), b(0,2));
        std::cout << "c: ";
        print(c);

        auto c_ = c(0,2);
        add(a(1,3), b(0,2), c_);
        std::cout << "c: ";
        print(c);
    }

    // old copy
    {
        Memory<float> bla;
        bla = a;
    }

}

void test_slicing_large()
{
    std::cout << "Test slicing large" << std::endl;

    size_t Nfloats = 100000000;

    StopWatch sw;
    double el;

    { // normal version
        Memory<float> data(Nfloats);
        size_t chunkSize = 1000;

        sw();
        for(size_t i=0; i<data.size(); i+=chunkSize)
        {
            for(size_t j=0; j<chunkSize; j++)
            {
                data[i + j] = j;
            }
        }
        el = sw();
        std::cout << "- Normal: " << el << "s" << std::endl;
        print(data(Nfloats-10, Nfloats));
    }

    { // sliced version
        Memory<float> data(Nfloats);
        size_t chunkSize = 1000;
        sw();
        for(size_t i=0; i<data.size(); i+=chunkSize)
        {
            auto slice = data(i, i+chunkSize);
            for(size_t j=0; j<chunkSize; j++)
            {
                slice[j] = j;
            }
        }
        el = sw();
        std::cout << "- Sliced: " << el << "s" << std::endl;
        print(data(Nfloats-10, Nfloats));
    }
}

MemoryView<float> func()
{
    // this should not work
    Memory<float> data(100);
    return data(0,5);
}

int main(int argc, char** argv)
{
    std::cout << "Rmagine Tests: Memory" << std::endl;

    test_cpu();
    test_sensor_models();

    test_slicing_small();
    test_slicing_large();

    return 0;
}