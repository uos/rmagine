#include <iostream>
#include <imagine/types/Memory.hpp>
#include <imagine/types/MemoryCuda.hpp>
#include <imagine/types/sensor_models.h>


using namespace imagine;


struct Foo {
    float x;
};

struct Bar {
    float x;

    Bar() : x(2.0) {}
};

void placement_new_example()
{
    Bar* a = (Bar*)malloc(sizeof(Bar) * 10);

    // // call constructor
    // a[0];
    // placement new
    for(size_t i=0; i<10; i++)
    {
        new (&a[i]) Bar();
    }

    std::cout << a[0].x << std::endl;
    std::cout << a[9].x << std::endl;

    free(a);
}

void realloc_test()
{
    int* a = (int*)malloc(sizeof(int) * 5);
    a[0] = 5;
    a = (int*)realloc(a, sizeof(int) * 10 );
    std::cout << a[0] << std::endl;
    free(a);
}

void free_test()
{
    int* a = nullptr;

    free(a);
}


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


void test_gpu()
{
    Memory<float, RAM_CUDA> arr_cpu(10000);

    for(size_t i=0; i<10000; i++)
    {
        arr_cpu[i] = i;
    }

    Memory<float, VRAM_CUDA> arr;
    arr.resize(5);
    arr = arr_cpu;


    Memory<float, RAM> dest;
    dest = arr;

    std::cout << dest[8] << std::endl;

}

class TestClass 
{
public:
    TestClass()
    :mem(1)
    {

    }

    ~TestClass()
    {
        
    }

    void setMem(const Memory<O1DnModel<RAM>, RAM>& bla)
    {
        mem = bla;
    }

    void printSomething()
    {
        std::cout << mem->getHeight() << std::endl;
    }
    
private:
    Memory<O1DnModel<RAM>, RAM> mem;

};

void test_sensor_models()
{
    O1DnModel<RAM> model;
    model.height = 100;

    Memory<O1DnModel<RAM>, RAM> model_mem(1);
    model_mem[0] = model;

    TestClass bla;
    bla.setMem(model_mem);
    bla.printSomething();

}

int main(int argc, char** argv)
{
    std::cout << "Imagine Tests: Memory" << std::endl;

    test_cpu();

    // test_gpu();

    test_sensor_models();

    return 0;
}