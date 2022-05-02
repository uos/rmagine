#include <iostream>
#include <rmagine/types/Memory.hpp>
#include <rmagine/types/MemoryCuda.hpp>
#include <rmagine/types/sensor_models.h>


using namespace rmagine;


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

    void setMem(const Memory<O1DnModel_<RAM>, RAM>& bla)
    {
        mem = bla;
    }

    void printSomething()
    {
        std::cout << mem->getHeight() << std::endl;
    }
    
private:
    Memory<O1DnModel_<RAM>, RAM> mem;

};

void test_sensor_models()
{
    O1DnModel_<RAM> model;
    model.height = 100;

    Memory<O1DnModel_<RAM>, RAM> model_mem(1);
    model_mem[0] = model;

    TestClass bla;
    bla.setMem(model_mem);
    bla.printSomething();

}


class DataView {
public:
    DataView(float* mem, size_t N)
    :m_mem(mem)
    ,m_size(N)
    {
        
    }

    float& operator[](unsigned long idx)
    {
        return m_mem[idx];
    }

    const float& operator[](unsigned long idx) const
    {
        return m_mem[idx];
    }

    DataView slice(unsigned int idx_start, unsigned int idx_end)
    {
        return DataView(m_mem + idx_start, idx_end - idx_start);
    }

    size_t size() const
    {
        return m_size;
    }

protected:
    float* m_mem;
    size_t m_size;
};


class Data : public DataView
{
public:
    using Base = DataView;

    Data(size_t N)
    :DataView((float*)malloc(N * sizeof(float)), N)
    {
        std::cout << "Data_ construct" << std::endl;
    }

    ~Data()
    {
        std::cout << "Data destruct" << std::endl;
        free(m_mem);
    }

protected:
    using Base::m_mem;
    using Base::m_size;
};

Data add(const DataView& a, const DataView& b )
{
    Data c(a.size());
    return c;
}

void test_view()
{
    Data a(1000);
    Data b(1000);

    auto a_ = a.slice(10, 20);
    auto b_ = b.slice(10, 20);

    Data c = add(a_, b_);
    // std::cout << "Test View" << std::endl;

    // Memory<float, RAM> data(1000);

    // auto view = data.slice(40, 50);

    // for(size_t i=0; i<data.size(); i++)
    // {
    //     data[i] = i;
    // }

    // size_t N = 10;
    // for(size_t i=0; i<data.size(); i+=N)
    // {
    //     auto view = data.slice(i, i+N);
    //     std::cout << "(" << i << ", " << i+N << "), " << view.size() << std::endl;
    //     for(size_t j=0; j<N; j++)
    //     {
    //         view[j] = i;
    //     }
    // }

    // for(size_t i=0; i<data.size(); i++)
    // {
    //     std::cout << data[i] << ", ";
    // }
    // std::cout << std::endl;
}

int main(int argc, char** argv)
{
    std::cout << "Rmagine Tests: Memory" << std::endl;

    // test_cpu();

    // test_gpu();

    // test_sensor_models();

    test_view();

    return 0;
}