#include <iostream>
#include <memory>
#include <type_traits>
#include <rmagine/types/Memory.hpp>
#include <rmagine/util/StopWatch.hpp>


using namespace rmagine;


struct MyCollection
{
    Memory<float> a;
    Memory<float> b;
};

void fill(MemoryView<float>& a)
{
    for(size_t i=0; i<a.size(); i++)
    {
        a[i] = i;
    }
}

MemoryView<float> problem1()
{
    Memory<float> a(100);
    // a will be deleted at return. Slice will be invalid. Solution right now: hust dont return slices 
    return a(0, 10);
}

void problem2()
{
    std::shared_ptr<Memory<float> > a;
    a.reset(new Memory<float>(100));

    MemoryView<float> a_ = a->slice(10, 20);

    a.reset();

    // now a_ is invalid. How to handle?
}

MyCollection make_collection(const MemoryView<float>& a)
{
    MyCollection res;
    res.a = a;
    res.b = a;
    return res;
}

MyCollection make_collection2(Memory<float> a)
{
    MyCollection res;
    res.a = a;
    res.b = a;
    return res;
}

void test_slicing_1()
{
    // a is an memory object. memory specific stuff allowed: malloc, free, resize
    Memory<float> a(100);
    a.resize(200);
    fill(a);

    // a_ is an memory view object. memory specif stuff not allowed. only view
    MemoryView<float>& a_ = a;

    std::cout << a_[10] << " == " << a[10] << std::endl;


    MemoryView<float> a__ = a(10, 20);
    std::cout << a__[5] << " == " << a[15] << std::endl;

    auto res = make_collection(a__);
    std::cout << res.a[0] << " == " << a[10] << std::endl;

    // auto res2 = make_collection2(a);
    // std::cout << res.a[20] << " == " << a[20] << std::endl;

    // stuff that should not be allowed
    // TODO: eleminate these problems
    // MemoryView<float> p1 = problem1();
    // problem2();

    // copy memory
    Memory<float> b = a;

    size_t b_size = b.size();
    size_t a_size = a.size();

    if(b.raw() != a.raw() 
        && abs(b[30] - a[30]) < 0.001
        && a.size() == b.size() )
    {
        std::cout << "- Copy: Memory -> Memory: correct" << std::endl;
    } else {
        std::cout << "- Copy: Memory -> Memory: wrong" << std::endl;
        if(b.raw() == a.raw())
        {
            std::cout << "-- pointer should not be the same after copy!" << std::endl;
        }
    }
    std::cout << b[30] << " == " << a[30] << std::endl;

    // move memory: a should be invalid
    Memory<float> c = std::move(a);
    if(a.raw() == nullptr)
    {
        std::cout << "- Move: Memory -> Memory: correct" << std::endl;
    } else {
        std::cout << "- Move: Memory -> Memory: wrong" << std::endl;
        if(a.raw() != nullptr)
        {
            std::cout << "-- moved memory should have nullptr" << std::endl;
        }
        
    }


}


int main(int argc, char** argv)
{
    std::cout << "Rmagine Tests: Memory Slicing" << std::endl;

    test_slicing_1();


    return 0;
}