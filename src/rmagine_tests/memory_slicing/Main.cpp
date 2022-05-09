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

bool equal(float a, float b)
{
    return abs(a-b) < 0.001;
}

void test_memory()
{
    std::cout << "1. Memory: " << std::endl;
    
    
    {   // default
        Memory<float> a1(100);
        fill(a1);

        if(a1.size() == 100 && a1.raw() != nullptr)
        {
            std::cout << "- Constructor correct" << std::endl;
        } else {
            std::cout << "- Constructor wrong" << std::endl;
            return;
        }
    }


    {   // copy
        Memory<float> a1(100);
        fill(a1);

        Memory<float> a2 = a1;
        if(a2.size() == a1.size() && a2.raw() != a1.raw() && equal(a2[2], a1[2]) )
        {
            std::cout << "- Copy-Constructor correct" << std::endl;
        } else {
            std::cout << "- Copy-Constructor wrong" << std::endl;
            return;
        }
    }

    {   // move

        Memory<float> a1(100);
        fill(a1);
        Memory<float> a2 = a1;
        
        Memory<float> a3 = std::move(a1);
        if(a1.size() == 0 && a1.raw() == nullptr && equal(a3[2], a2[2]))
        {
            std::cout << "- Move-Constructor correct" << std::endl;
        } else {
            std::cout << "- Move-Constructor wrong" << std::endl;
            return;
        }
    }

    {   // assign-copy
        Memory<float> a1(100);
        fill(a1);

        Memory<float> a2(2);
        a2 = a1;

        if(a2.size() == a1.size() && a2.raw() != a1.raw() && equal(a2[2], a1[2]) )
        {
            std::cout << "- operator=(Memory) correct" << std::endl;
        } else {
            std::cout << "- operator=(Memory) wrong" << std::endl;
        }
    }

    {   // assign-move: should not work
        // Memory<float> a1(100);
        // fill(a1);

        // Memory<float> a2 = a1;

        // Memory<float> a3(2);
        // a3 = std::move(a1);

        // if(a1.size() == 0 && a1.raw() == nullptr && equal(a3[2], a2[2]) )
        // {
        //     std::cout << "- operator=(Memory) correct" << std::endl;
        // } else {
        //     std::cout << "- operator=(Memory) wrong" << std::endl;
        //     std::cout << "-- size==0: " << (a1.size() == 0) << std::endl;
        //     std::cout << "-- ptr==null: " << (a1.raw() == nullptr) << std::endl;
        //     std::cout << "-- data same: " << equal(a3[2], a2[2]) << std::endl;
        // }
    }
}

void test_memory_view()
{
    std::cout << "2. MemoryView" << std::endl;
    
    {   // simple casting conversion
        Memory<float> a1(100);
        MemoryView<float>& a1_ = a1;
        fill(a1_);

        if(a1.size() == a1_.size() 
            && a1.raw() == a1_.raw() 
            && equal(a1[5], 5.0) )
        {
            std::cout << "- Casting correct" << std::endl;
        } else {
            std::cout << "- Casting wrong" << std::endl;
            return;
        }
    }

    {
        Memory<float> a1(100);
        
        MemoryView<float>& a1_ = a1;
        fill(a1_);

        // copy constructor 
        Memory<float> b = a1_;
        
        if(b.raw() != a1_.raw() && equal(b[3], a1_[3]))
        {
            std::cout << "- Copy-Constructor Mem(MemView) correct" << std::endl; 
        } else {
            std::cout << "- Copy-Constructor Mem(MemView) wrong" << std::endl; 
        }
    }

    // Copy constructor MemView(MemView) does not exist because MemView cannot create memory itself
    // Move constructor MemView(MemView) does not exist because it would destroy the source memory

}

void test_slicing_1()
{
    // a is an memory object. memory specific stuff allowed: malloc, free, resize

    std::cout << "Test Slicing" << std::endl;

    
    
    // a.resize(200);
    // fill(a);

    // // a_ is an memory view object. memory specif stuff not allowed. only view
    // MemoryView<float>& a_ = a;
    // std::cout << a_[10] << " == " << a[10] << std::endl;

    // MemoryView<float> a__ = a(10, 20);
    
    // std::cout << a__[5] << " == " << a[15] << std::endl;

    // auto res = make_collection(a__);
    // std::cout << res.a[0] << " == " << a[10] << std::endl;
    // if(res.a.size() == a__.size()
    //     && abs(res.a[0] - a__[0]) < 0.001
    //     && res.a.raw() != a__.raw())
    // {
    //     std::cout << "- Make Collection 1: correct" << std::endl;
    // } else {
    //     std::cout << "- Make Collection 1: wrong" << std::endl;
    // }

    // std::cout << res.a[20] << " == " << a[20] << std::endl;

    // MyCollection res_bla;
    // Memory<float>& tmp = res_bla.a;
    // const MemoryView<float>& tmp2 = a;
    // tmp.operator=(tmp2);
    // res_bla.a = a;

    // if(res_bla.a.size() == )
    


    // return;


    // MyCollection res2;
    // res2.a = a;

    // // auto res2 = make_collection2(a);
    // if(res2.a.size() == a.size()
    //     && abs(res2.a[0] - a[0]) < 0.001
    //     && res2.a.raw() != a.raw())
    // {
    //     std::cout << "- Make Collection 2: correct" << std::endl;
    // } else {
    //     std::cout << "- Make Collection 2: wrong" << std::endl;
    // }
    
    
    // stuff that should not be allowed
    // TODO: eleminate these problems
    // MemoryView<float> p1 = problem1();
    // problem2();

    // copy memory
    // Memory<float> b = a;

    // if(b.raw() != a.raw() 
    //     && abs(b[30] - a[30]) < 0.001
    //     && a.size() == b.size() )
    // {
    //     std::cout << "- Copy: Memory -> Memory: correct" << std::endl;
    // } else {
    //     std::cout << "- Copy: Memory -> Memory: wrong" << std::endl;
    //     if(b.raw() == a.raw())
    //     {
    //         std::cout << "-- pointer should not be the same after copy!" << std::endl;
    //     }
    // }
    // std::cout << b[30] << " == " << a[30] << std::endl;

    // // move memory: a should be invalid
    // Memory<float> c = std::move(a);
    // if(a.raw() == nullptr)
    // {
    //     std::cout << "- Move: Memory -> Memory: correct" << std::endl;
    // } else {
    //     std::cout << "- Move: Memory -> Memory: wrong" << std::endl;
    //     if(a.raw() != nullptr)
    //     {
    //         std::cout << "-- moved memory should have nullptr" << std::endl;
    //     }
        
    // }


}


int main(int argc, char** argv)
{
    std::cout << "Rmagine Tests: Memory Slicing" << std::endl;

    test_memory();
    test_memory_view();

    test_slicing_1();


    return 0;
}