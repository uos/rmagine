#include <iostream>
#include <memory>
#include <type_traits>
#include <rmagine/types/Memory.hpp>
#include <rmagine/util/StopWatch.hpp>

#include <rmagine/math/memory_math.h>


using namespace rmagine;

static int chapter_counter = 0;

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
    chapter_counter++;
    std::cout << chapter_counter << ". Memory: " << std::endl;
    
    
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
    chapter_counter++;
    std::cout << chapter_counter << ". MemoryView" << std::endl;
    
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

    chapter_counter++;
    std::cout << chapter_counter << ". Test Slicing" << std::endl;

    Memory<float> a(100);
    
    a.resize(200);
    fill(a);

    // a_ is an memory view object. memory specif stuff not allowed. only view
    MemoryView<float> a_ = a;
    if(equal(a_[2], a[2]) && a_.raw() == a.raw())
    {
        std::cout << "- Mem to View correct" << std::endl;
    } else {
        std::cout << "- Mem to View wrong" << std::endl;
    }
    
    MemoryView<float> a__ = a(10, 20);
    if(equal(a__[5], a[15]) && a__.raw() == (a.raw() + 10) && a__.size() == 10)
    {
        std::cout << "- Slice correct" << std::endl;
    } else {
        std::cout << "- Slice wrong" << std::endl;
    }

    auto res = make_collection(a__);
    if(res.a.size() == a__.size()
        && equal(res.a[2], a__[2])
        && res.a.raw() != a__.raw())
    {
        std::cout << "- Make Collection 1: correct" << std::endl;
    } else {
        std::cout << "- Make Collection 1: wrong" << std::endl;
    }


    MyCollection res2;
    res2.a = a;

    // auto res2 = make_collection2(a);
    if(res2.a.size() == a.size()
        && equal(res2.a[2], a[2])
        && res2.a.raw() != a.raw())
    {
        std::cout << "- Make Collection 2: correct" << std::endl;
    } else {
        std::cout << "- Make Collection 2: wrong" << std::endl;
    }
    
    Memory<float> b = a(0,5);
    if(b.raw() != a.raw())
    {
        std::cout << "- Mem b = a(x,y) correct" << std::endl;
    } else {
        std::cout << "- Mem b = a(x,y) wrong" << std::endl;
    }

    {
        MemoryView<float> av = a(0,5);
        if(av.raw() == a.raw())
        {
            std::cout << "- MemView b = a(x,y) correct" << std::endl;
        } else {
            std::cout << "- MemView b = a(x,y) wrong" << std::endl;
        }
    }
    
    // stuff that should not be allowed
    // TODO: eleminate these problems
    // MemoryView<float> p1 = problem1();
    // problem2();
}

void test_slicing_2()
{
    chapter_counter++;
    std::cout << chapter_counter << ". Test Slicing User" << std::endl;

    Memory<float> a(100);
    fill(a);

    // copy memory into a new one
    Memory<float> b = a(20, 30);

    // shallow manipulate memory
    a(20, 30) = a(30, 40);

    

}

int main(int argc, char** argv)
{
    std::cout << "Rmagine Tests: Memory Slicing" << std::endl;

    test_memory();
    test_memory_view();

    test_slicing_1();
    test_slicing_2();


    return 0;
}