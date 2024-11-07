#include "Memory.hpp"

namespace rmagine
{

//// MEMORY VIEW
// template<typename DataT, typename MemT>
// MemoryView<DataT, MemT>::MemoryView()
// :m_mem(nullptr)
// ,m_size(0)
// {
//     // std::cout << "[MemoryView::MemoryView(DataT*, size_t)]" << std::endl;
// }

template<typename DataT, typename MemT>
MemoryView<DataT, MemT>::MemoryView(DataT* mem, size_t N)
:m_mem(mem)
,m_size(N)
{
    // std::cout << "[MemoryView::MemoryView(DataT*, size_t)]" << std::endl;
}

template<typename DataT, typename MemT>
MemoryView<DataT, MemT>::~MemoryView()
{
    // Do nothing on destruction
}

template<typename DataT, typename MemT>
MemoryView<DataT, MemT>& MemoryView<DataT, MemT>::operator=(const MemoryView<DataT, MemT>& o)
{
    // std::cout << "[MemoryView::operator=(MemoryView)]" << std::endl;
    #if !defined(NDEBUG)
    if(o.size() != this->size())
    {
        throw std::runtime_error("MemoryView MemT assignment of different sizes");
    }
    #endif // NDEBUG
    copy(o, *this);
    return *this;
}

template<typename DataT, typename MemT>
template<typename MemT2>
MemoryView<DataT, MemT>& MemoryView<DataT, MemT>::operator=(const MemoryView<DataT, MemT2>& o)
{
    #if !defined(NDEBUG)
    if(o.size() != this->size())
    {
        throw std::runtime_error("MemoryView MemT2 assignment of different sizes");
    }
    #endif // NDEBUG
    copy(o, *this);   
    return *this;
}

template<typename DataT, typename MemT>
DataT* MemoryView<DataT, MemT>::raw()
{
    return m_mem;
}

template<typename DataT, typename MemT>
const DataT* MemoryView<DataT, MemT>::raw() const {
    return m_mem;
}

/// MEMORY
template<typename DataT, typename MemT>
Memory<DataT, MemT>::Memory()
:Base(nullptr, 0)
{
    // std::cout << "[Memory::Memory()]" << std::endl;
}

template<typename DataT, typename MemT>
Memory<DataT, MemT>::Memory(size_t N)
:Base(MemT::template alloc<DataT>(N), N)
{
    // std::cout << "[Memory::Memory(size_t)]" << std::endl;
}

template<typename DataT, typename MemT>
Memory<DataT, MemT>::Memory(const MemoryView<DataT, MemT>& o)
:Memory(o.size())
{
    copy(o, *this);
}

template<typename DataT, typename MemT>
Memory<DataT, MemT>::Memory(const Memory<DataT, MemT>& o)
:Memory(o.size())
{
    // std::cout << "[Memory] Copy" << std::endl;
    copy(o, *this);
}

template<typename DataT, typename MemT>
template<typename MemT2>
Memory<DataT, MemT>::Memory(const MemoryView<DataT, MemT2>& o)
:Memory(o.size())
{
    copy(o, *this);
}

template<typename DataT, typename MemT>
Memory<DataT, MemT>::Memory(Memory<DataT, MemT>&& o) noexcept
:Base(o.m_mem, o.m_size)
{
    // std::cout << "[Memory] Move" << std::endl;
    // move
    o.m_mem = nullptr;
    o.m_size = 0;
}



template<typename DataT, typename MemT>
Memory<DataT, MemT>::~Memory()
{
    // std::cout << "[Memory] Destructor" << std::endl;
    MemT::free(m_mem, m_size);
}

template<typename DataT, typename MemT>
void Memory<DataT, MemT>::resize(size_t N) 
{
    if(m_mem != nullptr)
    {
        // initialized -> resize
        m_mem = MemT::realloc(m_mem, m_size, N);
    } else {
        // not initialized -> make new buffer of size N
        m_mem = MemT::template alloc<DataT>(N);
    }
    m_size = N;
}

template<typename DataT, typename MemT>
Memory<DataT, MemT>& Memory<DataT, MemT>::operator=(
    const MemoryView<DataT, MemT>& o)
{
    // std::cout << "[Memory::operator=(MemoryView)]" << std::endl;
    if(o.size() != this->size())
    {
        this->resize(o.size());
    }
    Base::operator=(o);
    return *this;
}

template<typename DataT, typename MemT>
template<typename MemT2>
Memory<DataT, MemT>& Memory<DataT, MemT>::operator=(
    const MemoryView<DataT, MemT2>& o)
{
    if(o.size() != this->size())
    {
        this->resize(o.size());
    }
    Base::operator=(o);
    return *this;
}

//// RAM
template<typename DataT>
DataT* RAM::alloc(size_t N)
{
    DataT* ret = static_cast<DataT*>(malloc(N * sizeof(DataT)));

    if constexpr( !std::is_trivially_constructible<DataT>::value )
    {
        for(size_t i=0; i<N; i++)
        {
            new (&ret[i]) DataT();
        }
    }

    return ret;
}

template<typename DataT>
DataT* RAM::realloc(DataT* mem, size_t Nold, size_t Nnew)
{
    DataT* ret = static_cast<DataT*>(::realloc(mem, Nnew * sizeof(DataT)));
    
    if constexpr( !std::is_trivially_constructible<DataT>::value )
    {
        // construct new elements
        size_t id_b = Nold;
        size_t id_e = Nnew;

        if(ret != mem)
        {
            // beginnings differ: recall all constructors
            id_b = 0;
        }

        for(size_t i=id_b; i < id_e; i++)
        {
            new (&ret[i]) DataT();
        }
    }

    return ret;
}

template<typename DataT>
void RAM::free(DataT* mem, size_t N)
{
    if constexpr( !std::is_trivially_destructible<DataT>::value )
    {
        // we need to destruct the elements first
        // std::cout << "Call buffers desctructors..." << std::endl;
        for(size_t i=0; i<N; i++)
        {
            mem[i].~DataT();
        }
    }

    if(N > 0)
    {
        // std::cout << "Free " << mem << std::endl;
        ::free(mem);
    }
}



} // namespace rmagine