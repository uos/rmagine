#include "Memory.hpp"

namespace imagine
{


template<typename DataT, typename MemT>
Memory<DataT, MemT>::Memory()
:m_size(0)
,m_mem(nullptr)
{
    
}

template<typename DataT, typename MemT>
Memory<DataT, MemT>::Memory(size_t N)
:m_size(N)
,m_mem(MemT::template alloc<DataT>(N))
{
    
}

template<typename DataT, typename MemT>
Memory<DataT, MemT>::Memory(const Memory<DataT, MemT>& o)
:Memory(o.size())
{
    copy(o, *this);
}

template<typename DataT, typename MemT>
Memory<DataT, MemT>::Memory(Memory<DataT, MemT>&& o) noexcept
: m_mem(o.m_mem)
, m_size(o.m_size)
{
    o.m_size = 0;
    o.m_mem = nullptr;
}

template<typename DataT, typename MemT>
Memory<DataT, MemT>::~Memory()
{
    MemT::free(m_mem, m_size);
}

template<typename DataT, typename MemT>
Memory<DataT, MemT>& Memory<DataT, MemT>::operator=(const Memory<DataT, MemT>& o)
{
    if(o.size() != this->size())
    {
        this->resize(o.size());
    }
    copy(o, *this);
    return *this;
}

template<typename DataT, typename MemT>
template<typename MemT2>
Memory<DataT, MemT>& Memory<DataT, MemT>::operator=(const Memory<DataT, MemT2>& o)
{
    if(o.size() != this->size())
    {
        this->resize(o.size());
    }
    copy(o, *this);
    return *this;
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
DataT* Memory<DataT, MemT>::raw()
{
    return m_mem;
}

template<typename DataT, typename MemT>
const DataT* Memory<DataT, MemT>::raw() const {
    return m_mem;
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

    ::free(mem);
}



} // namespace imagine