/**
 * Copyright (c) 2021, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Memory.hpp
 *
 *  Created on: Jul 17, 2021
 *      Author: Alexander Mock
 */

#ifndef IMAGINE_MEMORY_HPP
#define IMAGINE_MEMORY_HPP

#include <type_traits>
#include <iostream>
#include <cstring>
#include <type_traits>

#include <imagine/types/SharedFunctions.hpp>

namespace imagine {

struct RAM;

template<typename DataT, typename MemT = RAM>
class Memory {
public:
    using DataType = DataT;
    using MemType = MemT;
    
    Memory();
    Memory(size_t N);
    // Copy Constructor
    Memory(const Memory<DataT, MemT>& o);
    // Move Constructor
    Memory(Memory<DataT, MemT>&& o) noexcept;

    ~Memory();

    // Copy for assignment of same MemT
    Memory<DataT, MemT>& operator=(const Memory<DataT, MemT>& o);
    
    // Copy for assignment of different MemT
    template<typename MemT2>
    Memory<DataT, MemT>& operator=(const Memory<DataT, MemT2>& o);

    void resize(size_t N);

    #ifdef __CUDA_ARCH__ 
    __host__ __device__ 
    #endif
    DataT* raw();
    
    #ifdef __CUDA_ARCH__ 
    __host__ __device__ 
    #endif
    const DataT* raw() const;

    #ifdef __CUDA_ARCH__ 
    __host__ __device__ 
    #endif
    DataT* operator->()
    {
        return raw();
    }

    #ifdef __CUDA_ARCH__ 
    __host__ __device__ 
    #endif
    const DataT* operator->() const
    {
        return raw();
    }

    #ifdef __CUDA_ARCH__ 
    __host__ __device__ 
    #endif
    DataT& at(unsigned long idx)
    {
        return m_mem[idx];
    }

    #ifdef __CUDA_ARCH__ 
    __host__ __device__ 
    #endif
    const DataT& at(unsigned long idx) const
    {
        return m_mem[idx];
    }

    #ifdef __CUDA_ARCH__ 
    __host__ __device__ 
    #endif
    DataT& operator[](unsigned long idx)
    {
        return m_mem[idx];
    }

    #ifdef __CUDA_ARCH__ 
    __host__ __device__ 
    #endif
    const DataT& operator[](unsigned long idx) const
    {
        return m_mem[idx];
    }

    #ifdef __CUDA_ARCH__ 
    __host__ __device__ 
    #endif
    const DataT& operator*() const {
        return *m_mem;
    }

    #ifdef __CUDA_ARCH__ 
    __host__ __device__ 
    #endif
    DataT& operator*() {
        return *m_mem;
    }

    #ifdef __CUDA_ARCH__ 
    __host__ __device__ 
    #endif
    size_t size() const {
        return m_size;
    }

private:
    DataT* m_mem;
    size_t m_size;
};

template<typename DataT, typename MemT>
Memory<DataT, MemT>::Memory()
:m_size(1)
{
    m_mem = reinterpret_cast<DataT*>(MemT::alloc(sizeof(DataT)));
}

template<typename DataT, typename MemT>
Memory<DataT, MemT>::Memory(size_t N)
:m_size(N)
{
    m_mem = reinterpret_cast<DataT*>(MemT::alloc(sizeof(DataT) * N)) ;
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
    MemT::free(m_mem);
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
void Memory<DataT, MemT>::resize(size_t N) {
    m_mem = reinterpret_cast<DataT*>(MemT::realloc(m_mem, sizeof(DataT) * N));
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

// Some functions that can be specialized:
template<typename DataT, typename SMT, typename TMT>
void copy(const Memory<DataT, SMT>& from, Memory<DataT, TMT>& to);

// template<typename DataT, typename MemT>
// Memory<DataT, MemT> wrap(const DataT& data);

// RAM specific

// RAM Type
struct RAM {
    
    static void* alloc(size_t N)
    {
        return malloc(N);
    }

    static void* realloc(void* mem, size_t N)
    {
        return ::realloc(mem, N);
    }

    static void free(void* mem)
    {
        if(mem)
        {
            ::free(mem);
        }
    }
};

template<typename DataT>
void copy(const Memory<DataT, RAM>& from, Memory<DataT, RAM>& to)
{
    std::memcpy(to.raw(), from.raw(), sizeof(DataT) * from.size() );
}

template<typename DataT>
void copy(const DataT& from, Memory<DataT, RAM>& to)
{
    std::memcpy(to.raw(), &from, sizeof(DataT));
}

template<typename DataT>
void copy(const Memory<DataT, RAM>& from, DataT& to)
{
    std::memcpy(&to, from.raw(), sizeof(DataT));
}

} // namespace imagine

#endif // IMAGINE_MEMORY_HPP