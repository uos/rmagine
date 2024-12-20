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

#ifndef RMAGINE_MEMORY_HPP
#define RMAGINE_MEMORY_HPP

#include <type_traits>
#include <iostream>
#include <cstring>
#include <type_traits>
#include <exception>

#include <rmagine/types/shared_functions.h>

namespace rmagine {

struct RAM;

class RMAGINE_API MemoryResizeError : public std::runtime_error {
public:
  MemoryResizeError()
  :std::runtime_error("rmagine: cannot resize memory view!")
  {

  }
};

template<typename DataT, typename MemT = RAM>
class RMAGINE_API MemoryView {
public:
    using DataType = DataT;
    using MemType = MemT;
    
    // use this 
    MemoryView() = delete;

    static MemoryView<DataT, MemT> Empty()
    {
        return MemoryView<DataT, MemT>(nullptr, 0);
    }

    MemoryView(DataT* mem, size_t N);

    // no virtual: we dont want to destruct memory of a view
    ~MemoryView();

    RMAGINE_INLINE_FUNCTION
    bool empty() const 
    {
        return m_mem == nullptr;
    }

    // Copy for assignment of same MemT
    MemoryView<DataT, MemT>& operator=(
        const MemoryView<DataT, MemT>& o);
    
    // Copy for assignment of different MemT
    template<typename MemT2>
    MemoryView<DataT, MemT>& operator=(
        const MemoryView<DataT, MemT2>& o);

    // TODO: Check CUDA usage (in kernels)
    RMAGINE_FUNCTION
    DataT* raw();
    
    RMAGINE_FUNCTION
    const DataT* raw() const;

    RMAGINE_FUNCTION
    DataT* operator->()
    {
        return raw();
    }

    RMAGINE_FUNCTION
    const DataT* operator->() const
    {
        return raw();
    }

    RMAGINE_FUNCTION
    DataT& at(size_t idx)
    {
        return m_mem[idx];
    }

    RMAGINE_FUNCTION
    const DataT& at(size_t idx) const
    {
        return m_mem[idx];
    }

    RMAGINE_FUNCTION
    DataT& operator[](size_t idx)
    {
        return m_mem[idx];
    }

    RMAGINE_FUNCTION
    const DataT& operator[](size_t idx) const
    {
        return m_mem[idx];
    }

    RMAGINE_FUNCTION
    const DataT& operator*() const {
        return *m_mem;
    }

    RMAGINE_FUNCTION
    DataT& operator*() {
        return *m_mem;
    }

    RMAGINE_FUNCTION
    size_t size() const {
        return m_size;
    }

    // Shall we introduce this? 
    // virtual void resize(size_t N) {
    //     throw MemoryResizeError();
    // }

    MemoryView<DataT, MemT> slice(size_t idx_start, size_t idx_end)
    {
        return MemoryView<DataT, MemT>(m_mem + idx_start, idx_end - idx_start);
    }

    // problem solved: just dont write to it
    const MemoryView<DataT, MemT> slice(size_t idx_start, size_t idx_end) const 
    {
        return MemoryView<DataT, MemT>(m_mem + idx_start, idx_end - idx_start);
    }

    MemoryView<DataT, MemT> operator()(size_t idx_start, size_t idx_end)
    {
        return slice(idx_start, idx_end);
    }

    const MemoryView<DataT, MemT> operator()(size_t idx_start, size_t idx_end) const
    {
        return slice(idx_start, idx_end);
    }

protected:
    DataT* m_mem = nullptr;
    size_t m_size = 0;
};

template<typename DataT, typename MemT = RAM>
using MemView = MemoryView<DataT, MemT>;

template<typename T>
MemoryView<T, RAM> make_view(T& data)
{
  return MemoryView<T>(&data, 1);
}

template<typename DataT, typename MemT = RAM>
class RMAGINE_API Memory 
: public MemoryView<DataT, MemT> 
{
public:
    using Base = MemoryView<DataT, MemT>;

    Memory();
    Memory(size_t N);
    // Copy Constructor
    Memory(const MemoryView<DataT, MemT>& o);
    Memory(const Memory<DataT, MemT>& o);

    template<typename MemT2>
    Memory(const MemoryView<DataT, MemT2>& o);

    Memory(Memory<DataT, MemT>&& o) noexcept;

    ~Memory();

    // virtual void resize(size_t N);
    void resize(size_t N);

    // Copy for assignment of same MemT
    Memory<DataT, MemT>& operator=(
        const MemoryView<DataT, MemT>& o);

    // Why do I need them
    // if this function is not: 
    // Memory a(1);
    // Memory b(10);
    // a = b; -> would call the operator= of MemView class
    inline Memory<DataT, MemT>& operator=(const Memory<DataT, MemT>& o)
    {
        const MemoryView<DataT, MemT>& c = o;
        return operator=(c);
    }

    // Copy for assignment of different MemT
    template<typename MemT2>
    Memory<DataT, MemT>& operator=(const MemoryView<DataT, MemT2>& o);

    template<typename MemT2>
    inline Memory<DataT, MemT>& operator=(const Memory<DataT, MemT2>& o)
    {
        const MemoryView<DataT, MemT2>& c = o;
        return operator=(c);
    }

protected:

    using Base::m_mem;
    using Base::m_size;
};

template<typename DataT, typename MemT = RAM>
using Mem = Memory<DataT, MemT>;

// RAM specific

// RAM Type
struct RAM {
    
    template<typename DataT>
    static DataT* alloc(size_t N);

    template<typename DataT>
    static DataT* realloc(DataT* mem, size_t Nold, size_t Nnew);

    template<typename DataT>
    static void free(DataT* mem, size_t N);
};

// Some functions that can be specialized:
template<typename DataT, typename SMT, typename TMT>
void copy(const MemoryView<DataT, SMT>& from, MemoryView<DataT, TMT>& to);


template<typename DataT>
void copy(const MemoryView<DataT, RAM>& from, MemoryView<DataT, RAM>& to)
{
    std::memcpy(to.raw(), from.raw(), sizeof(DataT) * from.size() );
}

template<typename DataT>
void copy(const DataT& from, MemoryView<DataT, RAM>& to)
{
    std::memcpy(to.raw(), &from, sizeof(DataT));
}

template<typename DataT>
void copy(const MemoryView<DataT, RAM>& from, DataT& to)
{
    std::memcpy(&to, from.raw(), sizeof(DataT));
}

} // namespace rmagine


// Edit: const return type required for safety
//
// hmm that is difficult. Return a const object is not possible.
// const MemoryView a;
// MemoryView b = a.slice(0,2);
// b[0] = 1;
// - here we want to return an object that cannot be modified
// 
// possible fix: MemoryView<const ...> but than the object itself
// enable this for none const 

// little complicated here. If DataT is not const -> make return value const
// template<typename U=DataT> 
// typename std::enable_if<!std::is_const<U>::value, MemoryView<const DataT, MemT> >::type // Return Value
// slice(size_t idx_start, size_t idx_end) const 
// {
//     std::cout << "Make data const!" << std::endl;
//     return MemoryView<const DataT, MemT>(m_mem + idx_start, idx_end - idx_start);
// }

// // if DataT is const return same data type
// template<typename U=DataT> 
// typename std::enable_if<std::is_const<U>::value, MemoryView<DataT, MemT> >::type // Return value
// slice(size_t idx_start, size_t idx_end) const 
// {
//     std::cout << "Let data const!" << std::endl;
//     return MemoryView<DataT, MemT>(m_mem + idx_start, idx_end - idx_start);
// }

#include "Memory.tcc"

#endif // RMAGINE_MEMORY_HPP
