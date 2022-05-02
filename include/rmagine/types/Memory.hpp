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

#include <rmagine/types/SharedFunctions.hpp>

namespace rmagine {

struct RAM;




template<typename DataT, typename MemT = RAM>
class MemoryView;



template<typename DataT, typename MemT>
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
    DataT& at(unsigned long idx)
    {
        return m_mem[idx];
    }

    RMAGINE_FUNCTION
    const DataT& at(unsigned long idx) const
    {
        return m_mem[idx];
    }

    RMAGINE_FUNCTION
    DataT& operator[](unsigned long idx)
    {
        return m_mem[idx];
    }

    RMAGINE_FUNCTION
    const DataT& operator[](unsigned long idx) const
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

    RMAGINE_FUNCTION
    MemoryView<DataT, MemT> slice(unsigned int idx_start, unsigned int idx_end);

protected:
    DataT* m_mem = nullptr;
    size_t m_size = 0;
    bool   owner = true;
};

// template<typename DataT, typename MemT>
// class MemoryView : public Memory<DataT, MemT>
// {
// public:
//     using Base = Memory<DataT, MemT>;

//     MemoryView(DataT* data, size_t N);
//     virtual ~MemoryView() override;

//     // RMAGINE_FUNCTION
//     // size_t size() const {
//     //     return m_size;
//     // }

//     // RMAGINE_FUNCTION
//     // DataT& operator[](unsigned long idx)
//     // {
//     //     return m_mem[idx];
//     // }

//     // RMAGINE_FUNCTION
//     // const DataT& operator[](unsigned long idx) const
//     // {
//     //     return m_mem[idx];
//     // }

//     // RMAGINE_FUNCTION
//     // DataT* raw();
    
//     // RMAGINE_FUNCTION
//     // const DataT* raw() const;

// protected:
//     using Base::m_mem;
//     using Base::m_size;
//     // DataT* m_mem;
//     // size_t m_size;
// };

// template<typename DataT, typename MemT>
// Memory<DataT, MemT> wrap(const DataT& data);

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
void copy(const Memory<DataT, SMT>& from, Memory<DataT, TMT>& to);


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

} // namespace rmagine

#include "Memory.tcc"

#endif // RMAGINE_MEMORY_HPP