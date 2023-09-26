/*
 * Copyright (c) 2022, University Osnabrück
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

/**
 * @file
 * 
 * @brief Datatype hashing
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_UTIL_HASING_H
#define RMAGINE_UTIL_HASING_H

#include <memory>
#include <utility>

namespace rmagine
{

template<typename T>
struct weak_hash
{
    size_t operator()(const std::weak_ptr<T>& elem) const 
    {
        if(auto sh = elem.lock())
        {
            return std::hash<decltype(sh)>()(sh);
        } else {
            return 0;
        }
    }
};

template<typename T>
struct weak_equal_to
{
    bool operator()(const std::weak_ptr<T>& lhs, const std::weak_ptr<T>& rhs) const 
    {
        auto lptr = lhs.lock();
        auto rptr = rhs.lock();
        return lptr == rptr;
    }
};

template<typename T>
struct weak_less {
    bool operator() (const std::weak_ptr<T> &lhs, const std::weak_ptr<T> &rhs) const 
    {
        auto rptr = rhs.lock();
        if (!rptr) 
        {
             // nothing after expired pointer 
            return false;
        }

        auto lptr = lhs.lock();
        if (!lptr) 
        {
            // every not expired after expired pointer
            return true; 
        }

        return lptr < rptr;
    }
};

} // namespace rmagine

#endif // RMAGINE_UTIL_HASHING_H