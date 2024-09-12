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
 * @brief Bundle different structs into one.
 *
 * @date 03.10.2022
 * @author Alexander Mock
 * 
 * @copyright Copyright (c) 2022, University Osnabrück. All rights reserved.
 * This project is released under the 3-Clause BSD License.
 * 
 */

#ifndef RMAGINE_TYPES_BUNDLE_HPP
#define RMAGINE_TYPES_BUNDLE_HPP

#include <utility>

namespace rmagine {

/**
 * @brief CRTP Container for User defined elements.
 * 
 * Define your own intersection type that is passed to the Raycaster
 * you like. Allowed elements are in lvr2::intelem
 * 
 * @code
 * // define you intersection type
 * using MyIntType = Intersection<intelem::Point, intelem::Face>;
 * // creating the raycaster
 * RaycasterBasePtr<MyIntType> rc(new BVHRaycaster(meshbuffer));
 * // cast a ray
 * MyIntType intersection_result;
 * bool hit = rc->castRay(ray_origin, ray_direction, intersection_result);
 * // print:
 * if(hit)
 * {
 *      std::cout << intersection_result << std::endl;
 * }
 * @endcode
 * 
 * @tparam Tp List of intelem::* types  
 */
template<typename ...Tp>
struct Bundle : public Tp... 
{
public:
    static constexpr std::size_t N = sizeof...(Tp);
    using elems = std::tuple<Tp...>;

private:
    template <typename T, typename Tuple>
    struct has_type;

    template <typename T>
    struct has_type<T, std::tuple<>> : std::false_type {};

    template <typename T, typename U, typename... Ts>
    struct has_type<T, std::tuple<U, Ts...>> : has_type<T, std::tuple<Ts...>> {};

    template <typename T, typename... Ts>
    struct has_type<T, std::tuple<T, Ts...>> : std::true_type {};    

    template<typename F> 
    struct has_elem {
        static constexpr bool value = has_type<F, elems>::type::value;
    };

    template <typename T, typename Tuple>
    struct Index;

    template <typename T, class... Ts>
    struct Index<T, std::tuple<T, Ts...>> {
        static constexpr std::size_t value = 0;
    };

    template <class T, class U, class... Ts>
    struct Index<T, std::tuple<U, Ts...>> {
        static constexpr std::size_t value = 1 + Index<T, std::tuple<Ts...>>::value;
    };

    template<typename F> 
    struct IndexOf {
        static constexpr std::size_t value = Index<F, elems>::type::value;
    };

public:
    /**
     * @brief Check if GenericStruct container has a specific element.
     * 
     * @tparam F lvr2::intelem type
     * @return true 
     * @return false 
     */
    template<typename F>
    static constexpr bool has() {
        return has_elem<F>::value;
    }

    template<typename F>
    static constexpr std::size_t index_of() {
        return IndexOf<F>::value;
    }
};

} // namespace rmagine


#endif // RMAGINE_TYPES_BUNDLE_HPP