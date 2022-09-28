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