
#include <algorithm>


namespace rmagine
{

template<typename DataT>
Vector3_<DataT> min(const Vector3_<DataT>& a, const Vector3_<DataT>& b)
{
    Vector3_<DataT> ret;
    ret.x = std::min(a.x, b.x);
    ret.y = std::min(a.y, b.y);
    ret.z = std::min(a.z, b.z);
    return ret;
}

template<typename DataT>
Vector3_<DataT> max(const Vector3_<DataT>& a, const Vector3_<DataT>& b)
{
    Vector3_<DataT> ret;
    ret.x = std::max(a.x, b.x);
    ret.y = std::max(a.y, b.y);
    ret.z = std::max(a.z, b.z);
    return ret;
}

} // namespace rmagine