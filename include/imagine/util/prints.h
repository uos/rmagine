#ifndef IMAGINE_UTIL_PRINTS_H
#define IMAGINE_UTIL_PRINTS_H

#include <iostream>
#include <imagine/math/types.h>

std::ostream& operator<<(std::ostream& os, const imagine::Vector& v)
{
    os << "v[" << v.x << "," << v.y << "," << v.z << "]";

    return os;
}

std::ostream& operator<<(std::ostream& os, const imagine::Quaternion& q)
{
    os << "q[" << q.x << "," << q.y << "," << q.z << "," << q.w << "]";
    return os;
}

std::ostream& operator<<(std::ostream& os, const imagine::Transform& T)
{
    os << "T[" << T.t << ", " << T.R << "]";
    return os;
}

#endif // IMAGINE_UTIL_PRINTS_H