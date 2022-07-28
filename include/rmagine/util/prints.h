#ifndef RMAGINE_UTIL_PRINTS_H
#define RMAGINE_UTIL_PRINTS_H

#include <iostream>
#include <rmagine/math/types.h>

inline std::ostream& operator<<(std::ostream& os, const rmagine::Vector& v)
{
    os << "v[" << v.x << "," << v.y << "," << v.z << "]";

    return os;
}

inline std::ostream& operator<<(std::ostream& os, const rmagine::Quaternion& q)
{
    os << "q[" << q.x << "," << q.y << "," << q.z << "," << q.w << "]";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const rmagine::Transform& T)
{
    os << "T[" << T.t << ", " << T.R << "]";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const rmagine::Matrix3x3& M)
{
    os << "M3x3[\n";
    os << " " << M(0, 0) << " " << M(0, 1) << " " << M(0, 2) << "\n";
    os << " " << M(1, 0) << " " << M(1, 1) << " " << M(1, 2) << "\n";
    os << " " << M(2, 0) << " " << M(2, 1) << " " << M(2, 2) << "\n";
    os << "]";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const rmagine::Matrix4x4& M)
{
    os << "M4x4[\n";
    os << " " << M(0, 0) << " " << M(0, 1) << " " << M(0, 2) << " " << M(0, 3) << "\n";
    os << " " << M(1, 0) << " " << M(1, 1) << " " << M(1, 2) << " " << M(1, 3) << "\n";
    os << " " << M(2, 0) << " " << M(2, 1) << " " << M(2, 2) << " " << M(2, 3) << "\n";
    os << " " << M(3, 0) << " " << M(3, 1) << " " << M(3, 2) << " " << M(3, 3) << "\n";
    os << "]";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const rmagine::EulerAngles& e)
{
    os << "E [" << e.roll << ", " << e.pitch << ", " << e.yaw << "]";
    return os;
}

inline std::ostream& operator<<(std::ostream& os, const rmagine::AABB& aabb)
{
    os << "AABB [" << aabb.min <<  " - " << aabb.max << "]";
    return os;
}


#endif // RMAGINE_UTIL_PRINTS_H