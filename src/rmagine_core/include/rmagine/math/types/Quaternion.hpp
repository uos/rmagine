#ifndef RMAGINE_MATH_QUATERNION_HPP
#define RMAGINE_MATH_QUATERNION_HPP

#include "definitions.h"
#include <rmagine/types/shared_functions.h>
#include <initializer_list>

namespace rmagine
{


/**
 * @brief Quaternion_<DataT> type
 * 
 */
template<typename DataT>
struct Quaternion_
{
    // Data
    DataT x;
    DataT y;
    DataT z;
    DataT w;

    RMAGINE_FUNCTION
    static Quaternion_<DataT> Identity()
    {
        Quaternion_<DataT> ret;
        ret.setIdentity();
        return ret;
    }

    RMAGINE_INLINE_FUNCTION
    void setIdentity();

    /**
     * @brief Invert this Quaternion
     * 
     * @return Quaternion_<DataT> 
     */
    RMAGINE_INLINE_FUNCTION
    Quaternion_<DataT> inv() const;

    RMAGINE_INLINE_FUNCTION
    void invInplace();

    /**
     * @brief Multiply quaternion
     * 
     * @param q2 
     * @return Quaternion_<DataT> 
     */
    RMAGINE_INLINE_FUNCTION
    Quaternion_<DataT> mult(const Quaternion_<DataT>& q2) const;

    RMAGINE_INLINE_FUNCTION
    void multInplace(const Quaternion_<DataT>& q2);

    /**
     * @brief Rotate a vector with this quaternion
     * 
     * @param p 
     * @return Vector 
     */
    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> mult(const Vector3_<DataT>& p) const;

    RMAGINE_INLINE_FUNCTION
    DataT dot(const Quaternion_<DataT>& q) const;

    RMAGINE_INLINE_FUNCTION
    DataT l2normSquared() const;

    RMAGINE_INLINE_FUNCTION
    DataT l2norm() const;

    RMAGINE_INLINE_FUNCTION
    Quaternion_<DataT> normalize() const;
    
    RMAGINE_INLINE_FUNCTION
    void normalizeInplace();

    RMAGINE_INLINE_FUNCTION
    void set(const Matrix_<DataT, 3, 3>& M);

    RMAGINE_INLINE_FUNCTION
    void set(const EulerAngles_<DataT>& e);

    // TODO: Quatenrion from rotation around an axis v by an angle a
    // RMAGINE_INLINE_FUNCTION
    // void set(const Vector3& v, float a);

    // OPERATORS
    RMAGINE_INLINE_FUNCTION
    Quaternion_<DataT> operator~() const 
    {
        return inv();
    }

    RMAGINE_INLINE_FUNCTION
    Quaternion_<DataT> operator*(const Quaternion_<DataT>& q2) const 
    {
        return mult(q2);
    }

    RMAGINE_INLINE_FUNCTION
    Vector3_<DataT> operator*(const Vector3_<DataT>& p) const
    {
        return mult(p);
    }

    RMAGINE_INLINE_FUNCTION
    Quaternion_<DataT>& operator*=(const Quaternion_<DataT>& q2)
    {
        multInplace(q2);
        return *this;
    }

    /////////////////////
    // CASTING

    /**
     * @brief Quaternion -> EulerAngles
     * 
     * @return EulerAngles_<DataT> 
     */
    RMAGINE_INLINE_FUNCTION
    operator EulerAngles_<DataT>() const;

    /**
     * @brief Quaternion -> Rotation Matrix
     * 
     * @return Matrix_<DataT, 3, 3> 
     */
    RMAGINE_INLINE_FUNCTION
    operator Matrix_<DataT, 3, 3>() const;

    /**
     * @brief Data Type cast to ConvT
     * 
     * @tparam ConvT 
     * @return RMAGINE_INLINE_FUNCTION 
     */
    template<typename ConvT>
    RMAGINE_INLINE_FUNCTION
    Quaternion_<ConvT> cast() const;
};

} // namespace rmagine

#include "Quaternion.tcc"

#endif // RMAGINE_MATH_QUATERNION_HPP