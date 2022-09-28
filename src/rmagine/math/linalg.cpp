#include "rmagine/math/linalg.h"

#include <Eigen/Dense>

namespace rmagine
{

Matrix4x4 compose(const Transform& T, const Vector3& scale)
{
    Matrix4x4 M;
    M.set(T);

    Matrix4x4 S;
    S.setIdentity();
    S(0,0) = scale.x;
    S(1,1) = scale.y;
    S(2,2) = scale.z;

    return M * S;
}

Matrix4x4 compose(const Transform& T, const Matrix3x3& S)
{
    Matrix4x4 M;
    M.set(T);

    Matrix4x4 S_;
    S_.setZeros();
    for(size_t i=0; i<3; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            S_(i,j) = S(i,j);
        }
    }
    S_(3,3) = 1.0;

    return M * S_;
}

void decompose(const Matrix4x4& M, Transform& T, Matrix3x3& S)
{
    Eigen::Matrix4f Meig;
    for(size_t i=0; i<4; i++)
    {
        for(size_t j=0; j<4; j++)
        {
            Meig(i, j) = M(i, j);
        }
    }

    Eigen::Affine3f A;
    A.matrix() = Meig;


    Eigen::Matrix3f Reig;
    Eigen::Matrix3f Seig;
    A.computeRotationScaling(&Reig, &Seig);
    
    Eigen::Vector3f t = A.translation();
    
    Matrix3x3 R;
    for(size_t i=0; i<3; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            R(i,j) = Reig(i,j);
            S(i,j) = Seig(i,j);
        }
    }

    T.t = {t.x(), t.y(), t.z()};
    T.R.set(R);
}

void decompose(const Matrix4x4& M, Transform& T, Vector3& scale)
{
    Matrix3x3 S;
    decompose(M, T, S);

    // TODO: check if S is diagonal

    scale.x = S(0,0);
    scale.y = S(1,1);
    scale.z = S(2,2);
}



} // namespace rmagine