#include "rmagine/math/SVD.hpp"
#include "rmagine/types/Memory.hpp"
#include <assert.h>
#include <Eigen/Dense>

namespace rmagine {


SVD::SVD()
{
   
}


SVD::~SVD()
{
    
}

void SVD::calcUV(
    const Matrix3x3& A,
    Matrix3x3& U,
    Matrix3x3& V) const
{
    // reinterpret cast: data order is the same
    // TODO: test eigen mapping. this could be a cleaner solution
    const Eigen::Matrix3f* Aeig = reinterpret_cast<const Eigen::Matrix3f*>(&A);
    Eigen::Matrix3f* Ueig = reinterpret_cast<Eigen::Matrix3f*>(&U);
    Eigen::Matrix3f* Veig = reinterpret_cast<Eigen::Matrix3f*>(&V);
    
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(Aeig[0], Eigen::ComputeFullU | Eigen::ComputeFullV);
    Ueig[0] = svd.matrixU();
    Veig[0] = svd.matrixV();
}

void SVD::calcUSV(const Matrix3x3& A,
    Matrix3x3& U,
    Vector& S,
    Matrix3x3& V) const
{
    const Eigen::Matrix3f* Aeig = reinterpret_cast<const Eigen::Matrix3f*>(&A);
    Eigen::Matrix3f* Ueig = reinterpret_cast<Eigen::Matrix3f*>(&U);
    Eigen::Vector3f* Seig = reinterpret_cast<Eigen::Vector3f*>(&S);
    Eigen::Matrix3f* Veig = reinterpret_cast<Eigen::Matrix3f*>(&V);
    
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(Aeig[0], Eigen::ComputeFullU | Eigen::ComputeFullV);
    Ueig[0] = svd.matrixU();
    Seig[0] = svd.singularValues();
    Veig[0] = svd.matrixV();
}

void SVD::calcUV(
    const MemoryView<Matrix3x3, RAM>& As,
    MemoryView<Matrix3x3, RAM>& Us,
    MemoryView<Matrix3x3, RAM>& Vs) const
{
    #pragma omp parallel for
    for(size_t i=0; i<As.size(); i++)
    {
        // map to eigen
        calcUV(As[i], Us[i], Vs[i]);
    }
}

void SVD::calcUSV(const MemoryView<Matrix3x3, RAM>& As,
        MemoryView<Matrix3x3, RAM>& Us,
        MemoryView<Vector, RAM>& Ss,
        MemoryView<Matrix3x3, RAM>& Vs) const
{
    #pragma omp parallel for
    for(size_t i=0; i<As.size(); i++)
    {
        // map to eigen
        calcUSV(As[i], Us[i], Ss[i], Vs[i]);
    }
}

} // namespace rmagine