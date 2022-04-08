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
    const Memory<Matrix3x3, RAM>& As,
    Memory<Matrix3x3, RAM>& Us,
    Memory<Matrix3x3, RAM>& Vs) const
{
    Memory<Vector, RAM> Ss(Us.size());
    calcUSV(As, Us, Ss, Vs);
}

void SVD::calcUSV(const Memory<Matrix3x3, RAM>& As,
        Memory<Matrix3x3, RAM>& Us,
        Memory<Vector, RAM>& Ss,
        Memory<Matrix3x3, RAM>& Vs) const
{

    const Eigen::Matrix3f* Aeig = reinterpret_cast<const Eigen::Matrix3f*>(As.raw());
    Eigen::Matrix3f* Ueig = reinterpret_cast<Eigen::Matrix3f*>(Us.raw());
    Eigen::Vector3f* Seig = reinterpret_cast<Eigen::Vector3f*>(Ss.raw());
    Eigen::Matrix3f* Veig = reinterpret_cast<Eigen::Matrix3f*>(Vs.raw());

    #pragma omp parallel for
    for(size_t i=0; i<As.size(); i++)
    {
        // map to eigen
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(Aeig[i], Eigen::ComputeFullU | Eigen::ComputeFullV);
        Ueig[i] = svd.matrixU();
        Seig[i] = svd.singularValues();
        Veig[i] = svd.matrixV();
    }
}

} // namespace rmagine