#include <iostream>


#include "rmagine/math/types.h"

#include <rmagine/math/math.h>
#include <rmagine/util/StopWatch.hpp>

#include <rmagine/util/exceptions.h>
#include <rmagine/util/prints.h>

#include <rmagine/math/SVD2.hpp>

#include <Eigen/Dense>

#include <stdint.h>
#include <string.h>

namespace rm = rmagine;

Eigen::Matrix3f& eigen_view(rm::Matrix3x3& M)
{
    return *reinterpret_cast<Eigen::Matrix3f*>( &M );
}

template<typename DataT, int Rows, int Cols>
void testSVD(const Eigen::Matrix<DataT, Rows, Cols>& A)
{
    Eigen::JacobiSVD<Eigen::Matrix<DataT, Rows, Cols> > svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
}

template<typename DataT, unsigned Rows, unsigned Cols>
void testSVD(const rm::Matrix_<DataT, Rows, Cols>& A)
{
    using AMatT = rm::Matrix_<DataT, Rows, Cols>;
    using UMatT = typename rm::svd_dims<AMatT>::U;
    using WMatT = typename rm::svd_dims<AMatT>::W;
    using VMatT = typename rm::svd_dims<AMatT>::V;



    UMatT U = UMatT::Zeros();
    WMatT W = WMatT::Zeros();
    VMatT V = VMatT::Zeros();

    rm::svd(A, U, W, V);
}

void svdTestWithPrints()
{
    rm::Matrix3x3 Arm;

    Eigen::Matrix3f Aeig = Eigen::Matrix3f::Random(3, 3);
    for(size_t i=0; i<3; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            Arm(i, j) = Aeig(i, j);
        }
    }
    

    std::cout << "A: " << std::endl << Aeig << std::endl;

    std::cout << std::endl;
    std::cout << "Eigen: " << std::endl;
    

    Eigen::JacobiSVD<Eigen::Matrix3f> svdeig(Aeig, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // std::cout << "U: " << std::endl << svdeig.matrixU() << std::endl;
    // std::cout << "V: " << std::endl << svdeig.matrixV() << std::endl;

    std::cout << "U * V.T" << std::endl;
    std::cout << svdeig.matrixU() * svdeig.matrixV().transpose() << std::endl;

    // rm::SVD2 svdrm;
    // svdrm.decompose(Arm);

    std::cout << std::endl;
    std::cout << "Rmagine: " << std::endl;
    // std::cout  << "U: " << svdrm.u << std::endl;
    // std::cout << "V: " << svdrm.v << std::endl;
    // std::cout << "w: " << svdrm.w << std::endl;

    std::cout << "U * V.T" << std::endl;
    // std::cout << svdrm.u * svdrm.v.transpose() << std::endl;

    // std::cout << svdrm.w << std::endl;

    using AMat = rm::Matrix_<float, 3, 3>;
    using UMat = typename rm::svd_dims<AMat>::U;
    using WMat = typename rm::svd_dims<AMat>::W;
    using VMat = typename rm::svd_dims<AMat>::V;

    UMat Urm = UMat::Zeros();
    WMat Wrm = WMat::Zeros();
    VMat Vrm = VMat::Zeros();

    rm::svd(Arm, Urm, Wrm, Vrm);

    std::cout << Urm * Vrm.T() << std::endl;
}

template<int N, int M>
void runtimeTest()
{
    rm::Matrix_<float, N, M> Arm;

    Eigen::Matrix<float, N, M> Aeig = Eigen::Matrix<float, N, M>::Random(N, M);
    for(size_t i=0; i<N; i++)
    {
        for(size_t j=0; j<M; j++)
        {
            Arm(i, j) = Aeig(i, j);
        }
    }
    
    rm::StopWatch sw;
    double el_eig, el_rm;

    sw();
    testSVD(Aeig);
    el_eig = sw();

    sw();
    testSVD(Arm);
    el_rm = sw();

    std::cout << "N = " << N << ", M = " << M << std::endl;
    std::cout << "- eigen: " << el_eig*1000.0 << "ms" << std::endl;
    std::cout << "- rmagine: " << el_rm*1000.0 << "ms" << std::endl;
}


template<int N, int M>
void equalityTest()
{
    rm::Matrix_<float, N, M> Arm;

    Eigen::Matrix<float, N, M> Aeig = Eigen::Matrix<float, N, M>::Random(N, M);
    for(size_t i=0; i<N; i++)
    {
        for(size_t j=0; j<M; j++)
        {
            Arm(i, j) = Aeig(i, j);
        }
    }
    
    rm::StopWatch sw;
    double el_eig, el_rm;

    std::cout << "A: " << std::endl;
    std::cout << Aeig << std::endl;

    Eigen::JacobiSVD<Eigen::Matrix<float, N, M> > svdeig(Aeig, Eigen::ComputeFullU | Eigen::ComputeFullV);

    auto s = svdeig.singularValues();

    Eigen::Matrix<float, N, M> Seig = Eigen::Matrix<float, N, M>::Zero();
    for(size_t i=0; i<s.rows(); i++)
    {
        Seig(i, i) = s(i);
    }
    // std::cout << "Seig: " << std::endl;
    // std::cout << Seig << std::endl;

    std::cout << "Eigen: " << std::endl;
    auto uvt_eig = svdeig.matrixU() * Seig * svdeig.matrixV().transpose();

    std::cout << uvt_eig << std::endl;

    float error_eig = 0.0;
    for(size_t i=0; i<N; i++)
    {
        for(size_t j=0; j<M; j++)
        {
            error_eig += abs(uvt_eig(i, j) - Aeig(i, j));
        }
    }

    std::cout << "- error: " << error_eig << std::endl;

    using AMat = rm::Matrix_<float, N, M>;
    using UMat = typename rm::svd_dims<AMat>::U;
    using WMat = typename rm::svd_dims<AMat>::W;
    using VMat = typename rm::svd_dims<AMat>::V;

    UMat Urm = UMat::Zeros();
    WMat Wrm = WMat::Zeros();
    VMat Vrm = VMat::Zeros();

    rm::svd(Arm, Urm, Wrm, Vrm);

    auto uvt_rm = Urm * Wrm * Vrm.T();

    std::cout << "Rmagine: " << std::endl;
    std::cout << uvt_rm << std::endl;

    float error_rm = 0.0;
    for(size_t i=0; i<N; i++)
    {
        for(size_t j=0; j<M; j++)
        {
            error_rm += abs(uvt_rm(i, j) - Arm(i, j));
        }
    }

    std::cout << "- error: " << error_eig << std::endl;

}

int main(int argc, char** argv)
{
    srand((unsigned int) time(0));

    
    svdTestWithPrints();

    runtimeTest<20, 30>();
    equalityTest<5, 10>();



    return 0;
}

