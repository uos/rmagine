#include <iostream>


#include <rmagine/math/types.h>

#include <rmagine/math/math.h>
#include <rmagine/util/StopWatch.hpp>

#include <rmagine/util/exceptions.h>
#include <rmagine/util/prints.h>

#include <rmagine/math/linalg.h>

#include <Eigen/Dense>

#include <stdint.h>
#include <string.h>

namespace rm = rmagine;


float compute_error(Eigen::Matrix3f gt, Eigen::Matrix3f m)
{
    float ret = 0.0;
    for(size_t i=0; i<3; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            ret += abs(gt(i, j) - m(i, j));
        }
    }
    return ret;
}

float compute_error(rm::Matrix3x3 gt, rm::Matrix3x3 m)
{
    float ret = 0.0;
    for(size_t i=0; i<3; i++)
    {
        for(size_t j=0; j<3; j++)
        {
            ret += abs(gt(i, j) - m(i, j));
        }
    }
    return ret;
}

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
void accuracyTest()
{
    std::cout << "Accuracy Test" << std::endl;
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

    // std::cout << "A: " << std::endl;
    // std::cout << Aeig << std::endl;

    Eigen::JacobiSVD<Eigen::Matrix<float, N, M> > svdeig(Aeig, Eigen::ComputeFullU | Eigen::ComputeFullV);

    auto s = svdeig.singularValues();

    Eigen::Matrix<float, N, M> Seig = Eigen::Matrix<float, N, M>::Zero();
    for(size_t i=0; i<s.rows(); i++)
    {
        Seig(i, i) = s(i);
    }
    // std::cout << "Seig: " << std::endl;
    // std::cout << Seig << std::endl;


    // std::cout << "Eigen: " << std::endl;
    auto uvt_eig = svdeig.matrixU() * Seig * svdeig.matrixV().transpose();

    // std::cout << uvt_eig << std::endl;

    float error_eig = 0.0;
    for(size_t i=0; i<N; i++)
    {
        for(size_t j=0; j<M; j++)
        {
            error_eig += abs(uvt_eig(i, j) - Aeig(i, j));
        }
    }

    std::cout << "- Eigen JacobiSVD error: " << error_eig << std::endl;

    using AMat = rm::Matrix_<float, N, M>;
    using UMat = typename rm::svd_dims<AMat>::U;
    using WMat = typename rm::svd_dims<AMat>::W;
    using VMat = typename rm::svd_dims<AMat>::V;

    UMat Urm = UMat::Zeros();
    WMat Wrm = WMat::Zeros();
    VMat Vrm = VMat::Zeros();

    rm::svd(Arm, Urm, Wrm, Vrm);

    auto uvt_rm = Urm * Wrm * Vrm.T();

    // std::cout << "Rmagine: " << std::endl;
    // std::cout << uvt_rm << std::endl;

    float error_rm = 0.0;
    for(size_t i=0; i<N; i++)
    {
        for(size_t j=0; j<M; j++)
        {
            error_rm += abs(uvt_rm(i, j) - Arm(i, j));
        }
    }

    std::cout << "- Rmagine error: " << error_eig << std::endl;
}


void parallelTest()
{
    std::cout << "parallelTest" << std::endl;
    // correct num_objects objects in parallel
    size_t num_objects = 1000000;

    std::vector<Eigen::Matrix3f> covs_eigen(num_objects);
    rm::Memory<rm::Matrix3x3, rm::RAM> covs_rm(num_objects);

    for(size_t obj_id=0; obj_id<num_objects; obj_id++)
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

        covs_eigen[obj_id] = Aeig;
        covs_rm[obj_id] = Arm;
    }

    std::cout << "First Mat: " << std::endl;
    std::cout << covs_rm[0] << std::endl;


    // C -> SVD -> UWT* -> U * W * T* -> C

    std::cout << "Start computing SVD of " << num_objects << " 3x3 matrices" << std::endl;

    std::vector<Eigen::Matrix3f> res_eigen(num_objects);

    rm::StopWatch sw;
    double el_eigen, el_rmagine, el_rmagine2;

    sw();
    #pragma omp parallel for
    for(size_t obj_id=0; obj_id<num_objects; obj_id++)
    {
        Eigen::JacobiSVD<Eigen::Matrix3f> svdeig(covs_eigen[obj_id], 
            Eigen::ComputeFullU | Eigen::ComputeFullV);
        auto s = svdeig.singularValues();
        Eigen::Matrix3f Seig = Eigen::Matrix3f::Zero();
        for(size_t i=0; i<s.rows(); i++)
        {
            Seig(i, i) = s(i);
        }
        auto uvt_eig = svdeig.matrixU() * Seig * svdeig.matrixV().transpose();
        res_eigen[obj_id] = uvt_eig;
    }
    el_eigen = sw();

    float err_eigen = 0.0;
    for(size_t obj_id = 0; obj_id < num_objects; obj_id++)
    {
        err_eigen += compute_error(covs_eigen[obj_id], res_eigen[obj_id]);
    }

    std::cout << "Eigen:" << std::endl;
    std::cout << "- run time: " << el_eigen << " s" << std::endl;
    std::cout << "- summed error: " << err_eigen << std::endl;


    rm::Memory<rm::Matrix3x3> res_rm(num_objects);

    sw();
    #pragma omp parallel for
    for(size_t obj_id=0; obj_id<num_objects; obj_id++)
    {
        rm::Matrix3x3 Urm = rm::Matrix3x3::Zeros();
        rm::Matrix3x3 Wrm = rm::Matrix3x3::Zeros();
        rm::Matrix3x3 Vrm = rm::Matrix3x3::Zeros();
        rm::svd(covs_rm[obj_id], Urm, Wrm, Vrm);
        auto uvt_rm = Urm * Wrm * Vrm.T();
        res_rm[obj_id] = uvt_rm;
    }
    el_rmagine = sw();
    
    
    float err_rmagine = 0.0;
    for(size_t obj_id = 0; obj_id < num_objects; obj_id++)
    {
        err_rmagine += compute_error(covs_rm[obj_id], res_rm[obj_id]);
    }
    
    std::cout << "Rmagine:" << std::endl;
    std::cout << "- run time: " << el_rmagine << " s" << std::endl;
    std::cout << "- summed error: " << err_rmagine << std::endl;
}

void parallelTest2()
{
    std::cout << "parallelTest2" << std::endl;

    size_t num_objects = 1000000;
    std::cout << "parallelTest. Computing SVD of " << num_objects << " 3x3 matrices" << std::endl;
    // correct num_objects objects in parallel
    

    std::vector<Eigen::Matrix3f> covs_eigen(num_objects);
    rm::Memory<rm::Matrix3x3, rm::RAM> covs_rm(num_objects);
    rm::Memory<rm::Matrix3x3, rm::RAM> Us(num_objects);
    rm::Memory<rm::Matrix3x3, rm::RAM> Ws(num_objects);
    rm::Memory<rm::Matrix3x3, rm::RAM> Vs(num_objects);

    for(size_t obj_id=0; obj_id<num_objects; obj_id++)
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

        covs_eigen[obj_id] = Aeig;
        covs_rm[obj_id] = Arm;
        Us[obj_id] = rm::Matrix3x3::Zeros();
        Ws[obj_id] = rm::Matrix3x3::Zeros();
        Vs[obj_id] = rm::Matrix3x3::Zeros();
    }

    std::cout << "First Mat: " << std::endl;
    std::cout << covs_rm[0] << std::endl;


    rm::StopWatch sw;
    double el_rmagine;


    sw();
    svd(covs_rm, Us, Ws, Vs);
    el_rmagine = sw();
    

    rm::Memory<rm::Matrix3x3> res_rm(num_objects);
    for(size_t obj_id=0; obj_id<num_objects; obj_id++)
    {
        auto uvt_rm = Us[obj_id] * Ws[obj_id] * Vs[obj_id].T();
        res_rm[obj_id] = uvt_rm;
    }
    
    float err_rmagine = 0.0;
    for(size_t obj_id = 0; obj_id < num_objects; obj_id++)
    {
        err_rmagine += compute_error(covs_rm[obj_id], res_rm[obj_id]);
    }
    
    std::cout << "Rmagine:" << std::endl;
    std::cout << "- run time: " << el_rmagine << " s" << std::endl;
    std::cout << "- summed error: " << err_rmagine << std::endl;
}

int main(int argc, char** argv)
{
    srand((unsigned int) time(0));

    // svdTestWithPrints();
    runtimeTest<20, 30>();
    accuracyTest<20, 30>();


    parallelTest();
    parallelTest2();


    return 0;
}

