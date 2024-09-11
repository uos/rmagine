#include <rmagine/math/linalg.h>
#include <rmagine/util/prints.h>
#include <rmagine/math/math_batched.cuh>
#include <rmagine/util/StopWatch.hpp>

#include <Eigen/Dense>

#include <cuda_runtime.h>

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

void parallelTest()
{
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


    // upload
    rm::Memory<rm::Matrix3x3, rm::VRAM_CUDA> covs_rm_ = covs_rm;
    rm::Memory<rm::Matrix3x3, rm::VRAM_CUDA> Us_ = Us;
    rm::Memory<rm::Matrix3x3, rm::VRAM_CUDA> Ws_ = Ws;
    rm::Memory<rm::Matrix3x3, rm::VRAM_CUDA> Vs_ = Vs;

    sw();
    svd(covs_rm_, Us_, Ws_, Vs_);
    cudaDeviceSynchronize();
    el_rmagine = sw();
    
    // download
    Us = Us_;
    Ws = Ws_;
    Vs = Vs_;

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

    parallelTest();

    return 0;
}