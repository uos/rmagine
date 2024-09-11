#include <rmagine/math/SVD2.hpp>

#include <rmagine/math/math_batched.cuh>

#include <Eigen/Dense>

namespace rm = rmagine;


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


    rm::Memory<rm::Matrix3x3, rm::VRAM_CUDA> covs_rm_ = covs_rm;




}

int main(int argc, char** argv)
{
    srand((unsigned int) time(0));

    parallelTest();

    return 0;
}