#include <iostream>
#include <memory>
#include <rmagine/types/MemoryCuda.hpp>

namespace rmagine
{

int shared_function();

int shared_function()
{
  // on host: return 10
  return 10;
}

namespace cuda
{
__device__ 
int shared_function();

__device__ 
int shared_function()
{
  // on device: return 20
  return 20;
}

__global__ void test_function_kernel(int* res) 
{
  res[0] = shared_function();
}

} // namespace cuda

} // namespace rmagine


namespace rm = rmagine;

int main(int argc, char** argv)
{
  std::cout << "Rmagine Cuda Tests: Test Shared Functions" << std::endl;

  rm::Memory<int, rm::VRAM_CUDA> mem_gpu(1);

  rm::cuda::test_function_kernel<<<1,1>>>(mem_gpu.raw());
  rm::Memory<int> mem_cpu = mem_gpu; // download

  std::cout << "How it should be:" << std::endl;
  std::cout << "- Host: " << rm::shared_function() << std::endl;
  std::cout << "- Device: " << mem_cpu[0] << std::endl;
  if(mem_cpu[0] != 20)
  {
    // in a cuda kernel the implementation of the CUDA library should be called
    throw std::runtime_error("Wrong implementation is called!");
  }


  std::cout << "----" << std::endl;

  return 0;
}