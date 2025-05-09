#include "rmagine/simulation/SimulatorOptix.hpp"

#include "rmagine/simulation/optix/sim_program_data.h"
#include "rmagine/util/optix/OptixDebug.hpp"

#include <optix.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>


#include <rmagine/util/Debug.hpp>

#include <rmagine/util/cuda/CudaStream.hpp>

namespace rmagine
{

SimulatorOptix::SimulatorOptix()
:m_Tsb(1)
{
  std::cout << "SimulatorOptix!" << std::endl;
  Memory<Transform, RAM_CUDA> I(1);
  I->setIdentity();
  m_Tsb = I;
}

SimulatorOptix::SimulatorOptix(OptixMapPtr map)
:SimulatorOptix()
{
  setMap(map);
}

SimulatorOptix::~SimulatorOptix()
{

}

void SimulatorOptix::setMap(OptixMapPtr map)
{
  m_map = map;
  m_stream = m_map->stream();
}

void SimulatorOptix::setTsb(const Memory<Transform, RAM>& Tsb)
{
  m_Tsb = Tsb;
}

void SimulatorOptix::setTsb(const Transform& Tsb)
{
  Memory<Transform, RAM> tmp(1);
  tmp[0] = Tsb;
  setTsb(tmp);
}

} // namespace rmagine