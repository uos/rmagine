

namespace rmagine
{

template<typename BundleT>
void SimulatorOptix::simulate(
  const Transform& Tbm,
  BundleT& ret,
  Culling culling) const
{
  // upload pose
  Transform Tbm_tmp = Tbm;
  const MemoryView<Transform, RAM> Tbm_mem(&Tbm_tmp, 1);
  Memory<Transform, VRAM_CUDA> Tbm_gpu = Tbm_mem;
  simulate(Tbm_gpu, ret, culling);
}

template<typename BundleT>
BundleT SimulatorOptix::simulate(const Transform& Tbm, Culling culling) const
{
  BundleT res;
  resize_memory_bundle<VRAM_CUDA>(res, m_width, m_height, 1);
  simulate(Tbm, res, culling);
  return res;
}

template<typename BundleT>
BundleT SimulatorOptix::simulate(
    const Memory<Transform, VRAM_CUDA>& Tbm,
    Culling culling) const
{
  BundleT res;
  resize_memory_bundle<VRAM_CUDA>(res, m_width, m_height, Tbm.size());
  simulate(Tbm, res, culling);
  return res;
}

} // namespace rmagine
