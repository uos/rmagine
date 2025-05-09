

namespace rmagine
{

template<typename BundleT>
void SimulatorOptix::simulate(
  const Transform& Tbm, BundleT& ret) const
{
  // upload pose
  Transform Tbm_tmp = Tbm;
  const MemoryView<Transform, RAM> Tbm_mem(&Tbm_tmp, 1);
  Memory<Transform, VRAM_CUDA> Tbm_gpu = Tbm_mem;
  simulate(Tbm_gpu, ret);
}

template<typename BundleT>
BundleT SimulatorOptix::simulate(const Transform& Tbm) const
{
  BundleT res;
  resize_memory_bundle<VRAM_CUDA>(res, m_width, m_height, 1);
  simulate(Tbm, res);
  return res;
}

template<typename BundleT>
BundleT SimulatorOptix::simulate(
    const Memory<Transform, VRAM_CUDA>& Tbm) const
{
  BundleT res;
  resize_memory_bundle<VRAM_CUDA>(res, m_width, m_height, Tbm.size());
  simulate(Tbm, res);
  return res;
}

} // namespace rmagine