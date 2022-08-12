#include "rmagine/noise/NoiseCuda.hpp"
#include <rmagine/util/cuda/random.cuh>

namespace rmagine
{


NoiseCuda::NoiseCuda(Options options)
:m_options(options)
{
    if(m_options.estimated_memory_size > 0)
    {
        m_states.resize(m_options.estimated_memory_size);
        random_init(m_states, m_options.seed);
    }
}

void NoiseCuda::updateStates(MemoryView<float, VRAM_CUDA>& ranges)
{
    if(!m_options.fixed_memory)
    {
        if(m_states.size() < ranges.size())
        {
            Memory<curandState, VRAM_CUDA> states_old = m_states;
            m_states.resize(ranges.size());

            // restore old states
            m_states(0, states_old.size()) = states_old;
            
            // create new states
            auto slice_new = m_states(states_old.size(), m_states.size());
            random_init(slice_new, m_options.seed);
        } else if(m_states.size() > ranges.size()) {
            if(!m_options.never_shrink_memory)
            {
                Memory<curandState, VRAM_CUDA> states_old = m_states;
                m_states.resize(ranges.size());
                // restore old states
                m_states = states_old(0, m_states.size());
            }
        }
    }
}
    

} // namespace rmagine
