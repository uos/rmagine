#ifndef RMAGINE_SIMULATION_SIMULATOR_HPP
#define RMAGINE_SIMULATION_SIMULATOR_HPP

namespace rmagine
{

// TODO: make this possible

struct NoneType
{

};

template<typename ModelT, typename CompT>
class SimulatorType
{
public:
    using Class = NoneType;
    using Ptr = NoneType;
};

template<typename ModelT, typename CompT>
using Simulator = typename SimulatorType<ModelT, CompT>::Class; 

template<typename ModelT, typename CompT>
using SimulatorPtr = typename SimulatorType<ModelT, CompT>::Ptr; 


template<typename ModelT, typename CompT>
auto make_simulator()
{
    using SimT = SimulatorType<ModelT, CompT>;
    
    if constexpr(std::is_same<SimT, NoneType>())
    {
        return false;
    } else {
        typename SimT::Ptr sim(new typename SimT::Class);
        return sim;
    }
}

} // namespace rmagine

#endif // RMAGINE_SIMULATION_SIMULATOR_HPP