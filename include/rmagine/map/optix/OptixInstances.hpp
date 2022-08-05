#ifndef RMAGINE_MAP_OPTIX_INSTANCES_HPP
#define RMAGINE_MAP_OPTIX_INSTANCES_HPP


#include "optix_definitions.h"
#include "OptixGeometry.hpp"

#include <rmagine/util/IDGen.hpp>

#include <map>

namespace rmagine
{

class OptixInstances 
: public OptixGeometry
{
public:
    using Base = OptixGeometry;

    OptixInstances(OptixContextPtr context = optix_default_context());

    virtual ~OptixInstances();

    virtual void apply();
    virtual void commit();


    OptixInstPtr remove(unsigned int id);
    bool remove(OptixInstPtr inst);

    unsigned int add(OptixInstPtr inst);
    unsigned int get(OptixInstPtr inst) const;
    OptixInstPtr get(unsigned int id) const;

    std::map<unsigned int, OptixInstPtr> instances() const;
    std::unordered_map<OptixInstPtr, unsigned int> ids() const;

private:

    // void build_acc();
    void build_acc_old();

    IDGen gen;

    std::map<unsigned int, OptixInstPtr> m_instances;
    std::unordered_map<OptixInstPtr, unsigned int> m_ids;

    bool m_requires_build = false;

    // filled after commit
    // std::unordered_map<unsigned int, CUdeviceptr> m_instances_gpu;
};

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_INSTANCES_HPP