#ifndef RMAGINE_MAP_OPTIX_SCENE_HPP
#define RMAGINE_MAP_OPTIX_SCENE_HPP

#include <rmagine/util/optix/OptixContext.hpp>
#include <rmagine/util/IDGen.hpp>

#include "OptixAccelerationStructure.hpp"
#include "OptixGeometry.hpp"
#include "OptixInstance.hpp"

namespace rmagine
{

class OptixScene
{
public:
    OptixScene(OptixContextPtr context = optix_default_context());
    unsigned int add(OptixInstPtr inst);

    void commit();
private:
    IDGen gen;
    OptixContextPtr m_ctx;

    OptixAccelerationStructurePtr m_as;

    std::unordered_map<unsigned int, OptixInstPtr> m_instances;
    std::unordered_map<OptixInstPtr, unsigned int> m_ids;
};

using OptixScenePtr = std::shared_ptr<OptixScene>;

} // namespace rmagine

#endif // RMAGINE_MAP_OPTIX_SCENE_HPP