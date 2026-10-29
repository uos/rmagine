/**
 * @file HiprtScene.hpp
 * @brief HIPRT scene (geometry + BVH)
 *
 * Manages a collection of meshes and builds the HIPRT acceleration structure.
 */

#ifndef RMAGINE_MAP_HIPRT_SCENE_HPP
#define RMAGINE_MAP_HIPRT_SCENE_HPP

#include <memory>
#include <vector>
#include <map>

#include <rmagine/util/hiprt/HiprtContext.hpp>
#include <rmagine/map/hiprt/HiprtMesh.hpp>
#include <rmagine/simulation/hiprt/sim_program_data.h>

// Forward declarations
struct _hiprtGeometry;
typedef _hiprtGeometry* hiprtGeometry;

namespace rmagine
{

class HiprtScene;
using HiprtScenePtr = std::shared_ptr<HiprtScene>;

/**
 * @brief HIPRT scene containing meshes and their BVH
 *
 * This is the HIPRT equivalent of OptixScene (GAS-only mode).
 * Instance acceleration (IAS) is not yet implemented.
 */
class HiprtScene
{
public:
    explicit HiprtScene(HiprtContextPtr ctx);
    ~HiprtScene();

    // Non-copyable
    HiprtScene(const HiprtScene&) = delete;
    HiprtScene& operator=(const HiprtScene&) = delete;

    /**
     * @brief Add a mesh to the scene
     * @return The mesh ID within this scene
     */
    unsigned int add(HiprtMeshPtr mesh);

    /**
     * @brief Build or rebuild the BVH
     *
     * Must be called after adding/removing meshes.
     */
    void commit();

    /**
     * @brief Get the HIPRT geometry handle (for tracing)
     */
    hiprtGeometry geometry() const { return m_geom; }

    /**
     * @brief Get the geometry handle as unsigned long long
     *
     * This is what gets passed to the trace kernel.
     */
    unsigned long long handle() const;

    /**
     * @brief Get the scene data for trace kernel
     */
    const HiprtSceneData& sceneData() const { return m_scene_data; }

    /**
     * @brief Get number of meshes
     */
    size_t numMeshes() const { return m_meshes.size(); }

    /**
     * @brief Get the HIPRT context
     */
    HiprtContextPtr context() const { return m_ctx; }

private:
    HiprtContextPtr m_ctx;
    hiprtGeometry m_geom = nullptr;

    std::map<unsigned int, HiprtMeshPtr> m_meshes;
    unsigned int m_next_id = 0;

    // Scene data for trace kernel (device memory)
    HiprtSceneData m_scene_data;
    HiprtMeshData* m_mesh_data_device = nullptr;

    bool m_needs_rebuild = true;
};

} // namespace rmagine

#endif // RMAGINE_MAP_HIPRT_SCENE_HPP
