#ifndef RMAGINE_MAP_ASSIMP_IO_HPP
#define RMAGINE_MAP_ASSIMP_IO_HPP

#include <assimp/Importer.hpp>
#include <assimp/Exporter.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

namespace rmagine
{

/**
 * @brief Mixture of assimp Importer and assimp Exporter 
 *  enriched by own default settings
 * 
 */
class AssimpIO
: public Assimp::Importer
, public Assimp::Exporter
{
public:
    using Importer = Assimp::Importer;
    using Exporter = Assimp::Exporter;
    AssimpIO();
};

} // namespace rmagine


#endif // RMAGINE_MAP_ASSIMP_IO_HPP