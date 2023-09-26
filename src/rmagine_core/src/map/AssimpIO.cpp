#include "rmagine/map/AssimpIO.hpp"

namespace rmagine
{

AssimpIO::AssimpIO()
:Assimp::Importer()
,Assimp::Exporter()
{
    SetPropertyBool(AI_CONFIG_IMPORT_COLLADA_IGNORE_UP_DIRECTION, true);
    // TODO: add further settings if necessary
}


} // namespace rmagine