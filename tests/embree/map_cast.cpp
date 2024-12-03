#include <rmagine/map/EmbreeMap.hpp>
#include <rmagine/map/MapMap.hpp>
#include <memory>
#include <rmagine/map/embree/embree_shapes.h>

namespace rm = rmagine;

rm::EmbreeMapPtr make_map()
{
    rm::EmbreeScenePtr scene = std::make_shared<rm::EmbreeScene>();

    rm::EmbreeGeometryPtr mesh = std::make_shared<rm::EmbreeCube>();
    mesh->commit();
    scene->add(mesh);
    scene->commit();

    return std::make_shared<rm::EmbreeMap>(scene);
}   

int main(int argc, char** argv)
{
    rm::MapMap maps;
    
    {
      rm::EmbreeMapPtr map1 = make_map();
      // store map into generic container
      maps["my_map"] = map1;
    }

    // load map from generic container
    rm::EmbreeMapPtr map2 = std::dynamic_pointer_cast<rm::EmbreeMap>(maps["my_map"]);

    return 0;
}