#include <iostream>

#include <rmagine/map/EmbreeMap.hpp>
#include <rmagine/map/embree/embree_shapes.h>
#include <rmagine/math/types.h>
#include <rmagine/util/prints.h>

namespace rm = rmagine;

rm::EmbreeMapPtr make_map_1()
{
    rm::EmbreeScenePtr cube_scene = std::make_shared<rm::EmbreeScene>();

    rm::EmbreeMeshPtr cube_mesh = std::make_shared<rm::EmbreeCube>();
    rm::Transform T = rm::Transform::Identity();
    T.t.z = 5.0;
    cube_mesh->setTransform(T);
    cube_mesh->apply();
    cube_mesh->commit();

    cube_scene->add(cube_mesh);
    cube_scene->commit();

    return std::make_shared<rm::EmbreeMap>(cube_scene);;
}

// DOES NOT WORK YET
rm::EmbreeMapPtr make_map_2()
{
    rm::EmbreeScenePtr cube_scene = std::make_shared<rm::EmbreeScene>();

    rm::EmbreeMeshPtr cube_mesh = std::make_shared<rm::EmbreeCube>();
    cube_mesh->commit();


    rm::EmbreeInstancePtr cube_inst = cube_mesh->instantiate();

    rm::Transform T = rm::Transform::Identity();
    T.t.z = 5.0;
    cube_inst->setTransform(T);
    cube_inst->apply();
    cube_inst->commit();

    cube_scene->add(cube_inst);
    cube_scene->commit();

    return std::make_shared<rm::EmbreeMap>(cube_scene);;
}

rm::EmbreeMapPtr make_map_3()
{
    rm::EmbreeScenePtr cube_scene = std::make_shared<rm::EmbreeScene>();

    {
        rm::EmbreeMeshPtr cube_mesh = std::make_shared<rm::EmbreeCube>();
        rm::Transform T = rm::Transform::Identity();
        T.t.z = 5.0;
        cube_mesh->setTransform(T);
        cube_mesh->apply();
        cube_mesh->commit();
        cube_scene->add(cube_mesh);
    }
    
    {
        rm::EmbreeMeshPtr cube_mesh = std::make_shared<rm::EmbreeCube>();
        rm::Transform T = rm::Transform::Identity();
        T.t.z = 5.0;
        T.t.x = 20.0;
        cube_mesh->setTransform(T);
        cube_mesh->apply();
        cube_mesh->commit();
        cube_scene->add(cube_mesh);
    }


    cube_scene->commit();

    return std::make_shared<rm::EmbreeMap>(cube_scene);;
}

int main(int argc, char ** argv)
{
    std::cout << "EMBREE CLOSEST POINT" << std::endl;


    auto map = make_map_3();
    rm::Vector qp;
    rm::Vector cp;

    qp = {0.0, 0.0, 0.0};
    cp = map->closestPoint(qp).p;
    std::cout << qp << " -> " << cp << std::endl;

    qp = {50.0, 50.0, 0.0};
    cp = map->closestPoint(qp).p;
    std::cout << qp << " -> " << cp << std::endl;

    qp = {50.0, -50.0, 20.0};
    cp = map->closestPoint(qp).p;
    std::cout << qp << " -> " << cp << std::endl;

    qp = {0.1, 0.1, 20.0};
    cp = map->closestPoint(qp).p;
    std::cout << qp << " -> " << cp << std::endl;

    qp = {-5.0, 0.0, 5.0};
    cp = map->closestPoint(qp).p;
    std::cout << qp << " -> " << cp << std::endl;
    




    return 0;
}