#include <iostream>
#include <sstream>

// General rmagine includes
#include <rmagine/types/sensor_models.h>
#include <rmagine/util/StopWatch.hpp>
#include <rmagine/util/synthetic.h>

#include <rmagine/map/EmbreeMap.hpp>

#include <rmagine/map/embree_shapes.h>
#include <rmagine/util/prints.h>

using namespace rmagine;

void scene_1()
{
    EmbreeScenePtr scene = std::make_shared<EmbreeScene>();
    
    {
        // add sphere. sphere should stay alive as long as the scene lives
        EmbreeMeshPtr sphere = std::make_shared<EmbreeSphere>(1.0);
        std::cout << sphere->scale() << std::endl;
        scene->add(sphere);
    }
    std::cout << "now sphere should be alive" << std::endl;

    scene.reset();
    std::cout << "Scene destroyed." << std::endl;
}

void scene_2()
{
    EmbreeScenePtr scene = std::make_shared<EmbreeScene>();

    unsigned int sphere_id = 0;
    {
        // add sphere. sphere should stay alive as long as the scene lives
        EmbreeSpherePtr sphere = std::make_shared<EmbreeSphere>(1.0);
        std::cout << sphere->scale() << std::endl;
        sphere_id = scene->add(sphere);
    }

    std::cout << "now sphere should be alive" << std::endl;

    // save mesh
    EmbreeMeshPtr sphere = scene->meshes()[sphere_id];

    scene.reset();
    std::cout << "Now Scene should be destroyed but not the sphere" << std::endl;

    if(!sphere->parent.lock())
    {
        std::cout << "- sphere noticed parent was destroyed." << std::endl;
    }
}

void scene_3()
{
    EmbreeScenePtr scene = std::make_shared<EmbreeScene>();
    unsigned int sphere_id = 0;
    {
        // add sphere. sphere should stay alive as long as the scene lives
        EmbreeSpherePtr sphere = std::make_shared<EmbreeSphere>(1.0);
        std::cout << sphere->scale() << std::endl;
        sphere_id = scene->add(sphere);
    }

    EmbreeMeshPtr sphere = scene->removeMesh(sphere_id);
    

    std::cout << "Removed mesh from scene. the next numbers should be 0" << std::endl;
    std::cout << scene->meshes().size() << std::endl;
    std::cout << sphere->parent.lock() << std::endl;
}

void scene_4()
{
    EmbreeScenePtr scene = std::make_shared<EmbreeScene>();
    
    
    // add sphere. sphere should stay alive as long as the scene lives
    EmbreeSpherePtr sphere = std::make_shared<EmbreeSphere>(1.0);
    std::cout << sphere->scale() << std::endl;
    scene->add(sphere);
    

    // the scenes mesh should get invalid when destroying mesh
    sphere.reset();

    std::cout << "meshes: " << scene->meshes().size() << std::endl;
}


void scene_5()
{
    EmbreeScenePtr scene = std::make_shared<EmbreeScene>();

    { // make sphere
        
    }
}

int main(int argc, char** argv)
{
    std::cout << "Rmagine Embree Scene Building" << std::endl;
    
    // std::cout << "SCENE EXAMPLE 1" << std::endl;
    // scene_1();

    // std::cout << "SCENE EXAMPLE 2" << std::endl;
    // scene_2();

    // std::cout << "SCENE EXAMPLE 3" << std::endl;
    // scene_3();

    std::cout << "SCENE EXAMPLE 4" << std::endl;
    scene_4();




    return 0;
}
