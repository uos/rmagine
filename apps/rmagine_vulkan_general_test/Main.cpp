// Core rmagine includes
#include <rmagine/types/sensor_models.h>
#include <rmagine/util/StopWatch.hpp>
#include <rmagine/types/Memory.hpp>
#include <rmagine/map/AssimpIO.hpp>

// Vulkan rmagine includes
#include <rmagine/simulation/SphereSimulatorVulkan.hpp>
#include <rmagine/simulation/PinholeSimulatorVulkan.hpp>
#include <rmagine/simulation/O1DnSimulatorVulkan.hpp>
#include <rmagine/simulation/OnDnSimulatorVulkan.hpp>
#include <rmagine/map/VulkanMap.hpp>
#include <rmagine/types/MemoryVulkan.hpp>

using namespace rmagine;



int main()
{
    std::cout << "Main start." << std::endl;

    //mapdata:
    Memory<float, RAM> vertexMem_ram(18);
    Memory<uint32_t, RAM> indexMem_ram(6);
    //Vertecies
    vertexMem_ram[ 0] = -20.0f; vertexMem_ram[ 1] = -10.0f; vertexMem_ram[ 2] = -10.0f;
    vertexMem_ram[ 3] = -20.0f; vertexMem_ram[ 4] =  10.0f; vertexMem_ram[ 5] = -10.0f;
    vertexMem_ram[ 6] = -20.0f; vertexMem_ram[ 7] =   0.0f; vertexMem_ram[ 8] =  10.0f;
    vertexMem_ram[ 9] =  20.0f; vertexMem_ram[10] = -10.0f; vertexMem_ram[11] = -10.0f;
    vertexMem_ram[12] =  20.0f; vertexMem_ram[13] =  10.0f; vertexMem_ram[14] = -10.0f;
    vertexMem_ram[15] =  20.0f; vertexMem_ram[16] =   0.0f; vertexMem_ram[17] =  10.0f;
    //Indicies
    indexMem_ram[ 0] = 0; indexMem_ram[ 1] = 1; indexMem_ram[ 2] = 2;
    indexMem_ram[ 3] = 3; indexMem_ram[ 4] = 4; indexMem_ram[ 5] = 5;
    //logging
    uint32_t numVerticies = vertexMem_ram.size()/3;
    uint32_t numTriangles = indexMem_ram.size()/3;
    std::cout << "Using Mesh with " << numVerticies << " Verticies & " << numTriangles << " Triangles." << std::endl;


    //create map
    std::cout << "Creating map main." << std::endl;
    VulkanMapPtr map = import_vulkan_map(vertexMem_ram, indexMem_ram);


    //allocate sphere sensorbuffer
    std::cout << "Creating sphere sensor." << std::endl;
    SphericalModel sphereSensor = SphericalModel();
    sphereSensor.phi = {-15.0 * M_PI / 180.0, 2.0 * M_PI / 180.0, 16};
    sphereSensor.theta = {-M_PI, (2*M_PI)/128, 128};
    Memory<SphericalModel, RAM> sphereSensor_ram(1);
    sphereSensor_ram[0] = sphereSensor;

    //allocate pinhole sensorbuffer
    std::cout << "Creating pinhole sensor." << std::endl;
    PinholeModel pinholeSensor = PinholeModel();
    pinholeSensor.width = 64;
    pinholeSensor.height = 32;
    pinholeSensor.f[0] = 20.0;
    pinholeSensor.f[1] = 20.0;
    pinholeSensor.c[0] = 31.5;
    pinholeSensor.c[1] = 17.5;
    Memory<PinholeModel, RAM> pinholeSensor_ram(1);
    pinholeSensor_ram[0] = pinholeSensor;

    //allocate o1dn sensorbuffer
    std::cout << "Creating o1dn sensor." << std::endl;
    O1DnModel o1dnSensor = O1DnModel();
    o1dnSensor.width = 6;
    o1dnSensor.height = 1;
    o1dnSensor.orig = {0,0,0};
    Memory<Vector3, RAM> dirs(6);
    dirs[0] = {-1,0,0};
    dirs[1] = { 1,0,0};
    dirs[2] = {0,-1,0};
    dirs[3] = {0, 1,0};
    dirs[4] = {0,0,-1};
    dirs[5] = {0,0, 1};
    o1dnSensor.dirs = dirs;
    Memory<O1DnModel, RAM> o1dnSensor_ram(1);
    o1dnSensor_ram[0] = o1dnSensor;

    //allocate ondn sensorbuffer
    std::cout << "Creating ondn sensor." << std::endl;
    OnDnModel ondnSensor = OnDnModel();
    ondnSensor.width = 6;
    ondnSensor.height = 1;
    Memory<Vector3, RAM> origs(6);
    origs[0] = {60,0,0};
    origs[1] = {60,0,0};
    origs[2] = {0,0,0};
    origs[3] = {0,0,0};
    origs[4] = {0,0,0};
    origs[5] = {0,0,0};
    ondnSensor.origs = origs;
    Memory<Vector3, RAM> dirs2(6);
    dirs2[0] = {-1,0,0};
    dirs2[1] = { 1,0,0};
    dirs2[2] = {0,-1,0};
    dirs2[3] = {0, 1,0};
    dirs2[4] = {0,0,-1};
    dirs2[5] = {0,0, 1};
    ondnSensor.dirs = dirs2;
    Memory<OnDnModel, RAM> ondnSensor_ram(1);
    ondnSensor_ram[0] = ondnSensor;


    //allocate transformbuffer
    std::cout << "Creating transform." << std::endl;
    Transform tsb = Transform();
    Memory<Transform, RAM> tsb_ram(1);
    tsb_ram[0] = tsb;


    //create sphere simulator
    std::cout << "Creating sphere sim." << std::endl;
    SphereSimulatorVulkanPtr sim_gpu_sphere = std::make_shared<SphereSimulatorVulkan>(map);
    sim_gpu_sphere->setModel(sphereSensor_ram);
    sim_gpu_sphere->setTsb(tsb_ram);

    //create pinhole simulator
    std::cout << "Creating pinhole sim." << std::endl;
    PinholeSimulatorVulkanPtr sim_gpu_pinhole = std::make_shared<PinholeSimulatorVulkan>(map);
    sim_gpu_pinhole->setModel(pinholeSensor_ram);
    sim_gpu_pinhole->setTsb(tsb_ram);

    //create o1dn simulator
    std::cout << "Creating o1dn sim." << std::endl;
    O1DnSimulatorVulkanPtr sim_gpu_o1dn = std::make_shared<O1DnSimulatorVulkan>(map);
    sim_gpu_o1dn->setModel(o1dnSensor_ram);
    sim_gpu_o1dn->setTsb(tsb_ram);

    //create ondn simulator
    std::cout << "Creating ondn sim." << std::endl;
    OnDnSimulatorVulkanPtr sim_gpu_ondn = std::make_shared<OnDnSimulatorVulkan>(map);
    sim_gpu_ondn->setModel(ondnSensor_ram);
    sim_gpu_ondn->setTsb(tsb_ram);


    //allocate transformsbuffer 
    std::cout << "Creating transforms." << std::endl;
    Memory<Transform, RAM> tbm(2);
    Memory<Transform, VULKAN_DEVICE_LOCAL> tbm_device(2);
    tbm_device = tbm;

    //bundle
    using ResultT = Bundle<
        Hits<VULKAN_DEVICE_LOCAL>,
        Ranges<VULKAN_DEVICE_LOCAL>,
        Points<VULKAN_DEVICE_LOCAL>,
        Normals<VULKAN_DEVICE_LOCAL>,
        FaceIds<VULKAN_DEVICE_LOCAL>,   //primitive ids
        GeomIds<VULKAN_DEVICE_LOCAL>,   //geometry ids
        ObjectIds<VULKAN_DEVICE_LOCAL>  //instance ids
    >;

    //allocate sphere resultsbuffer
    std::cout << "Creating sphere results." << std::endl;
    Memory<uint8_t, RAM> sphereHits_ram(tbm.size()*sphereSensor.phi.size*sphereSensor.theta.size);
    ResultT res_sphere;
    resize_memory_bundle<VULKAN_DEVICE_LOCAL>(res_sphere, sphereSensor.phi.size, sphereSensor.theta.size, tbm.size());

    //allocate pinhole resultsbuffer
    std::cout << "Creating pinhole results." << std::endl;
    Memory<uint8_t, RAM> pinholeHits_ram(tbm.size()*pinholeSensor.width*pinholeSensor.height);
    ResultT res_pinhole;
    resize_memory_bundle<VULKAN_DEVICE_LOCAL>(res_pinhole, pinholeSensor.width, pinholeSensor.height, tbm.size());

    //allocate o1dn resultsbuffer
    std::cout << "Creating o1dn results." << std::endl;
    Memory<uint8_t, RAM> o1dnHits_ram(tbm.size()*o1dnSensor.width*o1dnSensor.height);
    Memory<float, RAM> o1dnRanges_ram(tbm.size()*o1dnSensor.width*o1dnSensor.height);
    Memory<Vector3, RAM> o1dnPoints_ram(tbm.size()*o1dnSensor.width*o1dnSensor.height);
    Memory<Vector3, RAM> o1dnNormals_ram(tbm.size()*o1dnSensor.width*o1dnSensor.height);
    Memory<unsigned int, RAM> o1dnPrimitiveIds_ram(tbm.size()*o1dnSensor.width*o1dnSensor.height);
    Memory<unsigned int, RAM> o1dnGeometryIds_ram(tbm.size()*o1dnSensor.width*o1dnSensor.height);
    Memory<unsigned int, RAM> o1dnInstanceIds_ram(tbm.size()*o1dnSensor.width*o1dnSensor.height);
    ResultT res_o1dn;
    resize_memory_bundle<VULKAN_DEVICE_LOCAL>(res_o1dn, o1dnSensor.width, o1dnSensor.height, tbm.size());

    //allocate ondn resultsbuffer
    std::cout << "Creating ondn results." << std::endl;
    Memory<uint8_t, RAM> ondnHits_ram(tbm.size()*ondnSensor.width*ondnSensor.height);
    Memory<float, RAM> ondnRanges_ram(tbm.size()*ondnSensor.width*ondnSensor.height);
    Memory<Vector3, RAM> ondnPoints_ram(tbm.size()*ondnSensor.width*ondnSensor.height);
    Memory<Vector3, RAM> ondnNormals_ram(tbm.size()*ondnSensor.width*ondnSensor.height);
    Memory<unsigned int, RAM> ondnPrimitiveIds_ram(tbm.size()*ondnSensor.width*ondnSensor.height);
    Memory<unsigned int, RAM> ondnGeometryIds_ram(tbm.size()*ondnSensor.width*ondnSensor.height);
    Memory<unsigned int, RAM> ondnInstanceIds_ram(tbm.size()*ondnSensor.width*ondnSensor.height);
    ResultT res_ondn;
    resize_memory_bundle<VULKAN_DEVICE_LOCAL>(res_ondn, ondnSensor.width, ondnSensor.height, tbm.size());


    //simulate sphere
    std::cout << "Simulating sphere..." << std::endl;
    sim_gpu_sphere->simulate(tbm_device, res_sphere);

    //simulate pinhole
    std::cout << "Simulating pinhole..." << std::endl;
    sim_gpu_pinhole->simulate(tbm_device, res_pinhole);

    //simulate o1dn
    std::cout << "Simulating o1dn..." << std::endl;
    sim_gpu_o1dn->simulate(tbm_device, res_o1dn);

    //simulate ondn
    std::cout << "Simulating ondn..." << std::endl;
    sim_gpu_ondn->simulate(tbm_device, res_ondn);


    //get sphere results
    sphereHits_ram = res_sphere.hits;
    std::cout << "\nSphere results:" << std::endl;
    for(size_t j = 0; j < tbm.size(); j++)
    {
        for(int32_t k = sphereSensor.phi.size-1; k >= 0; k--)
        {
            for(int32_t l = 0; l < sphereSensor.theta.size; l++)
            {
                std::cout << (int)sphereHits_ram[j*sphereSensor.phi.size*sphereSensor.theta.size + k*sphereSensor.theta.size + l];
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    //get pinhole results
    pinholeHits_ram = res_pinhole.hits;
    std::cout << "\nPinhole results:" << std::endl;
    for(size_t j = 0; j < pinholeHits_ram.size(); j++)
    {
        std::cout << (int)pinholeHits_ram[j];

        if((j+1)%(pinholeSensor.width) == 0)
            std::cout << std::endl;
        if((j+1)%(pinholeSensor.width*pinholeSensor.height) == 0)
            std::cout << std::endl;
    }

    //get o1dn results
    o1dnHits_ram = res_o1dn.hits;
    o1dnRanges_ram = res_o1dn.ranges;
    o1dnPoints_ram = res_o1dn.points;
    o1dnNormals_ram = res_o1dn.normals;
    o1dnPrimitiveIds_ram = res_o1dn.face_ids;
    o1dnGeometryIds_ram = res_o1dn.geom_ids;
    o1dnInstanceIds_ram = res_o1dn.object_ids;
    std::cout << "\no1dn results:" << std::endl;
    for(size_t j = 0; j < o1dnHits_ram.size(); j++)
    {
        std::cout << (int)o1dnHits_ram[j] << "; ";
        std::cout << o1dnRanges_ram[j] << "; ";
        std::cout << o1dnPoints_ram[j].x << ", " << o1dnPoints_ram[j].y << ", " << o1dnPoints_ram[j].z << "; ";
        std::cout << o1dnNormals_ram[j].x << ", " << o1dnNormals_ram[j].y << ", " << o1dnNormals_ram[j].z << "; ";
        std::cout << o1dnPrimitiveIds_ram[j] << "; ";
        std::cout << o1dnGeometryIds_ram[j] << "; ";
        std::cout << o1dnInstanceIds_ram[j] << "; ";
        std::cout << std::endl;

        if((j+1)%(o1dnSensor.width) == 0)
            std::cout << std::endl;
        if((j+1)%(o1dnSensor.width*o1dnSensor.height) == 0)
            std::cout << std::endl;
    }

    //get ondn results
    ondnHits_ram = res_ondn.hits;
    ondnRanges_ram = res_ondn.ranges;
    ondnPoints_ram = res_ondn.points;
    ondnNormals_ram = res_ondn.normals;
    ondnPrimitiveIds_ram = res_ondn.face_ids;
    ondnGeometryIds_ram = res_ondn.geom_ids;
    ondnInstanceIds_ram = res_ondn.object_ids;
    std::cout << "\nondn results:" << std::endl;
    for(size_t j = 0; j < ondnHits_ram.size(); j++)
    {
        std::cout << (int)ondnHits_ram[j] << "; ";
        std::cout << ondnRanges_ram[j] << "; ";
        std::cout << ondnPoints_ram[j].x << ", " << ondnPoints_ram[j].y << ", " << ondnPoints_ram[j].z << "; ";
        std::cout << ondnNormals_ram[j].x << ", " << ondnNormals_ram[j].y << ", " << ondnNormals_ram[j].z << "; ";
        std::cout << ondnPrimitiveIds_ram[j] << "; ";
        std::cout << ondnGeometryIds_ram[j] << "; ";
        std::cout << ondnInstanceIds_ram[j] << "; ";
        std::cout << std::endl;

        if((j+1)%(ondnSensor.width) == 0)
            std::cout << std::endl;
        if((j+1)%(ondnSensor.width*ondnSensor.height) == 0)
            std::cout << std::endl;
    }

    std::cout << "Main end." << std::endl;

    return EXIT_SUCCESS;
}
