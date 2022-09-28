#include <iostream>

#include <rmagine/simulation/SphereSimulatorOptix.hpp>
#include <rmagine/simulation/SphereSimulatorEmbree.hpp>
#include <rmagine/util/StopWatch.hpp>

using namespace rmagine;

SphereSimulatorOptixPtr   sim_gpu;
SphereSimulatorEmbreePtr  sim_cpu;

template<typename BundleT>
void resizeResults(BundleT& res, unsigned int W, unsigned int H, unsigned int N)
{
    resizeMemoryBundle<RAM>(res, W, H, N);
    resizeMemoryBundle<VRAM_CUDA>(res, W, H, N);
}

template<typename BundleT1, typename BundleT2>
bool compareResults(const BundleT1& res1, const BundleT2& res2)
{
    bool res = true;
    const float EPS = 0.00001;

    unsigned int checkN = 1000;

    if constexpr(BundleT1::template has<Hits<RAM> >() 
                && BundleT2::template has<Hits<VRAM_CUDA> >() )
    {
        Memory<uint8_t, RAM> hits;
        hits = res2.hits;

        if(hits.size() != res1.hits.size())
        {
            std::cout << "Hits size differ!" << std::endl;
            res &= false;
        }
        for(unsigned int i=0; i<checkN; i++)
        {
            if(hits[i] != res1.hits[i])
            {
                std::cout << "Hit entry " << i << " differ" << std::endl;
                std::cout << "- GPU: " << (int)hits[i] << std::endl;
                std::cout << "- CPU: " << (int)res1.hits[i] << std::endl;
                res &= false;
            }
        }
    }

    if constexpr(BundleT1::template has<Ranges<RAM> >() 
                && BundleT2::template has<Ranges<VRAM_CUDA> >() )
    {
        Memory<float, RAM> ranges;
        ranges = res2.ranges;

        if(ranges.size() != res1.ranges.size())
        {
            std::cout << "Ranges size differ!" << std::endl;
            res &= false;
        }
        for(unsigned int i=0; i<checkN; i++)
        {
            if(fabs(ranges[i] - res1.ranges[i]) > EPS )
            {
                std::cout << "Range entry " << i << " differ " << std::endl;
                std::cout << "- GPU: " << ranges[i] << std::endl;
                std::cout << "- CPU: " << res1.ranges[i] << std::endl;
                res &= false;
            }
        }
    }

    if constexpr(BundleT1::template has<Points<RAM> >() 
                && BundleT2::template has<Points<VRAM_CUDA> >() )
    {
        Memory<Point, RAM> points;
        points = res2.points;

        if(points.size() != res1.points.size())
        {
            std::cout << "Points size differ!" << std::endl;
            res &= false;
        }
        for(unsigned int i=0; i<checkN; i++)
        {
            if((points[i] - res1.points[i]).l2norm() > EPS )
            {
                std::cout << "Point entry " << i << " differ " << std::endl;
                std::cout << "- GPU: " << points[i].x << " " << points[i].y << " " << points[i].z << std::endl;
                std::cout << "- CPU: " << res1.points[i].x << " " << res1.points[i].y << " " << res1.points[i].z << std::endl;
                res &= false;
            }
        }
    }

    if constexpr(BundleT1::template has<Normals<RAM> >() 
                && BundleT2::template has<Normals<VRAM_CUDA> >() )
    {
        Memory<Vector, RAM> normals;
        normals = res2.normals;

        if(normals.size() != res1.normals.size())
        {
            std::cout << "Normals size differ!" << std::endl;
            return false;
        }
        for(unsigned int i=0; i<checkN; i++)
        {
            if((normals[i] - res1.normals[i]).l2norm() > EPS )
            {
                std::cout << "Normal entry " << i << " differ " << std::endl;
                std::cout << "- GPU: " << normals[i].x << " " << normals[i].y << " " << normals[i].z << std::endl;
                std::cout << "- CPU: " << res1.normals[i].x << " " << res1.normals[i].y << " " << res1.normals[i].z << std::endl;
                
                res &= false;
                // return false;
            }
        }
    }

    if constexpr(BundleT1::template has<FaceIds<RAM> >() 
                && BundleT2::template has<FaceIds<VRAM_CUDA> >() )
    {
        Memory<unsigned int, RAM> face_ids;
        face_ids = res2.face_ids;

        if(face_ids.size() != res1.face_ids.size())
        {
            std::cout << "FaceIds size differ!" << std::endl;
            return false;
        }
        for(unsigned int i=0; i<checkN; i++)
        {
            if( face_ids[i] != res1.face_ids[i] )
            {
                std::cout << "FaceId entry " << i << " differ " << std::endl;
                std::cout << "- GPU: " << face_ids[i] << std::endl;
                std::cout << "- CPU: " << res1.face_ids[i] << std::endl;
                res &= false;
            }
        }
    }

    if constexpr(BundleT1::template has<ObjectIds<RAM> >() 
                && BundleT2::template has<ObjectIds<VRAM_CUDA> >() )
    {
        Memory<unsigned int, RAM> object_ids;
        object_ids = res2.object_ids;

        if(object_ids.size() != res1.object_ids.size())
        {
            std::cout << "ObjectIds size differ!" << std::endl;
            return false;
        }
        for(unsigned int i=0; i<checkN; i++)
        {
            if( object_ids[i] != res1.object_ids[i] )
            {
                std::cout << "ObjectId entry " << i << " differ " << std::endl;
                std::cout << "- GPU: " << object_ids[i] << std::endl;
                std::cout << "- CPU: " << res1.object_ids[i] << std::endl;
                res &= false;
            }
        }
    }

    return res;
}

Memory<LiDARModel, RAM> velodyne_model()
{
    Memory<LiDARModel, RAM> model(1);
    model->theta.min = -M_PI;
    model->theta.inc = 0.4 * M_PI / 180.0;
    model->theta.size = 900;

    model->phi.min = -15.0 * M_PI / 180.0;
    model->phi.inc = 2.0 * M_PI / 180.0;
    model->phi.size = 16;
    
    model->range.min = 0.5;
    model->range.max = 130.0;
    return model;
}

int main(int argc, char** argv)
{
    std::cout << "Rmagine Test: Generic Simulation" << std::endl;

    if(argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " [meshfile]" << std::endl;
        return 0;
    }

    StopWatch sw;
    double el;

    std::cout << "Loading " << argv[1] << std::endl;

    sw();
    EmbreeMapPtr map_cpu = importEmbreeMap(argv[1]);
    sim_cpu.reset(new SphereSimulatorEmbree(map_cpu));
    el = sw();
    std::cout << "- CPU: loaded in " << el << "s" << std::endl;


    sw();
    OptixMapPtr map_gpu = importOptixMap(argv[1]);
    sim_gpu.reset(new SphereSimulatorOptix(map_gpu));
    el = sw();
    std::cout << "- GPU: loaded in " << el << "s" << std::endl;

    // Define and set Scanner Model
    auto model = velodyne_model();

    sim_gpu->setModel(model);
    sim_cpu->setModel(model);

    // Define and set Transformation between sensor and base 
    // (sensor offset to poses)
    Memory<Transform, RAM> Tsb;
    Tsb->R.x = 0.0;
    Tsb->R.y = 0.0;
    Tsb->R.z = 0.0;
    Tsb->R.w = 1.0;
    Tsb->t.x = 0.01;
    Tsb->t.y = 0.0;
    Tsb->t.z = 0.0; // offset on z axis

    sim_gpu->setTsb(Tsb);
    sim_cpu->setTsb(Tsb);


    // Define and set poses to transform from
    // Transformations between base and map
    size_t Nposes = 10000;
    Memory<Transform, RAM> Tbm_cpu(Nposes);
    for(size_t i=0; i<Nposes; i++)
    {
        Tbm_cpu[i].R.x = 0.0;
        Tbm_cpu[i].R.y = 0.0;
        Tbm_cpu[i].R.z = 0.0;
        Tbm_cpu[i].R.w = 1.0;
        Tbm_cpu[i].t.x = 0.1;
        Tbm_cpu[i].t.y = 0.0;
        Tbm_cpu[i].t.z = 0.0;
    }

    Memory<Transform, VRAM_CUDA> Tbm_gpu;
    Tbm_gpu = Tbm_cpu;


    // declere ResultT
    using ResultT1_CPU = Bundle<
            Hits<RAM>, 
            Ranges<RAM>,
            Points<RAM>,
            Normals<RAM>,
            FaceIds<RAM>,
            ObjectIds<RAM>
        >;

    using ResultT1_GPU = Bundle<
            Hits<VRAM_CUDA>, 
            Ranges<VRAM_CUDA>,
            Points<VRAM_CUDA>,
            Normals<VRAM_CUDA>,
            FaceIds<VRAM_CUDA>,
            ObjectIds<VRAM_CUDA> 
        >;

    ResultT1_CPU res_cpu;
    ResultT1_GPU res_gpu;

    unsigned int Nrays = model->theta.size * model->phi.size * Tbm_cpu.size();
    resizeResults(res_cpu, model->theta.size, model->phi.size, Tbm_cpu.size() );
    resizeResults(res_gpu, model->theta.size, model->phi.size, Tbm_cpu.size() );

    std::cout << "Simulate " << Tbm_cpu.size() << " poses / " << Nrays << " rays" << std::endl;

    sw();
    sim_cpu->simulate(Tbm_cpu, res_cpu);
    el = sw();
    std::cout << "- CPU: " << el << "s" << std::endl;

    sim_gpu->preBuildProgram<ResultT1_GPU>();

    sw();
    sim_gpu->simulate(Tbm_gpu, res_gpu);
    el = sw();
    std::cout << "- GPU: " << el << "s" << std::endl;

    std::cout << "Compare results" << std::endl;

    if(compareResults(res_cpu, res_gpu))
    {
        std::cout << "GPU == CPU" << std::endl;
    } else {
        std::cout << "GPU != CPU" << std::endl;
    }

    sim_gpu.reset();

    return 0;
}