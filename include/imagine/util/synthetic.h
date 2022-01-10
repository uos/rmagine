#ifndef IMAGINE_UTIL_SYNTHETIC_H
#define IMAGINE_UTIL_SYNTHETIC_H

#include <assimp/scene.h>
#include <vector>
#include <imagine/types/mesh_types.h>

namespace imagine 
{

void genSphere(
    std::vector<float>& vertices, 
    std::vector<unsigned int>& faces,
    unsigned int num_long,
    unsigned int num_lat
);

void genSphere(
    std::vector<Vector3>& vertices,
    std::vector<Face>& faces,
    unsigned int num_long,
    unsigned int num_lat
);

aiScene genSphere(unsigned int num_long = 50, unsigned int num_lat = 50);

} // namespace imagine

#endif // IMAGINE_UTIL_SYNTHETIC_H