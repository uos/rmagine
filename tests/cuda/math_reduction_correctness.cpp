// Asserting correctness + determinism gate for the wave-size-hardened GPU
// reductions (rm::sum, rm::mean, rm::cov). The other cuda_math* tests only
// print their reduction outputs; this one compares the GPU result against a
// double-precision CPU reference to ~float eps and asserts that two runs are
// bit-identical, so the wave64 hardening of cov_kernel<1024>/sum_kernel<1024>
// is actually gated by a test on real GPU.

#include <rmagine/math/memory_math.cuh>
#include <rmagine/math/memory_math.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace rm = rmagine;

static void require(bool cond, const std::string& msg)
{
    if(!cond)
    {
        throw std::runtime_error("FAIL: " + msg);
    }
}

static void check_rel(double got, double ref, double eps, const std::string& what)
{
    const double denom = std::max(1.0, std::abs(ref));
    const double rel = std::abs(got - ref) / denom;
    if(rel > eps || got != got)
    {
        throw std::runtime_error("FAIL: " + what + " got=" + std::to_string(got)
            + " ref=" + std::to_string(ref) + " rel=" + std::to_string(rel));
    }
}

int main()
{
    const size_t N = 4099; // not a power of two; exercises the masked tail rows

    rm::Memory<rm::Vector> a(N);
    rm::Memory<rm::Vector> b(N);

    double ref_sum_x = 0.0, ref_sum_y = 0.0, ref_sum_z = 0.0;
    double ref_cov[3][3] = {{0}};

    for(size_t i = 0; i < N; i++)
    {
        const double p = static_cast<double>(i) / static_cast<double>(N);
        rm::Vector va = { static_cast<float>(std::sin(p * 7.0)),
                          static_cast<float>(p * 3.0 - 1.0),
                          static_cast<float>(std::cos(p * 5.0) * 2.0) };
        rm::Vector vb = { static_cast<float>(p * 2.0),
                          static_cast<float>(std::cos(p * 11.0)),
                          static_cast<float>(std::sin(p * 13.0) - 0.5) };
        a[i] = va;
        b[i] = vb;

        ref_sum_x += va.x; ref_sum_y += va.y; ref_sum_z += va.z;
        const double ax = va.x, ay = va.y, az = va.z;
        const double bx = vb.x, by = vb.y, bz = vb.z;
        ref_cov[0][0] += ax * bx; ref_cov[1][0] += ax * by; ref_cov[2][0] += ax * bz;
        ref_cov[0][1] += ay * bx; ref_cov[1][1] += ay * by; ref_cov[2][1] += ay * bz;
        ref_cov[0][2] += az * bx; ref_cov[1][2] += az * by; ref_cov[2][2] += az * bz;
    }
    for(int r = 0; r < 3; r++)
        for(int c = 0; c < 3; c++)
            ref_cov[r][c] /= static_cast<double>(N);

    rm::Memory<rm::Vector, rm::VRAM_CUDA> a_gpu = a;
    rm::Memory<rm::Vector, rm::VRAM_CUDA> b_gpu = b;

    // --- rm::sum (sum_kernel<1024>) vs CPU reference ---
    rm::Memory<rm::Vector> sum_h = rm::sum(a_gpu);
    std::cout << "sum = " << sum_h[0].x << ", " << sum_h[0].y << ", " << sum_h[0].z << std::endl;
    check_rel(sum_h[0].x, ref_sum_x, 1e-4, "sum.x");
    check_rel(sum_h[0].y, ref_sum_y, 1e-4, "sum.y");
    check_rel(sum_h[0].z, ref_sum_z, 1e-4, "sum.z");

    // --- rm::mean (sum_kernel<1024> + divNx1) vs CPU reference ---
    rm::Memory<rm::Vector> mean_h = rm::mean(a_gpu);
    check_rel(mean_h[0].x, ref_sum_x / N, 1e-4, "mean.x");
    check_rel(mean_h[0].y, ref_sum_y / N, 1e-4, "mean.y");
    check_rel(mean_h[0].z, ref_sum_z / N, 1e-4, "mean.z");

    // --- rm::cov (cov_kernel<1024>) vs CPU reference ---
    rm::Memory<rm::Matrix3x3> cov_h = rm::cov(a_gpu, b_gpu);
    for(int r = 0; r < 3; r++)
    {
        for(int c = 0; c < 3; c++)
        {
            check_rel(cov_h[0](r, c), ref_cov[r][c], 1e-4,
                "cov(" + std::to_string(r) + "," + std::to_string(c) + ")");
        }
    }
    std::cout << "cov(0,0) = " << cov_h[0](0, 0) << " (ref " << ref_cov[0][0] << ")" << std::endl;

    // --- determinism: two runs must be bit-identical ---
    rm::Memory<rm::Vector> sum2_h = rm::sum(a_gpu);
    rm::Memory<rm::Matrix3x3> cov2_h = rm::cov(a_gpu, b_gpu);
    require(sum2_h[0].x == sum_h[0].x && sum2_h[0].y == sum_h[0].y
        && sum2_h[0].z == sum_h[0].z, "sum not bit-identical across two runs");
    for(int r = 0; r < 3; r++)
        for(int c = 0; c < 3; c++)
            require(cov2_h[0](r, c) == cov_h[0](r, c),
                "cov not bit-identical across two runs");

    std::cout << "PASS: rm::sum/mean/cov match CPU reference and are deterministic" << std::endl;
    return 0;
}
