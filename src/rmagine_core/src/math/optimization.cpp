#include "rmagine/math/optimization.h"

namespace rmagine
{


bool check(const Quaternion& q)
{
  return std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z) && std::isfinite(q.w);
}

Transform umeyama_transform(
    const Vector3& d,
    const Vector3& m,
    const Matrix3x3& C,
    const unsigned int n_meas)
{
  Transform ret;

  if(n_meas > 0)
  {
    // intermediate storage needed (yet)
    Matrix3x3 U, S, V;
    svd(C, U, S, V);
    S.setIdentity();
    if(U.det() * V.det() < 0)
    {
      S(2, 2) = -1;
    }
    ret.R.set(U * S * V.transpose());
    ret.R.normalizeInplace();
    
    // There are still situations where SVD results in an invalid result.
    // I assume there are numerical issues inside the rm::svd function.
    // TODO: Take an evening and check if we can write rm::svd more stable
    // or if we can detect situations earlier and skip a lot of computations
    // This is a suboptimal workaround that covers some issues:
    if(!check(ret.R))
    {
      ret.R.setIdentity();
    }
    
    ret.t = m - ret.R * d;
  } else {
    ret.setIdentity();
  }

  return ret;
}

Transform umeyama_transform(
    const CrossStatistics& stats)
{
  return umeyama_transform(stats.dataset_mean, stats.model_mean, stats.covariance, stats.n_meas);
}

} // namespace rmagine