/**
 * Templated math must run in the precision of its data type:
 * no float-only functions (sqrtf, sinf, ...) or float intermediates for DataT = double.
 */
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <string>

#include <rmagine/math/types.h>
#include <rmagine/math/lie.h>

namespace rm = rmagine;

static void check(bool cond, const std::string& what, double err)
{
  std::cout << (cond ? "[ OK ] " : "[FAIL] ") << what << " (error " << err << ")" << std::endl;
  if(!cond)
  {
    throw std::runtime_error(what);
  }
}

static double quat_err(const rm::Quaterniond& a, const rm::Quaterniond& b)
{
  // q and -q are the same rotation
  const double d1 = std::fabs(a.x - b.x) + std::fabs(a.y - b.y) + std::fabs(a.z - b.z) + std::fabs(a.w - b.w);
  const double d2 = std::fabs(a.x + b.x) + std::fabs(a.y + b.y) + std::fabs(a.z + b.z) + std::fabs(a.w + b.w);
  return std::min(d1, d2);
}

int main(int argc, char** argv)
{
  const double tol = 1e-13;

  // l2norm
  {
    const rm::Vector3d v{3.0 + 1e-9, 4.0, 1e-3};
    const double err = std::fabs(v.l2norm() - std::sqrt((3.0 + 1e-9) * (3.0 + 1e-9) + 16.0 + 1e-6));
    check(err < tol, "Vector3d::l2norm", err);
    const rm::Vector2d v2{3.0 + 1e-9, 4.0};
    const double err2 = std::fabs(v2.l2norm() - std::sqrt((3.0 + 1e-9) * (3.0 + 1e-9) + 16.0));
    check(err2 < tol, "Vector2d::l2norm", err2);
  }

  // EulerAngles -> Quaternion, compared with an explicit double computation
  const rm::EulerAnglesd e{0.1 + 1e-9, -0.2, 0.3};
  {
    const double cr = std::cos(e.roll / 2), sr = std::sin(e.roll / 2);
    const double cp = std::cos(e.pitch / 2), sp = std::sin(e.pitch / 2);
    const double cy = std::cos(e.yaw / 2), sy = std::sin(e.yaw / 2);
    rm::Quaterniond expected;
    expected.x = sr * cp * cy - cr * sp * sy;
    expected.y = cr * sp * cy + sr * cp * sy;
    expected.z = cr * cp * sy - sr * sp * cy;
    expected.w = cr * cp * cy + sr * sp * sy;
    const rm::Quaterniond q = e;
    const double err = quat_err(q, expected);
    check(err < tol, "EulerAnglesd -> Quaterniond", err);

    // EulerAngles -> Matrix -> Quaternion
    const rm::Matrix3x3d M = e;
    rm::Quaterniond qm;
    qm.set(M);
    const double err_m = quat_err(qm, expected);
    check(err_m < tol, "EulerAnglesd -> Matrix3x3d -> Quaterniond", err_m);

    // Matrix -> EulerAngles
    const rm::EulerAnglesd e2 = M;
    const double err_e = std::fabs(e2.roll - e.roll) + std::fabs(e2.pitch - e.pitch) + std::fabs(e2.yaw - e.yaw);
    check(err_e < tol, "Matrix3x3d -> EulerAnglesd", err_e);
  }

  // so3 exp/log round trip
  {
    const rm::Vector3d w{0.3, -0.2 + 1e-9, 0.1};
    const rm::Matrix3x3d R = rm::so3_exp(w);
    const rm::Vector3d w2 = rm::so3_log(R);
    const double err = (w2 - w).l2norm();
    check(err < 1e-12, "so3_log(so3_exp(w)) double", err);

    rm::Quaterniond q;
    q.set(R);
    const double err_q = (rm::so3_log(q) - w).l2norm();
    check(err_q < 1e-12, "so3_log(Quaterniond) double", err_q);
  }

  // Quaternion::pow
  {
    const rm::Vector3d w{0.0, 0.0, 1e-7}; // tiny rotation
    rm::Quaterniond q;
    q.set(rm::so3_exp(w));
    const rm::Quaterniond q_half = q.pow(0.5);
    const double err = (rm::so3_log(q_half) - w * 0.5).l2norm();
    check(err < 1e-18, "Quaterniond::pow small angle", err);

    const rm::Vector3d w_big{0.5, -1.0, 0.25};
    rm::Quaterniond qb;
    qb.set(rm::so3_exp(w_big));
    const double err_b = (rm::so3_log(qb.pow(0.3)) - w_big * 0.3).l2norm();
    check(err_b < 1e-12, "Quaterniond::pow", err_b);

    const rm::Quaterniond q_id = rm::Quaterniond::Identity().pow(0.5);
    const double err_id = quat_err(q_id, rm::Quaterniond::Identity());
    check(err_id < tol, "Quaterniond::pow identity", err_id);
  }

  return 0;
}
