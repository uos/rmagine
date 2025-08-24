#include <rmagine/math/types.h>
#include <rmagine/math/math.h>
#include <rmagine/math/linalg.h>

namespace rmagine
{

template<typename DataT>
Matrix3x3_<DataT> omega_hat(const Vector3_<DataT>& w)
{
  Matrix3x3_<DataT> W;
  W(0,0) =  0.0; W(0,1) = -w.z; W(0,2) =  w.y;
  W(1,0) =  w.z; W(1,1) =  0.0; W(1,2) = -w.x;
  W(2,0) = -w.y; W(2,1) =  w.x; W(2,2) =  0.f;
  return W;
}

template<typename DataT>
Matrix3x3_<DataT> rodrigues(
  const Vector3_<DataT>& axis, DataT theta)
{
  // Handle degenerate axis or zero rotation
  const DataT n2 = axis.x*axis.x + axis.y*axis.y + axis.z*axis.z;
  if(n2 < 1e-24f || std::abs(theta) < 1e-12f) 
  {
    return Matrix3x3_<DataT>::Identity();
  }

  // Normalize axis
  const DataT invn = 1.0f / std::sqrt(n2);
  Vector3_<DataT> k{axis.x * invn, axis.y * invn, axis.z * invn};

  Matrix3x3_<DataT> I = Matrix3x3_<DataT>::Identity();
  Matrix3x3_<DataT> K = omega_hat(k);
  Matrix3x3_<DataT> K2 = K * K; // rmagine's operator*

  // Stable small-angle branch
  const DataT t2 = theta * theta;
  Matrix3x3_<DataT> R;

  if(t2 < 1e-12f) 
  {
    // R ≈ I + θ K + 1/2 θ^2 K^2
    const float a1 = theta;
    const float a2 = 0.5f * t2;
    for(int r=0;r<3;++r)
    {
      for(int c=0;c<3;++c)
      {
        R(r,c) = I(r,c) + a1 * K(r,c) + a2 * K2(r,c);
      }
    }
    return R;
  }

  const DataT s = std::sin(theta);
  const DataT c = std::cos(theta);
  const DataT one_c = 1.0f - c;

  // R = I + sinθ K + (1-cosθ) K^2
  for(int r=0;r<3;++r)
  { 
    for(int c=0;c<3;++c)
    {
      R(r,c) = I(r,c) + s * K(r,c) + one_c * K2(r,c);
    }
  }
  return R;
}

template<typename DataT>
Matrix3x3_<DataT> rodrigues(const Vector3_<DataT>& omega)
{
  const float theta = std::sqrt(omega.x*omega.x + omega.y*omega.y + omega.z*omega.z);
  if(theta < 1e-12f) 
  {
      return Matrix3x3_<DataT>::Identity();
  }
  const float inv = 1.0f / theta;
  Vector3 k{omega.x * inv, omega.y * inv, omega.z * inv};
  return rodrigues(k, theta);
}

template<typename DataT>
Vector3_<DataT> so3_log(const Matrix3x3_<DataT>& R)
{
  // clamp trace for numerical stability
  DataT tr = R(0,0) + R(1,1) + R(2,2);
  tr = std::max(-1.0f, std::min(3.0f, tr));

  DataT c = (tr - 1.0f) * 0.5f;
  c = std::max(-1.0f, std::min(1.0f, c));

  const DataT theta = std::acos(c);
  if(theta < 1e-12f) {
      return {0.f, 0.f, 0.f};
  }

  // omega_hat = (R - R^T) / (2 sin θ)
  const DataT s = std::sin(theta);
  const DataT scale = 1.0f / (2.0f * s);

  // (R - R^T) = 2 * [omega_hat]× * sinθ
  // Extract vector from skew-symmetric part
  const DataT wx = (R(2,1) - R(1,2)) * scale;
  const DataT wy = (R(0,2) - R(2,0)) * scale;
  const DataT wz = (R(1,0) - R(0,1)) * scale;

  // Direction is unit axis, multiply by θ to get ω
  Vector3 axis{wx, wy, wz};
  const DataT n = std::sqrt(wx*wx + wy*wy + wz*wz);
  if(n < 1e-20f) 
  {
    return {0.f, 0.f, 0.f};
  }
  const DataT mul = theta / n;
  return {axis.x * mul, axis.y * mul, axis.z * mul};
}

template<typename DataT>
Matrix3x3_<DataT> so3_exp(
  const Vector3_<DataT> omega)
{
  // Delegate to Rodrigues (ω = θ k)
  return rodrigues(omega);
}

// --- Left Jacobian of SO(3) ---
// J_l(w) = I + (1-cosθ)/θ^2 [w]x + (θ - sinθ)/θ^3 [w]x^2
// Small-angle series: I - 1/2 [w]x + 1/6 [w]x^2
template<typename DataT>
Matrix3x3_<DataT> so3_left_jacobian(const Vector3_<DataT>& w)
{
  const DataT theta2 = w.x*w.x + w.y*w.y + w.z*w.z;
  Matrix3x3_<DataT> I = Matrix3x3_<DataT>::Identity();

  if(theta2 < 1e-16f) 
  {
    // series expansion
    Matrix3x3_<DataT> W  = omega_hat(w);
    Matrix3x3_<DataT> W2 = W * W;
    Matrix3x3_<DataT> J;
    // J ≈ I - 1/2 W + 1/6 W^2
    for(int r=0;r<3;++r)
    {
      for(int c=0;c<3;++c)
      {
        J(r,c) = I(r,c) - 0.5f*W(r,c) + (1.0f/6.0f)*W2(r,c);
      }
    }
    return J;
  }

  const DataT theta = std::sqrt(theta2);
  Matrix3x3_<DataT> W  = omega_hat(w);
  Matrix3x3_<DataT> W2 = W * W;

  const DataT a = (1.0f - std::cos(theta)) / (theta2);
  const DataT b = (theta - std::sin(theta)) / (theta2 * theta);

  Matrix3x3_<DataT> J;
  for(int r=0;r<3;++r)
  { 
    for(int c=0;c<3;++c)
    {
      J(r,c) = I(r,c) + a*W(r,c) + b*W2(r,c);
    }
  }
  return J;
}

template<typename DataT>
Matrix3x3_<DataT> so3_left_jacobian_inv(const Vector3_<DataT>& w)
{
  const DataT theta2 = w.x*w.x + w.y*w.y + w.z*w.z;
  Matrix3x3_<DataT> I = Matrix3x3_<DataT>::Identity();

  if(theta2 < 1e-16f) 
  {
    Matrix3x3_<DataT> W  = omega_hat(w);
    Matrix3x3_<DataT> W2 = W * W;
    Matrix3x3_<DataT> Jinv;
    for(int r=0;r<3;++r)
    {
      for(int c=0;c<3;++c)
      {
        Jinv(r,c) = I(r,c) + 0.5f*W(r,c) + (1.0f/12.0f)*W2(r,c);
      }
    }
    return Jinv;
  }

  const DataT theta = std::sqrt(theta2);
  const DataT half  = 0.5f * theta;
  const DataT cot_half = std::cos(half) / std::sin(half);

  Matrix3x3_<DataT> W  = omega_hat(w);
  Matrix3x3_<DataT> W2 = W * W; // rmagine’s operator*

  const DataT a = -0.5f;
  const DataT b = (1.0f - theta * cot_half) / theta2;

  Matrix3x3_<DataT> Jinv;
  for(int r=0;r<3;++r)
  {
    for(int c=0;c<3;++c)
    {
      Jinv(r,c) = I(r,c) + a*W(r,c) + b*W2(r,c);
    }
  }
  return Jinv;
}

template<typename DataT>
Transform_<DataT> se3_exp(
  const Vector3_<DataT>& v, 
  const Vector3_<DataT>& w)
{
  Transform_<DataT> T;
  // rotation from so3 exp
  T.R = so3_exp(w);
  // translation with left Jacobian
  const Matrix3x3_<DataT> J = so3_left_jacobian(w);
  T.t = J * v;
  return T;
}

template<typename DataT>
std::pair<Vector3_<DataT>, Vector3_<DataT> > se3_log(
  const Transform_<DataT>& T)
{
  // extract rotation and translation
  const Matrix3x3_<DataT> R = T.R;

  // so3 log
  const Vector3_<DataT> w = so3_log(R);

  // translation part via inverse Jacobian
  const Matrix3x3_<DataT> Jinv = so3_left_jacobian_inv(w);
  const Vector3_<DataT> v = Jinv * T.t;

  return {v, w};
}

} // namespace rmagine
