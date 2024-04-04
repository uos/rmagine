#include <iostream>
#include <rmagine/math/types.h>
#include <rmagine/util/prints.h>

namespace rm = rmagine;

namespace esoteric_math
{

rm::Quaternion operator-(rm::Quaternion b, rm::Quaternion a)
{
  return ~a * b;
}

rm::Quaternion sqrt(rm::Quaternion q)
{
  // r = norm(y)
  //   theta = np.arccos(y[0]/r)

  //   u = y + 0 # make a copy
  //   u[0] = 0
  //   u /= norm(u)

  //   x = np.sin(theta/2)*u
  //   x[0] = np.cos(theta/2)
  //   x *= r**0.5

  float exponent = 2.0;

  const float r = q.l2norm();
  const float theta = acos(q.w / r);
  const float imag_len = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z);

  rm::Quaternion res;
  res.x = sin(theta / exponent) * (q.x / imag_len) * powf(r, 1.0 / exponent);
  res.y = sin(theta / exponent) * (q.y / imag_len) * powf(r, 1.0 / exponent);
  res.z = sin(theta / exponent) * (q.z / imag_len) * powf(r, 1.0 / exponent);
  res.w = cos(theta / exponent);
  
  return res;
}

// {w*q2.x + x*q2.w + y*q2.z - z*q2.y,
//  w*q2.y - x*q2.z + y*q2.w + z*q2.x,
//  w*q2.z + x*q2.y - y*q2.x + z*q2.w,
//  w*q2.w - x*q2.x - y*q2.y - z*q2.z};
rm::Quaternion operator+(rm::Quaternion a, rm::Quaternion b)
{
  return a * b;
}

rm::Quaternion operator*(rm::Quaternion q, float scalar)
{
  return q.pow(scalar);
  // const float r = q.l2norm();
  // const float theta = acos(q.w / r);
  // const float imag_len = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z);

  // rm::Quaternion res;
  // res.x = sin(theta * scalar) * (q.x / imag_len) * powf(r, scalar);
  // res.y = sin(theta * scalar) * (q.y / imag_len) * powf(r, scalar);
  // res.z = sin(theta * scalar) * (q.z / imag_len) * powf(r, scalar);
  // res.w = cos(theta * scalar);

  // return res;
}

} // namespace esoteric_math

int main(int argc, char** argv)
{
  std::cout << "Quaternion" << std::endl;

  // euler (rpy) -> quat (xyzw)
  // (pi, 0, 0) -> (1, 0, 0, 0)
  // (0, pi, 0) -> (0, 1, 0, 0)
  // (0, 0, pi) -> (0, 0, 1, 0)


  rm::Point Pa = {1.0, 0.0, 0.0};


  rm::Quaternion Ra = rm::EulerAngles{0.0, 0.0, 0.0};

  rm::Quaternion Rb = rm::EulerAngles{0.0, 0.1, 0.2};

  {
    using namespace esoteric_math;

    std::cout << "Ra: " << Ra << std::endl;
    std::cout << "Rb: " << Rb << std::endl;

    auto Rba = (Rb - Ra);

    std::cout << "(Rb - Ra) = " << Rba << std::endl;
    std::cout << "(Rb - Ra) + Ra = " << Rba + Ra << std::endl;
    

    auto Rba_half = sqrt(Rba);

    std::cout << "(Rb - Ra)/2 * 2 + Ra = " << Rba_half * Rba_half * Ra << std::endl;


    auto Rba_third = Rba * (1.0/3.0);

    std::cout << "(Rb - Ra)/3 * 3 + Ra = " << Rba_third * Rba_third * Rba_third * Ra << std::endl;


    std::cout << "Predict:" << std::endl;
    rm::EulerAngles e = Rba * 2.0 + Ra;
    std::cout << "(Rb - Ra) * 2 + Ra = " << e << std::endl;


  }


  return 0;
}