#include <iostream>
#include <rmagine/math/lie.h>

#include <rmagine/math/memory_math.h>
#include <random>

#include <rmagine/util/prints.h>


namespace rm = rmagine;

// Small random twist around zero
rm::Transform sample_single(const rm::Transform& T0,
                           std::mt19937& rng,
                           float rot_std = 0.1f,
                           float trans_std   = 0.02f)
{
  std::normal_distribution<float> N(0.0f, 1.0f);

  // rotation noise in radians
  float s_rot = rot_std;
  rm::Vector3 w{ s_rot * N(rng), s_rot * N(rng), s_rot * N(rng) };

  // translation noise
  float s_trans = trans_std;
  rm::Vector3 v{ s_trans * N(rng), s_trans * N(rng), s_trans * N(rng) };

  // dT = exp([v,w])
  rm::Transform dT = se3_exp(v, w);

  // compose: T = T0 ∘ dT
  rm::Matrix3x3 R0; R0.set(T0.R);
  rm::Matrix3x3 Rd; Rd.set(dT.R);

  rm::Transform T;
  rm::Matrix3x3 Rn = R0 * Rd;
  T.R.set(Rn);
  T.t = T0.t + (R0 * dT.t);
  return T;
}


rm::Memory<rm::Transform> sample(
  rm::Transform Tmean, 
  std::mt19937& rng, 
  size_t n_samples, 
  float rot_std, 
  float trans_std)
{
  rm::Memory<rm::Transform> Ts(n_samples);

  for(size_t i=0; i<n_samples; i++)
  {
    Ts[i] = sample_single(Tmean, rng, rot_std, trans_std);
  }

  return Ts;
}

int main(int argc, char** argv)
{
  std::cout << "RMAGINE CORE MATH LIE" << std::endl;
  std::mt19937 rng(42);

  rm::Transform T_true;
  rm::Vector3 w0{ 0.20f, -0.05f, 0.08f };         // rad
  rm::Matrix3x3 R0 = rm::so3_exp(w0);
  T_true.t = rm::Vector3{ 1.0f, -0.5f, 0.7f };    // meters

  T_true.R.set(R0);

  rm::Memory<rm::Transform> Ts = sample(T_true, rng, 100, 0.02, 0.02);

  std::cout << "Samples: " << Ts << std::endl;

  // 4) Compute Karcher mean (uniform weights)
  rm::Transform T_karcher_mean = rm::karcher_mean(Ts);
  rm::Transform T_markley_mean = rm::markley_mean(Ts);

  std::cout << "Results: " << std::endl;
  std::cout << "- True: " << T_true << std::endl;
  std::cout << "- Mean (karcher):" << T_karcher_mean << ", delta: " << ~T_true * T_karcher_mean << std::endl;
  std::cout << "- Mean (markley): " << T_markley_mean << ", delta: " << ~T_true * T_markley_mean << std::endl;

  return 0;
}