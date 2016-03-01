#pragma once 

#include <random>
#include <cstdint>
// Boost Includes ==============================================================
#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
// #include <boost/cstdint.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
// PyUblas
#include <pyublas/numpy.hpp>

namespace crunch {
  typedef pyublas::numpy_vector<int> int_vector;
  typedef pyublas::numpy_vector<long> long_vector;
  typedef pyublas::numpy_vector<float> float_vector;
  typedef pyublas::numpy_vector<double> double_vector;

  class Sampler {
  // private:
  //   std::random_device rd;
  public:
    std::mt19937 gen;
    Sampler(): gen(std::random_device()()) {}
  };

  class Sampler2 {
  // private:
  //   std::random_device rd;
  public:
    std::mt19937 gen;
    std::uniform_real_distribution<> uniform;
    Sampler2(): gen(std::random_device()()), uniform(0, 1) {}
    double draw_uniform() {
      return uniform(gen);
    }
  };

  // Testspul
  extern int_vector test1(int n);
  extern void test2(float_vector v);
  extern void p_x(long_vector nrs, long persp, long w_id, long_vector ns,
                  long_vector ndk, long d, long_vector ntd, double beta,
                  long VO, double_vector p);
  extern void p_z(long_vector ndk, long d, long_vector nkw, long w_id,
                  long_vector nk, double alpha, double beta, long VT,
                  double_vector p);
  extern int64_t sample_from(double_vector p, class Sampler &rng);
  extern int64_t sample_from2(double_vector p, class Sampler &rng);
  extern int64_t sample_from3(double_vector p, class Sampler2 &rng);
}