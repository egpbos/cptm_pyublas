#include <cstdint>
#include <random>
#include <iterator>  // std::distance
#include "./crunch.hpp"

using namespace crunch;

// Calculate (normalized) probabilities for p(w|z) (topics).
// The probabilities are normalized, because that makes it easier to
// sample from them.

void crunch::p_z(long_vector ndk, int64_t d, long_vector nkw, int64_t w_id,
                 long_vector nk, double alpha, double beta, int64_t VT,
                 double_vector p) {
//             p = p_z(ndk[d], nkw[:, w_id], nk, alpha, beta, VT, p)
  npy_intp len_p = p.dims()[0];

  // pointers to numpy array data
  int64_t *ndk_ptr = ndk.data().begin();
  int64_t *nkw_ptr = nkw.data().begin();
  int64_t *nk_ptr = nk.data().begin();
  double *p_ptr = p.data().begin();

  // offsets
  npy_intp ndk_off = ndk.dims()[1] * static_cast<npy_intp>(d);
  npy_intp nkw_off = static_cast<npy_intp>(w_id);

  // strides
  npy_intp nkw_str1 = nkw.dims()[1];

  // double total_ndk = 0;
  double total = 0;
  double beta_VT = beta * static_cast<double>(VT);
  for (npy_intp ix = 0; ix < len_p; ++ix) {
    npy_intp ndk_ix = ndk_off + ix;
    p_ptr[ix] = (static_cast<double>(ndk_ptr[ndk_ix]) + alpha) *
                (static_cast<double>(nkw_ptr[nkw_off + nkw_str1 * ix]) + beta) /
                (static_cast<double>(nk_ptr[ix]) + beta_VT);
    // total_ndk += ndk_ptr[ndk_ix];
    total += p_ptr[ix];
  }

  // double norm = 1. / (total_ndk + static_cast<double>(len_p) * alpha) / total;
  // Note: no need to normalize twice! Just once is fine.
  double norm = 1. / total;
  for (auto pp = p.begin(); pp != p.end(); ++pp) {
    *pp *= norm;
  }
}

    // cdef double total = 0
    // for i in range(p.shape[0]):
    //     p[i] = ndk_d[i] + alpha
    //     total += ndk_d[i]
    // for i in range(p.shape[0]):
    //     p[i] /= (total + p.shape[0] * alpha)

    // total = 0
    // for i in range(p.shape[0]):
    //     p[i] *= (nkw_w_id[i] + beta) / (nk[i] + beta * VT)
    //     total += p[i]
    // for i in range(p.shape[0]):
    //     p[i] /= total

    // return p


// Calculate (normalized) probabilities for p(w|x) (opinions).
// The probabilities are normalized, because that makes it easier to
// sample from them.

void crunch::p_x(long_vector nrs, long persp, long w_id, long_vector ns,
                 long_vector ndk, long d, long_vector ntd, double beta,
                 long VO, double_vector p) {
  // p_x(nrs[persp, :, w_id], ns[persp], ndk[d], ntd[d], beta_o, VO, p)
  long len_p = p.dims()[0];

  // pointers to numpy array data
  long *nrs_ptr = nrs.data().begin();
  long *ns_ptr = ns.data().begin();
  long *ndk_ptr = ndk.data().begin();
  double *p_ptr = p.data().begin();

  // offsets
  long nrs_off1 = nrs.dims()[2] * nrs.dims()[1] * persp;
  long nrs_off = nrs_off1 + w_id;
  long ns_off = ns.dims()[1] * persp;
  long ndk_off = ndk.dims()[1] * d;

  // strides
  long nrs_str2 = nrs.dims()[2];

  // Note: conversion!
  double ntd_d = static_cast<double>(ntd[d]);

  double total = 0;
  for (long ix = 0; ix < len_p; ++ix) {
    p_ptr[ix] = (nrs_ptr[nrs_off + ix * nrs_str2] + beta) / (ns_ptr[ns_off + ix] + beta * VO) * (ndk_ptr[ndk_off + ix]/ntd_d);
    total += p_ptr[ix];
  }

  for (auto pp = p.begin(); pp != p.end(); ++pp) {
    *pp /= total;
  }
}


int64_t crunch::sample_from(double_vector p, class Sampler &rng) {
  std::discrete_distribution<long> d(p.begin(), p.end());
  return (static_cast<int64_t>(d(rng.gen)));
}

int64_t crunch::sample_from2(double_vector p, class Sampler &rng) {
  std::uniform_real_distribution<> dis(0, 1);
  double r = dis(rng.gen);
  double s = 0;
  for (auto pp = p.begin(); pp != p.end(); ++pp) {
    s += *pp;
    if (s >= r) {
      return static_cast<int64_t>(std::distance(p.begin(), pp));
    }
  }
  // Might occur because of floating point inaccuracies:
  return static_cast<int64_t>(std::distance(p.begin(), p.end()));
}

int64_t crunch::sample_from3(double_vector p, class Sampler2 &rng) {
  double r = rng.draw_uniform();
  double s = 0;
  for (auto pp = p.begin(); pp != p.end(); ++pp) {
    s += *pp;
    if (s >= r) {
      return static_cast<int64_t>(std::distance(p.begin(), pp));
    }
  }
  // Might occur because of floating point inaccuracies:
  return static_cast<int64_t>(std::distance(p.begin(), p.end()));
}