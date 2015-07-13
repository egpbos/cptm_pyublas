cimport cython

cimport numpy as np
import numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def gibbs_inner(self):
    cdef np.ndarray[long, ndim=2, mode='c'] z = self.z
    cdef np.ndarray[long, ndim=2, mode='c'] ndk = self.ndk
    cdef np.ndarray[long, ndim=2, mode='c'] nkw = self.nkw
    cdef np.ndarray[long, ndim=1, mode='c'] nk = self.nk
    cdef np.ndarray[long, ndim=2, mode='c'] ns = self.ns
    cdef np.ndarray[long, ndim=3, mode='c'] nrs = self.nrs

    cdef Py_ssize_t d, w_id, i, persp
    cdef long topic, opinion, nTopics = self.nTopics, VT = self.VT

    cdef double alpha = self.alpha, beta = self.beta

    for d, persp, d_p, doc in self.corpus:
        for w_id, i in self.corpus.words_in_document(doc, 'topic'):
            topic = z[d, i]

            ndk[d, topic] -= 1
            nkw[topic, w_id] -= 1
            nk[topic] -= 1

            p = p_z(ndk[d], nkw[:, w_id], nk, alpha, beta, nTopics, VT)
            topic = self.sample_from(p)

            z[d, i] = topic
            ndk[d, topic] += 1
            nkw[topic, w_id] += 1
            nk[topic] += 1

        for w_id, i in self.corpus.words_in_document(doc, 'opinion'):
            opinion = self.x[persp][d_p, i]

            nrs[persp, opinion, w_id] -= 1
            ns[persp, opinion] -= 1

            p = self.p_x(persp, d, w_id)
            opinion = self.sample_from(p)

            self.x[persp][d_p, i] = opinion
            nrs[persp, opinion, w_id] += 1
            ns[persp, opinion] += 1


@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
cdef p_z(np.ndarray[long, ndim=1, mode='c'] ndk_d,
         np.ndarray[long, ndim=1] nkw_w_id,
         np.ndarray[long, ndim=1, mode='c'] nk,
         double alpha, double beta, long nTopics, long VT):
    """Calculate (normalized) probabilities for p(w|z) (topics).

    The probabilities are normalized, because that makes it easier to
    sample from them.
    """
    cdef np.ndarray[double, ndim=1, mode='c'] p

    # f1 = (ndk_d+alpha) / (np.sum(ndk_d) + nTopics*alpha)
    p = np.empty(ndk_d.shape[0], dtype=np.double)
    cdef double total = 0
    for i in range(p.shape[0]):
        p[i] = ndk_d[i] + alpha
        total += ndk_d[i]
    for i in range(p.shape[0]):
        p[i] /= (total + nTopics * alpha)

    # f2 = (nkw_w_id + beta) / (nk+beta*VT)
    total = 0
    for i in range(p.shape[0]):
        p[i] *= (nkw_w_id[i] + beta) / (nk[i] + beta * VT)
        total += p[i]
    # p = (f1*f2) / np.sum(f1*f2)
    for i in range(p.shape[0]):
        p[i] /= total

    return p
