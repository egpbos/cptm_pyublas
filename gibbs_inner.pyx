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

            p = p_z(d, w_id, ndk, alpha, nTopics, nkw, beta, nk, VT)
            #p = self.p_z(d, w_id)
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


cdef p_z(Py_ssize_t d, Py_ssize_t w_id, np.ndarray[long, ndim=2, mode='c'] ndk,
         double alpha, long nTopics, np.ndarray[long, ndim=2, mode='c'] nkw,
         double beta, np.ndarray[long, ndim=1, mode='c'] nk,
         long VT):
    f1 = (ndk[d]+alpha) / (np.sum(ndk[d])+nTopics*alpha)
    f2 = (nkw[:, w_id]+beta) / (nk+beta*VT)
    p = f1*f2
    return p / np.sum(p)
