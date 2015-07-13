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
    cdef np.ndarray[long, ndim=1, mode='c'] ntd = self.ntd

    cdef Py_ssize_t d, w_id, i, persp
    cdef long topic, opinion, nTopics = self.nTopics, VT = self.VT

    cdef double alpha = self.alpha, beta = self.beta

    for d, persp, d_p, doc in self.corpus:
        for w_id, i in self.corpus.words_in_document(doc, 'topic'):
            topic = z[d, i]

            ndk[d, topic] -= 1
            nkw[topic, w_id] -= 1
            nk[topic] -= 1

            p = p_z(d, w_id, alpha, beta, nTopics, VT, ndk, nkw, nk, ntd)
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

cpdef p_z(Py_ssize_t d, Py_ssize_t w_id, double alpha, double beta,
         long nTopics, long VT,
         np.ndarray[long, ndim=2, mode='c'] ndk,
         np.ndarray[long, ndim=2, mode='c'] nkw,
         np.ndarray[long, ndim=1, mode='c'] nk,
         np.ndarray[long, ndim=1, mode='c'] ntd):
    cdef np.ndarray[np.double_t, ndim=1] p     
    p = np.zeros(nTopics, dtype=np.double)

    #cdef long sum_ndkd = 0
    #for z in xrange(nTopics):
    #    sum_ndkd += ndk[d, z]
    for z in xrange(nTopics):
        p[z] = (ndk[d, z]+alpha) / (ntd[d] + alpha * nTopics) * (nkw[z, w_id]+beta) / (nk[z]+beta*VT)
        #print f1
        #print f2
        #p[z] = f1*f2
    # normalize to obtain probabilities
    cdef double p_z_sum  = 0
    for z in xrange(nTopics):
        p_z_sum += p[z]
    
    #solve conversion rounding error
    #cdef double partial_sum = 0
    for z in xrange(nTopics):
        p[z] /= p_z_sum
    #    partial_sum += p[z]
    #    p[nTopics-1] = 1.0 - partial_sum
    return p
#    f1 = (ndk[d]+alpha) / (np.sum(ndk[d])+nTopics*alpha)
#    f2 = (nkw[:, w_id]+beta) / (nk+beta*VT)
#    p = f1*f2
#    return p / np.sum(p)
