import numpy as np
from numpy import random
from numpy.testing import assert_array_almost_equal
from DilipadTopicModelling.gibbs_inner import p_z


def p_z_reference(d, w_id, alpha, beta, nTopics, VT, ndk, nkw, nk):
    """Calculate (normalized) probabilities for p(w|z) (topics).

    The probabilities are normalized, because that makes it easier to
    sample from them.
    """
    f1 = (ndk[d]+alpha) / (np.sum(ndk[d])+nTopics*alpha)
    f2 = (nkw[:, w_id]+beta) / (nk+beta*VT)

    p = f1*f2
    return p / np.sum(p)


def test_p_z():
    """Compare output from reference p_z to cython p_z"""
    VT = 10
    DT = 10
    nTopics = random.randint(50, 300)
    alpha = random.random()
    beta = random.random()

    ndk = random.randint(1000, size=(DT, nTopics))
    nkw = random.randint(5000, size=(nTopics, VT))
    nk = random.randint(5000, size=(nTopics))
    #ntd = np.sum(nkw, axis=0)

    for w_id in range(VT):
        for d in range(DT):
            pz1 = p_z_reference(d, w_id, alpha, beta, nTopics, VT, ndk, nkw, nk)
            pz2 = p_z(ndk[d], nkw[:, w_id], nk, alpha, beta, nTopics, VT)

            yield almost_equal, pz1, pz2


def almost_equal(ar1, ar2):
    assert_array_almost_equal(ar1, ar2)
