from nose.tools import assert_equal

from .. import generateCPTCorpus
from .. import CPTCorpus
import shutil
from DilipadTopicModelling.CPT_Gibbs import GibbsSampler
from pandas import DataFrame
from numpy.testing import assert_almost_equal


def setup():
    global data_dir
    global persp_dirs
    global documents
    global corpus
    global sampler

    data_dir = 'test_data/'
    persp_dirs = ['{}{}'.format(data_dir, p) for p in ('p0', 'p1')]
    documents = generateCPTCorpus.generate_cpt_corpus(data_dir)
    corpus = CPTCorpus.CPTCorpus(persp_dirs)
    sampler = GibbsSampler(corpus, nTopics=3, nIter=2)
    sampler._initialize()
    sampler.run()


def teardown():
    shutil.rmtree(data_dir)


def test_jensen_shannon_divergence_self():
    """Jensen-Shannon divergence of a vector and itself must be 0"""
    v = [0.2, 0.2, 0.2, 0.2, 0.2]
    df = DataFrame({'p0': v, 'p1': v})

    assert_equal(0.0, sampler.jsd_opinions(df))


def test_jensen_shannon_divergence_symmetric():
    """Jensen-Shannon divergence is symmetric"""
    v1 = [0.2, 0.2, 0.2, 0.2, 0.2]
    v2 = [0.2, 0.2, 0.2, 0.3, 0.1]
    df1 = DataFrame({'p0': v1, 'p1': v2})
    df2 = DataFrame({'p0': v2, 'p1': v1})

    assert_equal(sampler.jsd_opinions(df1), sampler.jsd_opinions(df2))


def test_jensen_shannon_divergence_known_value():
    """Jensen-Shannon divergence of v1 and v2 == 0.01352883"""
    v1 = [0.2, 0.2, 0.2, 0.2, 0.2]
    v2 = [0.2, 0.2, 0.2, 0.3, 0.1]
    df1 = DataFrame({'p0': v1, 'p1': v2})

    assert_almost_equal(0.01352883, sampler.jsd_opinions(df1))
