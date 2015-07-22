from nose.tools import assert_equal

from .. import generateCPTCorpus
from .. import CPTCorpus
import shutil
from DilipadTopicModelling.CPT_Gibbs import GibbsSampler
from pandas import DataFrame
from numpy.testing import assert_almost_equal
import os.path


def setup():
    global data_dir
    global out_dir
    global persp_dirs
    global documents
    global corpus
    global sampler

    data_dir = 'test_data/'
    out_dir = 'test_output/'
    persp_dirs = ['{}{}'.format(data_dir, p) for p in ('p0', 'p1')]
    documents = generateCPTCorpus.generate_cpt_corpus(data_dir)
    corpus = CPTCorpus.CPTCorpus(persp_dirs)
    sampler = GibbsSampler(corpus, nTopics=3, nIter=2, out_dir=out_dir)
    sampler._initialize()
    sampler.run()


def teardown():
    shutil.rmtree(data_dir)
    shutil.rmtree(out_dir)


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


def test_contrastive_opinions_result_shape():
    """Verify the shape of the output of contrastive_opinions"""
    co = sampler.contrastive_opinions('carrot')
    assert_equal(co.shape, (sampler.VO, sampler.nPerspectives))


def test_contrastive_opinions_prob_distr():
    """Verify that the sum of all columns == 1.0 (probability distribution)"""
    co = sampler.contrastive_opinions('carrot')
    s = co.sum(axis=0)

    for v in s:
        yield assert_almost_equal, v, 1.0


def test_get_parameter_dir_name():
    """Test existence of parameter directory"""
    dir_name = sampler.get_parameter_dir_name()

    assert_equal(os.path.exists(dir_name), True)


def test_get_phi_topic_file_name():
    """Test existence of phi topic parameter files"""
    for i in range(sampler.nIter):
        file_name = sampler.get_phi_topic_file_name(i)
        yield assert_equal, os.path.isfile(file_name), True


def test_get_theta_file_name():
    """Test existence of theta parameter files"""
    for i in range(sampler.nIter):
        file_name = sampler.get_theta_file_name(i)
        yield assert_equal, os.path.isfile(file_name), True


def test_get_phi_opinion_file_name():
    """Test existence of phi opinion parameter files"""
    for i in range(sampler.nIter):
        for p, persp in enumerate(sampler.corpus.perspectives):
            file_name = sampler.get_phi_opinion_file_name(p, i)
            yield assert_equal, os.path.isfile(file_name), True


def test_load_parameters_illegal_index():
    mean = sampler.load_parameters(sampler.PHI_TOPIC)
    values = [-1, sampler.nIter+5, 100000000]
    for v in values:
        result = sampler.load_parameters(sampler.PHI_TOPIC, index=v)
        yield assert_almost_equal, result, mean

# TODO: add test for illegal start values
# TODO: add test for illegal end values
        #self, name, index=None, start=None, end=None):