from nose.tools import assert_equal, assert_true, assert_false

from .. import generateCPTCorpus
from .. import CPTCorpus
import shutil
from DilipadTopicModelling.CPT_Gibbs import GibbsSampler
from pandas import DataFrame
from numpy.testing import assert_almost_equal
from numpy import array
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
    sampler = GibbsSampler(corpus, nTopics=3, nIter=5, out_dir=out_dir)
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


def test_check_index():
    """Return index of nIter-1 for certain inputs of _check_index"""
    values = [None, sampler.nIter]
    for v in values:
        yield assert_equal, sampler._check_index(v), sampler.nIter-1


def test_check_start():
    """Test _check_start_and_end with values for start"""
    values = [(-1, 0), (sampler.nIter+5, 0),
              (sampler.nIter-3, sampler.nIter-3)]
    for input_v, output_v in values:
        start, end = sampler._check_start_and_end(input_v, sampler.nIter)
        yield assert_equal, start, output_v


def test_check_end():
    """Test _check_start_and_end with values for start"""
    values = [(-1, sampler.nIter), (sampler.nIter+5, sampler.nIter)]
    for input_v, output_v in values:
        start, end = sampler._check_start_and_end(0, input_v)
        yield assert_equal, end, output_v


def test_check_start_and_end():
    """Test _check_start_and_end with values for start and end"""
    # start > end
    start, end = sampler._check_start_and_end(3, 2)
    yield assert_equal, start, 0
    yield assert_equal, end, 2


def test_get_phi_topic_from_memory():
    sampler2 = GibbsSampler(corpus, nTopics=3, nIter=5)
    sampler2._initialize()
    sampler2.run()

    f = sampler2.get_phi_topic_file_name(0)
    yield assert_false, os.path.isfile(f)

    phi = sampler2.get_phi_topic()
    yield assert_equal, phi.shape, (sampler2.nTopics, sampler2.VT)

    phi = sampler2.get_phi_topic(index=2)
    yield assert_equal, phi.shape, (sampler2.nTopics, sampler2.VT)

    phi = sampler2.get_phi_topic(start=0, end=5)
    yield assert_equal, phi.shape, (sampler2.nTopics, sampler2.VT)


def test_get_theta_from_memory():
    sampler2 = GibbsSampler(corpus, nTopics=3, nIter=5)
    sampler2._initialize()
    sampler2.run()

    f = sampler2.get_theta_file_name(0)
    yield assert_false, os.path.isfile(f)

    r = sampler2.get_theta()
    yield assert_equal, r.shape, (sampler2.DT, sampler2.nTopics)

    r = sampler2.get_theta(index=2)
    yield assert_equal, r.shape, (sampler2.DT, sampler2.nTopics)

    r = sampler2.get_theta(start=0, end=5)
    yield assert_equal, r.shape, (sampler2.DT, sampler2.nTopics)


def test_get_phi_opinion_from_memory():
    sampler2 = GibbsSampler(corpus, nTopics=3, nIter=5)
    sampler2._initialize()
    sampler2.run()

    f = sampler2.get_phi_opinion_file_name(0, 0)
    yield assert_false, os.path.isfile(f)

    phi = sampler2.get_phi_opinion()
    yield assert_equal, len(phi), sampler2.nPerspectives
    yield assert_equal, phi[0].shape, (sampler2.nTopics, sampler2.VO)

    phi = sampler2.get_phi_opinion(index=2)
    yield assert_equal, len(phi), sampler2.nPerspectives
    yield assert_equal, phi[0].shape, (sampler2.nTopics, sampler2.VO)

    phi = sampler2.get_phi_opinion(start=0, end=5)
    yield assert_equal, len(phi), sampler2.nPerspectives
    yield assert_equal, phi[0].shape, (sampler2.nTopics, sampler2.VO)


def test_topic_word_perplexity():
    """Minimal test of caluclation of topic word perplexity"""
    corpus2 = CPTCorpus.CPTCorpus(persp_dirs, testSplit=20)
    sampler2 = GibbsSampler(corpus2, nTopics=3, nIter=5, out_dir=out_dir)
    perp = sampler2.topic_word_perplexity()

    yield assert_true, perp > 0.0
    yield assert_true, perp < len(sampler2.corpus.topicDictionary)
