from nose.tools import assert_equal

from .. import generateCPTCorpus
from .. import CPTCorpus
import shutil
import glob
from gensim import corpora
from collections import Counter


def setup():
    global data_dir
    global persp_dirs
    global documents
    global corpus

    data_dir = 'test_data/'
    persp_dirs = ['{}{}'.format(data_dir, p) for p in ('p0', 'p1')]
    documents = generateCPTCorpus.generate_cpt_corpus(data_dir)
    corpus = CPTCorpus.CPTCorpus(persp_dirs)


def teardown():
    shutil.rmtree(data_dir)


def test_perspective_directories_exist():
    """Test existence of generated directories"""
    generated_dirs = glob.glob('{}*'.format(data_dir))
    generated_dirs.sort()

    assert_equal(generated_dirs, persp_dirs)


def test_iterate_over_topic_words():
    """Test iteration over topic words."""
    for d, persp, d_p, doc in corpus:
        count_words = Counter()
        for w_id, i in corpus.words_in_document(doc, 'topic'):
            word = str(corpus.topicDictionary.get(w_id))
            count_words[word] += 1
        p_name = corpus.perspectives[persp].name
        yield equal_obj, documents[p_name][d_p]['topic'], count_words


def test_iterate_over_opinion_words():
    """Test iteration over opinion words."""
    for d, persp, d_p, doc in corpus:
        count_words = Counter()
        for w_id, i in corpus.words_in_document(doc, 'opinion'):
            word = str(corpus.opinionDictionary.get(w_id))
            count_words[word] += 1
        p_name = corpus.perspectives[persp].name
        yield equal_obj, documents[p_name][d_p]['opinion'], count_words


def equal_obj(obj1, obj2):
    assert_equal(obj1, obj2)


def test_corpus_lengths():
    """Compare length of corpus with the sum of the perspective corpora"""
    per_perspective = [len(p) for p in corpus.perspectives]
    assert_equal(len(corpus), sum(per_perspective))


def test_calculate_tf_and_df():
    """Test calculation of tf and df"""
    topic_df = Counter()
    opinion_df = Counter()

    topic_tf = Counter()
    opinion_tf = Counter()

    corpus.calculate_tf_and_df()

    for persp, data in documents.iteritems():
        for counters in data:

            for word, freq in counters['topic'].iteritems():
                [(w_id, f)] = corpus.topicDictionary.doc2bow([word])
                topic_tf[w_id] += freq
                topic_df[w_id] += 1


            for word, freq in counters['opinion'].iteritems():
                [(w_id, f)] = corpus.opinionDictionary.doc2bow([word])
                opinion_tf[w_id] += freq
                opinion_df[w_id] += 1

    yield equal_obj, topic_tf, corpus.topic_tf
    yield equal_obj, topic_df, corpus.topic_df

    yield equal_obj, opinion_tf, corpus.opinion_tf
    yield equal_obj, opinion_df, corpus.opinion_df


def test_word_lists():
    topicWords = ['sun', 'ice_cream', 'beach', 'vanilla', 'chocolate',
                  'broccoli', 'carrot']
    topicWords.sort()
    tw = corpus.topic_words()
    tw.sort()

    yield assert_equal, topicWords, tw

    opinionWords = ['warm', 'swimming', 'sunny', 'bad', 'good', 'cold']
    opinionWords.sort()
    ow = corpus.opinion_words()
    ow.sort()

    yield assert_equal, opinionWords, ow


def test_no_testSet():
    """CPTCorpus without testSplit has no test sets"""
    for p in corpus.perspectives:
        yield assert_equal, hasattr(p, 'testSet'), False


def test_testSet():
    """CPTCorpus with testSplit has test sets of particular length"""
    corpus2 = CPTCorpus.CPTCorpus(persp_dirs, testSplit=20)

    yield assert_equal, len(corpus2), 8

    for p in corpus2.perspectives:
        yield assert_equal, hasattr(p, 'testSet'), True
        yield assert_equal, len(p.trainSet), 4
        yield assert_equal, len(p.testSet), 1


def test_illigal_values_for_testSplit():
    """No test set when value for testSplit parameter is illegal"""
    values = [-1, 0, 100, 1000]
    for v in values:
        corpus2 = CPTCorpus.CPTCorpus(persp_dirs, testSplit=v)
        for p in corpus2.perspectives:
            yield assert_equal, hasattr(p, 'testSet'), False


def test_loop_over_testSet():
    """Test loop over documents in testSet"""
    corpus2 = CPTCorpus.CPTCorpus(persp_dirs, testSplit=20)
    for d, persp, d_p, doc in corpus2.testSet():
        pass

    yield assert_equal, d, 1
    yield assert_equal, persp, len(corpus.perspectives)-1
    yield assert_equal, d_p, 0
