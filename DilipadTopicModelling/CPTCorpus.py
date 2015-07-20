"""Class to access CPT corpus."""
import logging
from gensim import corpora
import glob
import codecs
from itertools import izip
from collections import Counter
import os
import random


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


class CPTCorpus():
    """Class to manage CPT corpus.

    Parameters:
        input : list of str
            A list containing names of directories representing the different
            perspectives. Every directory contains a text file for each
            document in the perspective. A text file contains the topic words
            on the first line and the opinion words on the second line. Words
            are separated by spaces.
        topicDict : str or gensim dictionary
        opinionDict : str or gensim dictionary
        testSplit : int
            Integer specifying the percentage of documents to be used as test
            set (for calculating perplexity).
    """
    def __init__(self, input, topicDict=None, opinionDict=None,
                 testSplit=None):
        logger.info('initialize CPT Corpus with {} perspectives'
                    .format(len(input)))
        input.sort()
        self.perspectives = [Perspective(glob.glob('{}/*.txt'.format(d)), d,
                                         testSplit)
                             for d in input]

        if isinstance(topicDict, str):
            self.topic_dict_file_name = topicDict
            self.load_dictionaries(topicDict=topicDict)
        elif isinstance(topicDict, corpora.Dictionary):
            self.topicDictionary = topicDict

        if isinstance(opinionDict, str):
            self.opinion_dict_file_name = opinionDict
            self.load_dictionaries(opinionDict=opinionDict)
        elif isinstance(opinionDict, corpora.Dictionary):
            self.opinionDictionary = opinionDict

        if not topicDict or not opinionDict:
            self._create_corpus_wide_dictionaries()

        self.topic_dict_file_name = None
        self.opinion_dict_file_name = None

        self.testSplit = testSplit

    def _create_corpus_wide_dictionaries(self):
        """Create dictionaries with all topic and opinion words.

        The created dictionaries contain mappings that can be used with across
        the corpora from different perspectives.
        """
        logger.info('creating corpus wide topic and opinion dictionaries')
        s = self.perspectives[0].trainSet
        self.topicDictionary = s.topicCorpus.dictionary
        self.opinionDictionary = s.opinionCorpus.dictionary
        for p in self.perspectives[1:]:
            s = p.trainSet
            self.topicDictionary.add_documents(s.topicCorpus.get_texts(),
                                               prune_at=None)
            self.opinionDictionary.add_documents(s.opinionCorpus.get_texts(),
                                                 prune_at=None)

    def words_in_document(self, doc, topic_or_opinion):
        """Iterator over the individual word positions in a document."""
        i = 0
        for w_id, freq in doc[topic_or_opinion]:
            for j in range(freq):
                yield w_id, i
                i += 1

    def __iter__(self):
        """Iterator over the documents in the corpus."""
        return self._iterate([p.trainSet for p in self.perspectives])

    def _iterate(self, documentSets):
        doc_id_global = 0
        for i, p in enumerate(documentSets):
            doc_id_perspective = 0
            for doc in p:
                doc['topic'] = self.topicDictionary.doc2bow(doc['topic'])
                doc['opinion'] = self.opinionDictionary.doc2bow(doc['opinion'])

                yield doc_id_global, i, doc_id_perspective, doc

                doc_id_global += 1
                doc_id_perspective += 1

    def __len__(self):
        return sum([len(p) for p in self.perspectives])

    def testSet(self):
        return self._iterate([p.testSet for p in self.perspectives])

    def calculate_tf_and_df(self):
        self.topic_tf = Counter()
        self.topic_df = Counter()

        self.opinion_tf = Counter()
        self.opinion_df = Counter()

        for doc_id_global, i, doc_id_perspective, doc in self:
            doc_words_topic = set()
            for w_id, freq in doc['topic']:
                self.topic_tf[w_id] += freq
                doc_words_topic.add(w_id)
            self.topic_df.update(doc_words_topic)

            doc_words_opinion = set()
            for w_id, freq in doc['opinion']:
                self.opinion_tf[w_id] += freq
                doc_words_opinion.add(w_id)
            self.opinion_df.update(doc_words_opinion)

    def filter_dictionaries(self, minFreq, removeTopTF, removeTopDF):
        logger.info('Filtering dictionaries')
        self.calculate_tf_and_df()
        self.filter_min_frequency(minFreq)
        self.filter_top_tf(removeTopTF)
        self.filter_top_df(removeTopDF)

        self.topicDictionary.compactify()
        self.opinionDictionary.compactify()
        logger.info('topic dictionary: {}'.format(self.topicDictionary))
        logger.info('opinion dictionary: {}'.format(self.opinionDictionary))

    def filter_min_frequency(self, minFreq=5):
        logger.info('Removing tokens from dictionaries with frequency < {}'.
                    format(minFreq))

        logger.debug('topic dict. before: {}'.format(self.topicDictionary))
        self._remove_from_dict_min_frequency(self.topicDictionary,
                                             self.topic_tf, minFreq)
        logger.debug('topic dict. after: {}'.format(self.topicDictionary))

        logger.debug('opinion dict. before: {}'.format(self.opinionDictionary))
        self._remove_from_dict_min_frequency(self.opinionDictionary,
                                             self.opinion_tf, minFreq)
        logger.debug('opinion dict. after: {}'.format(self.opinionDictionary))

    def _remove_from_dict_min_frequency(self, dictionary, tf, minFreq):
        remove_ids = []
        for w_id, freq in tf.iteritems():
            if freq < minFreq:
                remove_ids.append(w_id)
        logger.debug('removing {} tokens'.format(len(remove_ids)))
        dictionary.filter_tokens(bad_ids=remove_ids)

    def filter_top_tf(self, removeTop):
        logger.info('Removing {} most frequent tokens (top tf)'.
                    format(removeTop))

        logger.debug('topic dict. before: {}'.format(self.topicDictionary))
        self._remove_from_dict_top(self.topicDictionary, self.topic_tf,
                                   removeTop)
        logger.debug('topic dict. after: {}'.format(self.topicDictionary))

        logger.debug('opinion dict. before: {}'.format(self.opinionDictionary))
        self._remove_from_dict_top(self.opinionDictionary, self.opinion_tf,
                                   removeTop)
        logger.debug('opinion dict. after: {}'.format(self.opinionDictionary))

    def filter_top_df(self, removeTop):
        logger.info('Removing {} most frequent tokens (top df)'.
                    format(removeTop))

        logger.debug('topic dict. before: {}'.format(self.topicDictionary))
        self._remove_from_dict_top(self.topicDictionary, self.topic_df,
                                   removeTop)
        logger.debug('topic dict. after: {}'.format(self.topicDictionary))

        logger.debug('opinion dict. before: {}'.format(self.opinionDictionary))
        self._remove_from_dict_top(self.opinionDictionary, self.opinion_df,
                                   removeTop)
        logger.debug('opinion dict. after: {}'.format(self.opinionDictionary))

    def _remove_from_dict_top(self, dictionary, frequencies, top=100):
        remove_ids = []
        for w_id, freq in frequencies.most_common(top):
            remove_ids.append(w_id)
        dictionary.filter_tokens(bad_ids=remove_ids)
        logger.debug('removing {} tokens'.format(len(remove_ids)))

    def topic_words(self):
        """Return the list of topic words."""
        return self._create_word_list(self.topicDictionary)

    def opinion_words(self):
        """Return the list of opinion words."""
        return self._create_word_list(self.opinionDictionary)

    def _create_word_list(self, dictionary):
        """Return a list of all words in the dictionary.

        The word list is ordered by word id.
        """
        return [dictionary.get(i) for i in range(len(dictionary))]

    def save_dictionaries(self, directory=None):
        if directory:
            if not os.path.exists(directory):
                os.makedirs(directory)
        else:
            directory = ''

        self.topic_dict_file_name = os.path.join(directory, 'topicDict.dict')
        self.topicDictionary.save(self.topic_dict_file_name(directory))

        self.opinion_dict_file_name = os.path.join(directory,
                                                   'opinionDict.dict')
        self.opinionDictionary.save(self.opinion_dict_file_name(directory))

    def load_dictionaries(self, topicDict=None, opinionDict=None):
        if topicDict:
            self.topicDictionary = corpora.Dictionary.load(topicDict)
        if opinionDict:
            self.opinionDictionary = corpora.Dictionary.load(opinionDict)


class Perspective():
    """Class representing a perspective in cross perspective topic modeling.
    This class contains two text corpora, one for the topic words and one for
    the opinion words. It is used by the class CTPCorpus.

    Parameters:
        input : list of strings
            List containing the file names of the documents in the corpus
            (.txt). A text file contains the topic words on the first line and
            opinion words on the second line.
    """
    def __init__(self, input, directory, testSplit=None):
        name = directory.rsplit('/', 1)[1]
        logger.info('initialize perspective "{}" (path: {} - {} documents)'
                    .format(name, directory, len(input)))
        self.name = name
        self.directory = directory

        if testSplit and (testSplit > 99 or testSplit < 1):
            testSplit = None
            logger.warn('illegal value for testSplit ({}); ' +
                        'not creating test set'.format(testSplit))

        if testSplit:
            splitIndex = int(len(input)/100.0*testSplit)
            logger.info('saving {} of {} documents for testing'.
                        format(splitIndex, len(input)))
            random.shuffle(input)
            self.testFiles = input[:splitIndex]
            input = input[splitIndex:]
            self.testSet = Corpus(self.testFiles)

        self.input = input
        self.trainSet = Corpus(self.input)

    def __len__(self):
        return len(self.trainSet)


class Corpus():
    """Wrapper representing a Corpus of a perspective (train set or test set).
    A Corpus consists of two partial corpora (PartialCorpus): one for topic
    words and one for opinion words. This class is used by the Perspective
    class.
    """
    def __init__(self, input):
        self.topicCorpus = PartialCorpus(input, lineNumber=0)
        self.opinionCorpus = PartialCorpus(input, lineNumber=1)

    def __iter__(self):
        # topic_words and opinion_words are lists of actual words
        for topic_words, opinion_words in izip(self.topicCorpus.get_texts(),
                                               self.opinionCorpus.get_texts()):
            yield {'topic': topic_words, 'opinion': opinion_words}

    def __len__(self):
        return len(self.topicCorpus)


class PartialCorpus(corpora.TextCorpus):
    """Gensim TextCorpus containing either topic or opinion words.
    Used by the Corpus class.
    """
    def __init__(self, input, lineNumber=0):
        self.lineNumber = lineNumber
        self.maxDocLength = 0
        input.sort()
        super(PartialCorpus, self).__init__(input)

        self.input = input

    def get_texts(self):
        for txt in self.input:
            with codecs.open(txt, 'rb', 'utf8') as f:
                lines = f.readlines()
                words = []
                if len(lines) >= (self.lineNumber+1):
                        words = lines[self.lineNumber].split()

                        # keep track of the maximum document length
                        if len(words) > self.maxDocLength:
                            self.maxDocLength = len(words)
                yield words

    def __len__(self):
        return len(self.input)


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    #files = glob.glob('/home/jvdzwaan/data/dilipad/generated/p*')
    #files = glob.glob('/home/jvdzwaan/data/dilipad/perspectives/*')
    files = glob.glob('/home/jvdzwaan/data/tmp/dilipad/gov_opp/*')
    files.sort()
    #print '\n'.join(files)

    corpus = CPTCorpus(files, testSplit=20)
    #print len(corpus.perspectives[0].opinionCorpus)
    #print len(corpus.perspectives[0].opinionTestCorpus)
    for d in corpus.testSet():
        print d
    #corpus = CPTCorpus(files, topicDict='/home/jvdzwaan/data/dilipad/dictionaries/topicDict.dict',
    #                   opinionDict='/home/jvdzwaan/data/dilipad/dictionaries/opinionDict.dict')
    #corpus.filter_dictionaries(minFreq=5, removeTopTF=100, removeTopDF=100)
    #d = '/home/jvdzwaan/data/dilipad/dictionaries'
    #corpus.save_dictionaries(directory=d)
    #corpus.save_dictionaries(None)
    #corpus.load_dictionaries(topic_dict=corpus.topic_dict_file_name(d),
    #                         opinion_dict=corpus.opinion_dict_file_name(d))
