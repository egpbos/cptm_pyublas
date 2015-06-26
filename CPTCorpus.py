"""Class to access CPT corpus."""
import logging
import gensim
import glob
import codecs
from itertools import izip


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
    """
    def __init__(self, input):
        logger.info('initialize CPT Corpus with {} perspectives'
                    .format(len(input)))
        self.perspectives = [Perspective(glob.glob('{}/*.txt'.format(d)), d)
                             for d in input]

        # create dictionaries with all topic and opinion words (universal
        # mappings that can be used with the corpora from different
        # perspectives).
        self.topicDictionary = self.perspectives[0].topicCorpus.dictionary
        self.opinionDictionary = self.perspectives[0].opinionCorpus.dictionary
        for p in self.perspectives[1:]:
            self.topicDictionary.add_documents(p.topicCorpus.get_texts(),
                                               prune_at=None)
            self.opinionDictionary.add_documents(p.opinionCorpus.get_texts(),
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
        doc_id_global = 0
        for i, p in enumerate(self.perspectives):
            doc_id_perspective = 0
            for doc in p:
                doc['topic'] = self.topicDictionary.doc2bow(doc['topic'])
                doc['opinion'] = self.opinionDictionary.doc2bow(doc['opinion'])

                yield doc_id_global, i, doc_id_perspective, doc

                doc_id_global += 1
                doc_id_perspective += 1

    def __len__(self):
        return sum([len(p) for p in self.perspectives])


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
    def __init__(self, input, directory):
        name = directory.rsplit('/', 1)[1]
        logger.info('initialize perspective "{}" (path: {} - {} documents)'
                    .format(name, directory, len(input)))
        self.topicCorpus = PerspectiveCorpus(input, 0)
        self.opinionCorpus = PerspectiveCorpus(input, 1)
        self.input = input
        self.name = name
        self.directory = directory

    def __iter__(self):
        # topic_words and opinion_words are lists of actual words
        for topic_words, opinion_words in izip(self.topicCorpus.get_texts(),
                                               self.opinionCorpus.get_texts()):
            yield {'topic': topic_words, 'opinion': opinion_words}

    def __len__(self):
        return len(self.input)


class PerspectiveCorpus(gensim.corpora.TextCorpus):
    """Wrapper for corpus representing a perspective.
    Used by Perspective class.
    """
    def __init__(self, input, lineNumber=0):
        self.lineNumber = lineNumber
        self.maxDocLength = 0
        super(PerspectiveCorpus, self).__init__(input)

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
        return super(PerspectiveCorpus, self).__len__()


if __name__ == '__main__':
    files = glob.glob('/home/jvdzwaan/data/dilipad/generated/p*')
    files.sort()
    print '\n'.join(files)

    corpus = CPTCorpus(files)
    print len(corpus)
    #print corpus.dictionary
    for doc in corpus:
        print doc
        #for w in doc:
        #    print corpus.dictionary[w[0]]
        print '----------'
    #print len(corpus.dictionary)
    #a = [sum([f for w, f in doc]) for doc in corpus]
    #print len(a)
    #print sorted(a)
    #print max(a)

    print 'topic words'
    for k, v in corpus.topicDictionary.iteritems():
        print k, v
    print '\nopinion words'
    for k, v in corpus.opinionDictionary.iteritems():
        print k, v
    #b = corpus.dictionary.keys()
    #b.sort()
    #print b
    #print corpus.dictionary.get(0)
