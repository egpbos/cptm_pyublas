"""Class to access CPT corpus."""

import gensim
import glob
import codecs
from itertools import izip


class CorpusPerspective(gensim.corpora.TextCorpus):
    def __init__(self, input, lineNumber=0):
        self.lineNumber = lineNumber
        super(CorpusPerspective, self).__init__(input)

    def get_texts(self):
        for txt in self.input:
            with codecs.open(txt, 'rb', 'utf8') as f:
                lines = f.readlines()
                words = []
                if len(lines) >= (self.lineNumber+1):
                        words = lines[self.lineNumber].split()
                        #print words
                yield words

    def __len__(self):
        return super(CorpusPerspective, self).__len__()


class CPTCorpus():
    def __init__(self, input):
        # TODO: make sure there is a single dictionary for the topic part
        # (when there are multiple perspectives)
        self.topicCorpus = CorpusPerspective(input, 0)
        self.opinionCorpus = CorpusPerspective(input, 1)
        self.input = input

    def __iter__(self):
        for topic_words, opinion_words in izip(self.topicCorpus,
                                               self.opinionCorpus):
            yield {'topic': topic_words, 'opinion': opinion_words}

    def __len__(self):
        return len(self.input)

if __name__ == '__main__':
    files = glob.glob('/home/jvdzwaan/data/dilipad/generated/*.txt')
    files.sort()

    corpus = CPTCorpus(files)
    print len(corpus)
    print len(corpus.topicCorpus)
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
    for k, v in corpus.topicCorpus.dictionary.iteritems():
        print k, v
    print '\nopinion words'
    for k, v in corpus.opinionCorpus.dictionary.iteritems():
        print k, v
    #b = corpus.dictionary.keys()
    #b.sort()
    #print b
    #print corpus.dictionary.get(0)
