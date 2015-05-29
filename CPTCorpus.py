"""Class to access CPT corpus."""

import gensim
import glob
import codecs


class CPTCorpus(gensim.corpora.TextCorpus):
    def get_texts(self):
        for txt in self.input:
            with codecs.open(txt, 'rb', 'utf8') as f:
                lines = f.readlines()
                topic_words = []
                if len(lines) >= 1:
                        topic_words = lines[0].split()
                        #print topic_words
                yield topic_words

    def __len__(self):
        return len(self.input)


if __name__ == '__main__':
    files = glob.glob('/home/jvdzwaan/data/dilipad/generated/*.txt')
    files.sort()

    corpus = CPTCorpus(files)
    #print corpus.dictionary
    for doc in corpus:
        print doc
        for w in doc:
            print corpus.dictionary[w[0]]
        print '----------'
    #print len(corpus.dictionary)
    #a = [sum([f for w, f in doc]) for doc in corpus]
    #print len(a)
    #print sorted(a)
    #print max(a)

    #for k, v in corpus.dictionary.iteritems():
    #    print k, v
    #b = corpus.dictionary.keys()
    #b.sort()
    #print b
    #print corpus.dictionary.get(0)
