"""Class to access dilipad corpus."""

import gensim
import glob
import codecs


class DilipadCorpus(gensim.corpora.TextCorpus):
    def get_texts(self):
        for txt in self.input:
            with codecs.open(txt, 'rb', 'utf8') as f:
                words = f.read().split()
                yield words

    def __len__(self):
        return len(self.input)


if __name__ == '__main__':
    files = glob.glob('/home/jvdzwaan/data/dilipad/txt-sample/*.txt')

    corpus = DilipadCorpus(files)
    print corpus.dictionary
    #for doc in corpus:
    #    for w in doc:
    #        print w
    print len(corpus.dictionary)
    a = [sum([f for w, f in doc]) for doc in corpus]
    print len(a)
    print sorted(a)
    print max(a)

    #for k, v in corpus.dictionary.iteritems():
    #    print k, v
    b = corpus.dictionary.keys()
    b.sort()
    #print b
    print corpus.dictionary.get(0)
 
