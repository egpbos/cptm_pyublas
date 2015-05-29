"""Do LDA using Gibbs sampling.

Uses Gensim fucntionality.

Papers:
- Finding scientific topics
- A Theoretical and Practical Implementation Tutorial on Topic Modeling and
Gibbs Sampling


"""

from __future__ import division
import numpy as np
import CPTCorpus
import glob
import logging
import time


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


class GibbsSampler():
    def __init__(self, corpus, nTopics=10, alpha=0.02, beta=0.02, nIter=2):
        self.corpus = corpus
        self.nTopics = nTopics
        self.alpha = alpha
        self.beta = beta
        self.nIter = nIter
        self.maxDocLength = max([sum([f for w, f in doc]) for doc in corpus])

    def _initialize(self):
        """Initializes the Gibbs sampler."""
        self.V = len(self.corpus.dictionary)
        self.D = len(self.corpus)

        self.z = np.zeros((self.D, self.maxDocLength), dtype=np.int)
        self.ndk = np.zeros((self.D, self.nTopics), dtype=np.int)
        self.nkw = np.zeros((self.nTopics, self.V), dtype=np.int)
        self.nk = np.zeros(self.nTopics, dtype=np.int)

        for d, w_id, i in self._words_in_corpus():
            #print i,
            topic = np.random.randint(0, self.nTopics)
            self.z[d, i] = topic
            self.ndk[d, topic] += 1
            self.nkw[topic, w_id] += 1
            self.nk[topic] += 1

            #print
        #print self.z
        #print self.z.shape
        logger.debug('Finished initialization.')

    def _words_in_corpus(self):
        """Iterates over the words in  the corpus."""
        for d, doc in enumerate(self.corpus):
            i = 0
            for w_id, freq in doc:
                for j in range(freq):
                    yield d, w_id, i
                    i += 1

    def p_z(self, d, w_id):
        """Calculate (normalized) probabilities for p(w|z).

        The probabilities are normalized, because that makes it easier to
        sample from them.
        """
        f1 = (self.ndk[d]+self.alpha) / \
             (np.sum(self.ndk[d])+self.nTopics*self.alpha)
        f2 = (self.nkw[:, w_id]+self.beta) / \
             (self.nk+self.beta*len(self.corpus.dictionary))

        p = f1*f2
        return p / np.sum(p)

    def sample_from(self, p):
        """Sample (new) topic from multinomial distribution p.
        Returns a word's the topic index based on p_z.

        The searchsorted method is used instead of
        np.random.multinomial(1,p).argmax(), because despite normalizing the
        probabilities, sometimes the sum of the probabilities > 1.0, which
        causes the multinomial method to crash. This probably has to do with
        machine precision.
        """
        return np.searchsorted(np.cumsum(p), np.random.rand())

    def theta(self):
        """Calculate theta based on the current word/topic assignments.
        """
        f1 = self.ndk+self.alpha
        f2 = np.sum(self.ndk, axis=1, keepdims=True)+self.nTopics*self.alpha
        return f1/f2

    def phi(self):
        """Calculate phi based on the current word/topic assignments.
        """
        f1 = self.nkw+self.beta
        f2 = np.sum(self.nkw, axis=1, keepdims=True)+self.V*self.beta
        return f1/f2

    def run(self):
        theta = np.zeros((self.nIter, self.D, self.nTopics))
        phi = np.zeros((self.nIter, self.nTopics, self.V))

        for t in range(self.nIter):
            t1 = time.clock()
            logger.debug('Iteration {} of {}'.format(t+1, self.nIter))
            for d, w_id, i in self._words_in_corpus():
                #print self.corpus.dictionary[w_id]
                topic = self.z[d, i]
                topic_old = topic

                self.ndk[d, topic] -= 1
                self.nkw[topic, w_id] -= 1
                self.nk[topic] -= 1
                p = self.p_z(d, w_id)
                #print p
                #print type(p[0])

                topic = self.sample_from(p)

                self.z[d, i] = topic
                self.ndk[d][topic] += 1
                self.nkw[topic, w_id] += 1
                self.nk[topic] += 1

                #if topic_old != topic:
                #    logger.debug('Changed topic from {} to {}'.format(topic_old, topic))
            # calculate theta and phi
            theta[t] = self.theta()
            phi[t] = self.phi()

            t2 = time.clock()
            logger.debug('time elapsed: {}'.format(t2-t1))
        for t in np.mean(phi, axis=0):
            self.print_topic(t)

    def print_topic(self, weights):
        """Prints the top 10 words in the topics found."""
        words = [self.corpus.dictionary.get(i)
                 for i in range(len(self.corpus.dictionary))]
        l = zip(words, weights)
        l.sort(key=lambda tup: tup[1])
        print l[:len(l)-11:-1]


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)

    files = glob.glob('/home/jvdzwaan/data/dilipad/generated/*.txt')

    corpus = CPTCorpus.CPTCorpus(files)
    sampler = GibbsSampler(corpus, nTopics=3, nIter=100)
    sampler._initialize()
    sampler.run()
