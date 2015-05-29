"""Do CPT using Gibbs sampling.

Uses Gensim fucntionality.

Papers:
- Finding scientific topics
- A Theoretical and Practical Implementation Tutorial on Topic Modeling and
Gibbs Sampling
- Mining contrastive opinions on political texts using cross-perspective topic
model
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
    def __init__(self, corpus, nTopics=10, alpha=0.02, beta=0.02, beta_o=0.02,
                 nIter=2):
        self.corpus = corpus
        self.nTopics = nTopics
        self.alpha = alpha
        self.beta = beta
        self.beta_o = beta_o
        self.nIter = nIter
        self.maxDocLengthT = max([sum([f for w, f in doc]) for doc in corpus.topicCorpus])
        self.maxDocLengthO = max([sum([f for w, f in doc]) for doc in corpus.opinionCorpus])

        #self._initialize()

    def _initialize(self):
        """Initializes the Gibbs sampler."""
        self.VT = len(self.corpus.topicCorpus.dictionary)
        self.VO = len(self.corpus.opinionCorpus.dictionary)
        self.D = len(self.corpus)

        # topics
        self.z = np.zeros((self.D, self.maxDocLengthT), dtype=np.int)
        self.ndk = np.zeros((self.D, self.nTopics), dtype=np.int)
        self.nkw = np.zeros((self.nTopics, self.VT), dtype=np.int)
        self.nk = np.zeros(self.nTopics, dtype=np.int)
        self.ntd = np.zeros(self.D, dtype=np.float)

        for d, w_id, i in self._words_in_corpus(self.corpus.topicCorpus):
            topic = np.random.randint(0, self.nTopics)
            self.z[d, i] = topic
            self.ndk[d, topic] += 1
            self.nkw[topic, w_id] += 1
            self.nk[topic] += 1
            self.ntd[d] += 1

        # opinions
        self.x = np.zeros((self.D, self.maxDocLengthO), dtype=np.int)
        self.nsd = np.zeros((self.D, self.nTopics), dtype=np.int)
        self.nrs = np.zeros((self.nTopics, self.VO), dtype=np.int)
        self.ns = np.zeros(self.nTopics, dtype=np.int)

        for d, w_id, i in self._words_in_corpus(self.corpus.opinionCorpus):
            opinion = np.random.randint(0, self.nTopics)
            self.x[d, i] = opinion
            self.nsd[d, opinion] += 1
            self.nrs[opinion, w_id] += 1
            self.ns[opinion] += 1

        logger.debug('Finished initialization.')

    def _words_in_corpus(self, corpus):
        """Iterates over the words in  the corpus."""
        for d, doc in enumerate(corpus):
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
             (self.nk+self.beta*self.VT)

        p = f1*f2
        return p / np.sum(p)

    def p_x(self, d, w_id):
        f1 = (self.nrs[:, w_id]+self.beta_o)/(self.ns+self.beta_o*self.VO)
        f2 = self.nsd[d]/self.ntd[d]

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
        f2 = np.sum(self.nkw, axis=1, keepdims=True)+self.VT*self.beta
        return f1/f2

    def run(self):
        theta = np.zeros((self.nIter, self.D, self.nTopics))
        phi = np.zeros((self.nIter, self.nTopics, self.VT))
        # TODO: initialize opinion phi

        for t in range(self.nIter):
            t1 = time.clock()
            logger.debug('Iteration {} of {}'.format(t+1, self.nIter))

            # topics
            for d, w_id, i in self._words_in_corpus(self.corpus.topicCorpus):
                topic = self.z[d, i]

                self.ndk[d, topic] -= 1
                self.nkw[topic, w_id] -= 1
                self.nk[topic] -= 1

                p = self.p_z(d, w_id)
                topic = self.sample_from(p)

                self.z[d, i] = topic
                self.ndk[d, topic] += 1
                self.nkw[topic, w_id] += 1
                self.nk[topic] += 1

            # opinions
            for d, w_id, i in self._words_in_corpus(self.corpus.opinionCorpus):
                opinion = self.x[d, i]

                self.nsd[d, opinion] -= 1 # TODO: this line is missing from the notebook
                self.nrs[opinion, w_id] -= 1
                self.ns[opinion] -= 1

                p = self.p_x(d, w_id)
                opinion = self.sample_from(p)

                self.x[d, i] = opinion
                self.nsd[d, opinion] += 1 # TODO: this line is missing from the notebook
                self.nrs[opinion, w_id] += 1
                self.ns[opinion] += 1

            # calculate theta and phi
            theta[t] = self.theta()
            phi[t] = self.phi()
            # TODO: sample opinion phi

            t2 = time.clock()
            logger.debug('time elapsed: {}'.format(t2-t1))
        for t in np.mean(phi, axis=0):
            self.print_topic(t)

    def print_topic(self, weights):
        """Prints the top 10 words in the topics found."""
        words = [self.corpus.topicCorpus.dictionary.get(i)
                 for i in range(self.VT)]
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
