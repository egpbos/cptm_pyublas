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
import pandas as pd


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


class GibbsSampler():
    def __init__(self, corpus, nTopics=10, alpha=0.02, beta=0.02, beta_o=0.02,
                 nIter=2):
        self.corpus = corpus
        self.nTopics = nTopics
        self.alpha = alpha
        self.beta = beta
        self.numPerspectives = len(self.corpus.perspectives)
        self.beta_o = beta_o
        self.nIter = nIter

        #self._initialize()

    def _initialize(self):
        """Initializes the Gibbs sampler."""
        self.VT = len(self.corpus.topicDictionary)
        self.VO = len(self.corpus.opinionDictionary)
        self.DT = len(self.corpus)
        self.DO = [len(p.opinionCorpus) for p in self.corpus.perspectives]
        self.maxDocLengthT = max([p.topicCorpus.maxDocLength
                                 for p in self.corpus.perspectives])
        self.maxDocLengthO = [p.opinionCorpus.maxDocLength
                              for p in self.corpus.perspectives]

        # topics
        self.z = np.zeros((self.DT, self.maxDocLengthT), dtype=np.int)
        self.ndk = np.zeros((self.DT, self.nTopics), dtype=np.int)
        self.nkw = np.zeros((self.nTopics, self.VT), dtype=np.int)
        self.nk = np.zeros(self.nTopics, dtype=np.int)
        self.ntd = np.zeros(self.DT, dtype=np.float)

        # opinions
        self.x = [np.zeros((self.DO[i], self.maxDocLengthO[i]), dtype=np.int)
                  for i, p in enumerate(self.corpus.perspectives)]
        self.nrs = [np.zeros((self.nTopics, self.VO), dtype=np.int)
                    for p in self.corpus.perspectives]
        self.ns = [np.zeros(self.nTopics, dtype=np.int)
                   for p in self.corpus.perspectives]

        # loop over the words in the corpus
        for d, persp, d_p, doc in self.corpus:
            for w_id, i in self.corpus.words_in_document(doc, 'topic'):
                topic = np.random.randint(0, self.nTopics)
                self.z[d, i] = topic
                self.ndk[d, topic] += 1
                self.nkw[topic, w_id] += 1
                self.nk[topic] += 1
                self.ntd[d] += 1

            for w_id, i in self.corpus.words_in_document(doc, 'opinion'):
                opinion = np.random.randint(0, self.nTopics)
                self.x[persp][d_p, i] = opinion
                self.nrs[persp][opinion, w_id] += 1
                self.ns[persp][opinion] += 1
        logger.debug('Finished initialization.')

    def p_z(self, d, w_id):
        """Calculate (normalized) probabilities for p(w|z) (topics).

        The probabilities are normalized, because that makes it easier to
        sample from them.
        """
        f1 = (self.ndk[d]+self.alpha) / \
             (np.sum(self.ndk[d])+self.nTopics*self.alpha)
        f2 = (self.nkw[:, w_id]+self.beta) / \
             (self.nk+self.beta*self.VT)

        p = f1*f2
        return p / np.sum(p)

    def p_x(self, p, d, w_id):
        """Calculate (normalized) probabilities for p(w|x) (opinions).

        The probabilities are normalized, because that makes it easier to
        sample from them.
        """
        f1 = (self.nrs[p][:, w_id]+self.beta_o) / \
             (self.ns[p]+self.beta_o*self.VO)
        # The paper says f2 = nsd (the number of times topic s occurs in
        # document d) / Ntd (the number of topic words in document d).
        # 's' is used to refer to opinions. However, f2 makes more sense as the
        # fraction of topic words assigned to a topic.
        # Also in test runs of the Gibbs sampler, the topics and opinions might
        # have different indexes when the number of opinion words per document
        # is used instead of the number of topic words.
        f2 = self.ndk[d]/self.ntd[d]

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

    def theta_topic(self):
        """Calculate theta based on the current word/topic assignments.
        """
        f1 = self.ndk+self.alpha
        f2 = np.sum(self.ndk, axis=1, keepdims=True)+self.nTopics*self.alpha
        return f1/f2

    def phi_topic(self):
        """Calculate phi based on the current word/topic assignments.
        """
        f1 = self.nkw+self.beta
        f2 = np.sum(self.nkw, axis=1, keepdims=True)+self.VT*self.beta
        return f1/f2

    def phi_opinion(self, p):
        """Calculate phi based on the current word/topic assignments.
        """
        f1 = self.nrs[p]+float(self.beta_o)
        f2 = np.sum(self.nrs[p], axis=1, keepdims=True)+self.VO*self.beta_o
        return f1/f2

    def run(self):
        theta_topic = np.zeros((self.nIter, self.DT, self.nTopics))
        phi_topic = np.zeros((self.nIter, self.nTopics, self.VT))

        phi_opinion = [np.zeros((self.nIter, self.nTopics, self.VO))
                       for p in self.corpus.perspectives]

        for t in range(self.nIter):
            t1 = time.clock()
            logger.debug('Iteration {} of {}'.format(t+1, self.nIter))

            for d, persp, d_p, doc in self.corpus:
                for w_id, i in self.corpus.words_in_document(doc, 'topic'):
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

                for w_id, i in self.corpus.words_in_document(doc, 'opinion'):
                    opinion = self.x[persp][d_p, i]

                    self.nrs[persp][opinion, w_id] -= 1
                    self.ns[persp][opinion] -= 1

                    p = self.p_x(persp, d, w_id)
                    opinion = self.sample_from(p)

                    self.x[persp][d_p, i] = opinion
                    self.nrs[persp][opinion, w_id] += 1
                    self.ns[persp][opinion] += 1

            # calculate theta and phi
            theta_topic[t] = self.theta_topic()
            phi_topic[t] = self.phi_topic()

            for p in range(self.numPerspectives):
                phi_opinion[p][t] = self.phi_opinion(p)

            t2 = time.clock()
            logger.debug('time elapsed: {}'.format(t2-t1))
        print 'Topics:'
        for t in np.mean(phi_topic, axis=0):
            self.print_topic(t)
        print
        print 'Opinions:'
        for p in range(self.numPerspectives):
            print 'Perspective:', self.corpus.perspectives[p].name
            for t in np.mean(phi_opinion[p], axis=0):
                self.print_opinion(t)
            print

        self.topics = self.to_df(phi_topic, self.corpus.topicDictionary,
                                 self.VT)
        self.opinions = [self.to_df(phi_opinion[p],
                                    self.corpus.opinionDictionary,
                                    self.VO)
                         for p in range(self.numPerspectives)]
        print self.topics
        for p in range(self.numPerspectives):
            print self.corpus.perspectives[p].name
            print self.opinions[p]

    def print_topic(self, weights):
        """Prints the top 10 words in the topics found."""
        words = [self.corpus.topicDictionary.get(i)
                 for i in range(self.VT)]
        l = zip(words, weights)
        l.sort(key=lambda tup: tup[1])
        print l[:len(l)-11:-1]

    def print_opinion(self, weights):
        """Prints the top 10 words in the topics found."""
        words = [self.corpus.opinionDictionary.get(i)
                 for i in range(self.VO)]
        l = zip(words, weights)
        l.sort(key=lambda tup: tup[1])
        print l[:len(l)-11:-1]

    def to_df(self, phi, dictionary, vocabulary):
        words = [dictionary.get(i) for i in range(vocabulary)]
        data = np.mean(phi, axis=0)
        df = pd.DataFrame(data, columns=words)
        return df.transpose()

    def topics_and_opinions_to_csv(self):
        # TODO: allow user to specify the path where the files are saved
        # TODO: fix case when self.topics and/or self.opinions do not exist
        self.topics.to_csv('topics.csv')
        for p in range(self.numPerspectives):
            p_name = self.corpus.perspectives[p].name.rsplit('/', 1)[1]
            f_name = 'opinions_{}.csv'.format(p_name)
            self.opinions[p].to_csv(f_name)


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)

    files = glob.glob('/home/jvdzwaan/data/dilipad/generated/*')
    #files = glob.glob('/home/jvdzwaan/data/dilipad/perspectives/*')

    corpus = CPTCorpus.CPTCorpus(files)
    sampler = GibbsSampler(corpus, nTopics=3, nIter=2)
    sampler._initialize()
    sampler.run()
    #sampler.print_topics_and_opinions()
    sampler.topics_and_opinions_to_csv()
