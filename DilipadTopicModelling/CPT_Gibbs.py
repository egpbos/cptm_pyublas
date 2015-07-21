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
import os
from scipy.stats import entropy

from gibbs_inner import gibbs_inner


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


class GibbsSampler():
    PARAMETER_DIR = '{}/parameter_samples'
    PHI_TOPIC = 'phi_topic'
    PHI_OPINION = 'phi_opinion_{}'
    THETA = 'theta'
    NKS = 'nks'

    def __init__(self, corpus, nTopics=10, alpha=0.02, beta=0.02, beta_o=0.02,
                 nIter=2, out_dir=None):
        self.corpus = corpus
        self.nTopics = nTopics
        self.alpha = alpha
        self.beta = beta
        self.nPerspectives = len(self.corpus.perspectives)
        self.beta_o = beta_o
        self.nIter = nIter

        self.out_dir = out_dir
        if self.out_dir:
            if not os.path.exists(self.out_dir):
                os.makedirs(out_dir)
        parameter_dir = self.get_parameter_dir_name()
        if not os.path.exists(parameter_dir):
            os.makedirs(parameter_dir)

        #self._initialize()

    def _initialize(self):
        """Initializes the Gibbs sampler."""
        self.VT = len(self.corpus.topicDictionary)
        self.VO = len(self.corpus.opinionDictionary)
        self.DT = len(self.corpus)
        self.DO = max([len(p.trainSet.opinionCorpus)
                       for p in self.corpus.perspectives])
        self.maxDocLengthT = max([p.trainSet.topicCorpus.maxDocLength
                                 for p in self.corpus.perspectives])
        self.maxDocLengthO = max([p.trainSet.opinionCorpus.maxDocLength
                                  for p in self.corpus.perspectives])

        # topics
        self.z = np.zeros((self.DT, self.maxDocLengthT), dtype=np.int)
        self.ndk = np.zeros((self.DT, self.nTopics), dtype=np.int)
        self.nkw = np.zeros((self.nTopics, self.VT), dtype=np.int)
        self.nk = np.zeros(self.nTopics, dtype=np.int)
        self.ntd = np.zeros(self.DT, dtype=np.int)

        # opinions
        self.x = np.zeros((self.nPerspectives, self.DO, self.maxDocLengthO),
                          dtype=np.int)
        self.nrs = np.zeros((self.nPerspectives, self.nTopics, self.VO),
                            dtype=np.int)
        self.ns = np.zeros((self.nPerspectives, self.nTopics), dtype=np.int)

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
                self.nrs[persp, opinion, w_id] += 1
                self.ns[persp, opinion] += 1
        logger.debug('Finished initialization.')

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

    def phi_opinion(self, persp):
        """Calculate phi based on the current word/topic assignments.
        """
        f1 = self.nrs[persp]+float(self.beta_o)
        f2 = np.sum(self.nrs[persp], axis=1, keepdims=True)+self.VO*self.beta_o
        return f1/f2

    def run(self):
        if not self.out_dir:
            # store all parameter samples in memory
            theta_topic = np.zeros((self.nIter, self.DT, self.nTopics))
            phi_topic = np.zeros((self.nIter, self.nTopics, self.VT))

            phi_opinion = [np.zeros((self.nIter, self.nTopics, self.VO))
                           for p in self.corpus.perspectives]

        # variable to store nk for each iteration
        # nk is used for Contrastive Opinion Modeling
        self.nks = np.zeros((self.nIter, self.nTopics), dtype=np.int)

        for t in range(self.nIter):
            t1 = time.clock()
            logger.debug('Iteration {} of {}'.format(t+1, self.nIter))

            gibbs_inner(self)

            # calculate theta and phi
            if not self.out_dir:
                theta_topic[t] = self.theta_topic()
                phi_topic[t] = self.phi_topic()

                for p in range(self.nPerspectives):
                    phi_opinion[p][t] = self.phi_opinion(p)
            else:
                pd.DataFrame(self.theta_topic()). \
                    to_csv(self.get_theta_file_name(t))
                pd.DataFrame(self.phi_topic()). \
                    to_csv(self.get_phi_topic_file_name(t))
                for p in range(self.nPerspectives):
                    pd.DataFrame(self.phi_opinion(p)). \
                        to_csv(self.get_phi_opinion_file_name(p, t))

                # save nk (for Contrastive Opinion Mining)
                self.nks[t] = np.copy(self.nk)
                pd.DataFrame(self.nks).to_csv(self.get_nks_file_name())

            t2 = time.clock()
            logger.debug('time elapsed: {}'.format(t2-t1))

        if not self.out_dir:
            # calculate means of parameters in memory
            phi_topic = np.mean(phi_topic, axis=0)
            theta_topic = np.mean(theta_topic, axis=0)
            for p in range(self.nPerspectives):
                phi_opinion[p] = np.mean(phi_opinion[p], axis=0)
        else:
            # load numbers from files
            theta_topic = self.load_parameters(self.THETA)
            phi_topic = self.load_parameters(self.PHI_TOPIC)
            phi_opinion = {}
            for p in range(self.nPerspectives):
                phi_opinion[p] = self.load_parameters(self.PHI_OPINION.
                                                      format(p))

        self.topics = self.to_df(phi_topic, self.corpus.topic_words())
        self.opinions = [self.to_df(phi_opinion[p],
                                    self.corpus.opinion_words())
                         for p in range(self.nPerspectives)]
        self.document_topic_matrix = self.to_df(theta_topic)

    def load_parameters(self, name, index=None, start=None, end=None):
        if index < 0 or index > self.nIter:
            index = None

        if index:
            return pd.read_csv(self.get_parameter_file_name(name, index),
                               index_col=0).as_matrix()

        if start < 0 or start > self.nIter or start > end:
            start = 0

        if end < 0 or end > self.nIter or end < start:
            end = self.nIter

        data = None
        for i in range(start, end):
            ar = pd.read_csv(self.get_parameter_file_name(name, i),
                             index_col=0).as_matrix()
            if data is None:
                data = np.array([ar])
            else:
                data = np.append(data, np.array([ar]), axis=0)
        return np.mean(data, axis=0)

    def get_phi_topic_file_name(self, number):
        return self.get_parameter_file_name(self.PHI_TOPIC, number)

    def get_phi_opinion_file_name(self, persp, number):
        return self.get_parameter_file_name(self.PHI_OPINION.format(persp),
                                            number)

    def get_theta_file_name(self, number):
        return self.get_parameter_file_name(self.THETA, number)

    def get_nks_file_name(self):
        return '{}/{}.csv'.format(self.get_parameter_dir_name(), self.NKS)

    def get_parameter_file_name(self, name, number):
        return '{}/{}_{:04d}.csv'.format(self.get_parameter_dir_name(),
                                         name, number)

    def get_parameter_dir_name(self):
        return self.PARAMETER_DIR.format(self.out_dir)

    def print_topics_and_opinions(self, top=10):
        """Print topics and associated opinions.

        The <top> top words and weights are printed.
        """
        for i in range(self.nTopics):
            print u'Topic {}: {}'. \
                  format(i, self.print_topic(self.topics.loc[:, i].copy(),
                                             top))
            print
            for p in range(self.nPerspectives):
                print u'Opinion {}: {}'. \
                      format(self.corpus.perspectives[p].name,
                             self.print_topic(self.opinions[p].loc[:, i].copy(),
                                              top))
            print '-----'
            print

    def print_topic(self, series, top=10):
        """Prints the top 10 words in the topic/opinion on a single line."""
        s = series.copy()
        s.sort(ascending=False)
        t = [u'{} ({:.4f})'.format(word, p)
             for word, p in s[0:top].iteritems()]
        return u' - '.join(t)

    def to_df(self, data, words=None):
        if not words:
            # theta (topic document matrix)
            df = pd.DataFrame(data)
        else:
            # phi (topics and opinions)
            df = pd.DataFrame(data, columns=words)
            df = df.transpose()
        return df

    def topics_and_opinions_to_csv(self):
        # TODO: fix case when self.topics and/or self.opinions do not exist

        if self.out_dir:
            path = self.out_dir
        else:
            path = ''

        self.topics.to_csv(os.path.join(path, 'topics.csv'), encoding='utf8')
        self.document_topic_matrix.to_csv(os.path.join(path,
                                                       'document-topic.csv'))
        for p in range(self.nPerspectives):
            p_name = self.corpus.perspectives[p].name
            f_name = 'opinions_{}.csv'.format(p_name)
            self.opinions[p].to_csv(os.path.join(path, f_name),
                                    encoding='utf8')

    def contrastive_opinions(self, query):
        """Returns a DataFrame containing contrastive opinions for the query.

        Implements contrastive opinion modeling as specified in [Fang et al.,
        2012] equation 1. The resulting probability distributions over words
        are normalized, in order to facilitate mutual comparisons.

        Example usage:
            co = sampler.contrastive_opinions('mishandeling')
            print sampler.print_topic(co[0])

        Parameters:
            query : str
            The word contrastive opinions should be calculated for.

        Returns:
            pandas DataFrame
            The index of the DataFrame are the topic words and the columns
            represent the perspectives.
        """
        # TODO: allow the user to specify index or range of the parameters to use
        phi_topic = self.load_parameters(self.PHI_TOPIC, index=self.nIter-1)

        self.nks = pd.read_csv(self.get_nks_file_name(), index_col=0).values

        # TODO: fix case when word not in topicDictionary
        query_word_id = self.corpus.topicDictionary.doc2bow([query])[0][0]
        print query_word_id

        words = self.corpus.opinion_words()
        result = pd.DataFrame(np.zeros((self.VO, self.nPerspectives)), words)

        for p in range(self.nPerspectives):
            phi_opinion = self.load_parameters(self.PHI_OPINION.format(p),
                                               index=self.nIter-1)

            c_opinion = phi_opinion.transpose() * phi_topic[:, query_word_id] * self.nks[-1]
            c_opinion = np.sum(c_opinion, axis=1)
            c_opinion /= np.sum(c_opinion)

            result[p] = pd.Series(c_opinion, index=words)

        return result

    def jsd_opinions(self, co_df):
        """Calculate Jensen-Shannon divergence between contrastive opinions.

        Implements Jensen-Shannon divergence between contrastive opinions as
        described in [Fang et al., 2012] section 3.2.

        Example usage:
            co = sampler.contrastive_opinions('mishandeling')
            jsd =  sampler.jsd_opinions(co)

        Parameter:
            co_df : pandas DataFrame
            A pandas DataFrame containing contrastive opinions (see
            self.contrastive_opinions(query))

        Returns:
            float
            The Jensen-Shannon divergence between the contrastive opinions.

        """
        co = co_df.values
        result = np.zeros(self.nPerspectives, dtype=np.float)
        p_avg = np.mean(co, axis=1)
        for persp in range(self.nPerspectives):
            result[persp] = entropy(co[:, persp], p_avg)
        return np.mean(result)


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)

    files = glob.glob('/home/jvdzwaan/data/tmp/dilipad/gov_opp/*')
    #files = glob.glob('/home/jvdzwaan/data/dilipad/perspectives/*')

    corpus = CPTCorpus.CPTCorpus(files)
    corpus.filter_dictionaries(minFreq=5, removeTopTF=100, removeTopDF=100)
    sampler = GibbsSampler(corpus, nTopics=100, nIter=2, out_dir='/home/jvdzwaan/data/tmp/dilipad/test_parameters')
    #sampler = GibbsSampler(corpus, nTopics=100, nIter=2)
    sampler._initialize()
    #sampler.run()
    #sampler.print_topics_and_opinions()
    #sampler.topics_and_opinions_to_csv()
    co = sampler.contrastive_opinions('mishandeling')
    #print co
    print sampler.print_topic(co[0])
    print sampler.print_topic(co[1])
    print 'Jensen-Shannon divergence:', sampler.jsd_opinions(co)
    #sampler.parameter_dir = '/home/jvdzwaan/data/tmp/dilipad/test_parameters/parameter_samples/'
    #theta_topic = sampler.load_parameters('theta')
    #phi_topic = sampler.load_parameters('phi_topic')
    #phi_opinion = {}
    #for p in range(sampler.nPerspectives):
    #        phi_opinion[p] = sampler.load_parameters('phi_opinion_{}'.format(p))
    #print theta_topic
    #print phi_topic
   # print phi_opinion
