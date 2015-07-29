import logging
import os
import pandas as pd

from CPTCorpus import CPTCorpus
from CPT_Gibbs import GibbsSampler

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

# select experiment to get parameters from
nTopics = 100
start = 80
end = 199

alpha = 50.0/nTopics
beta = 0.02
nIter = 200

# load corpus
data_dir = '/home/jvdzwaan/data/tmp/generated/test_exp/'
corpus = CPTCorpus.load('{}corpus.json'.format(data_dir),
                        topicDict='{}/topicDict.dict'.format(data_dir),
                        opinionDict='{}/opinionDict.dict'.format(data_dir))

out_dir = '/home/jvdzwaan/data/tmp/generated/test_exp/{}'.format(nTopics)

sampler = GibbsSampler(corpus, nTopics=nTopics, nIter=nIter, alpha=alpha,
                       beta=beta, beta_o=beta, out_dir=out_dir)
sampler._initialize()
sampler.estimate_parameters(start=start, end=end)

pd.DataFrame(sampler.theta).to_csv(os.path.join(out_dir, 'theta_{}.csv'.
                                                         format(nTopics)))
topics = sampler.topics_to_df(phi=sampler.topics, words=corpus.topic_words())
topics.to_csv(os.path.join(out_dir, 'topics_{}.csv'.format(nTopics)),
              encoding='utf8')
for i, p in enumerate(sampler.corpus.perspectives):
    opinions = sampler.topics_to_df(phi=sampler.opinions[i],
                                    words=corpus.opinion_words())
    opinions.to_csv(os.path.join(out_dir,
                                 'opinions_{}_{}.csv'.format(p.name, nTopics)),
                    encoding='utf8')
