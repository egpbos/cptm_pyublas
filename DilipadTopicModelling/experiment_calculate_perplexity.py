import pandas as pd
import logging

from CPTCorpus import CPTCorpus
from CPT_Gibbs import GibbsSampler


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

# load corpus
data_dir = '/home/jvdzwaan/data/tmp/generated/test_exp/'
corpus = CPTCorpus.load('{}corpus.json'.format(data_dir))
#corpus = CPTCorpus.load('{}corpus.json'.format(data_dir),
#                        topicDict='{}/topicDict.dict'.format(data_dir),
#                        opinionDict='{}/opinionDict.dict'.format(data_dir))

nIter = 200
beta = 0.02
out_dir = '/home/jvdzwaan/data/tmp/generated/test_exp/{}'

nTopics = range(20, nIter+1, 20)
nPerplexity = range(0, nIter+1, 10)

topic_perp = pd.DataFrame(columns=nTopics, index=nPerplexity)
opinion_perp = pd.DataFrame(columns=nTopics, index=nPerplexity)

for n in nTopics:
    # load sampler
    sampler = GibbsSampler(corpus, nTopics=n, nIter=nIter, alpha=(50.0/n),
                           beta=beta, beta_o=beta,
                           out_dir=out_dir.format(n))
    sampler._initialize()
    sampler.run()

    for s in nPerplexity:
        tw_perp, ow_perp = sampler.perplexity(index=s)

        topic_perp.set_value(s, n, tw_perp)
        opinion_perp.set_value(s, n, ow_perp)

        logger.info('nTopics: {}, nPerplexity: {}, topic perplexity: {}, '
                    'opinion perplexity: {}'.format(n, s, tw_perp, ow_perp))

topic_perp.to_csv(out_dir.format('perplexity_topic.csv'))
opinion_perp.to_csv(out_dir.format('perplexity_opinion.csv'))
