import pandas as pd
import logging
from multiprocessing import Pool

from CPTCorpus import CPTCorpus
from CPT_Gibbs import GibbsSampler


def calculate_perplexity(corpus, nTopics, nIter, beta, out_dir, nPerplexity):
    alpha = 50.0/nTopics
    logger.info('running Gibbs sampler (nTopics: {}, nIter: {}, alpha: {}, '
                'beta: {})'.format(nTopics, nIter, alpha, beta))
    sampler = GibbsSampler(corpus, nTopics=nTopics, nIter=nIter,
                           alpha=alpha, beta=beta, beta_o=beta,
                           out_dir=out_dir.format(nTopics))
    sampler._initialize()
    #sampler.run()
    results = []
    for s in nPerplexity:
        tw_perp, ow_perp = sampler.perplexity(index=s)
        results.append((nTopics, s, tw_perp, ow_perp))
    return results


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

pool = Pool(processes=4)
results = [pool.apply_async(calculate_perplexity, args=(corpus, n, nIter, beta,
                                                        out_dir, nPerplexity))
           for n in nTopics]
pool.close()
pool.join()

data = [p.get() for p in results]

for result in data:
    for n, s, tw_perp, ow_perp in result:
        topic_perp.set_value(s, n, tw_perp)
        opinion_perp.set_value(s, n, ow_perp)

        logger.info('nTopics: {}, nPerplexity: {}, topic perplexity: {}, '
                    'opinion perplexity: {}'.format(n, s, tw_perp, ow_perp))

topic_perp.to_csv(out_dir.format('perplexity_topic.csv'))
opinion_perp.to_csv(out_dir.format('perplexity_opinion.csv'))
