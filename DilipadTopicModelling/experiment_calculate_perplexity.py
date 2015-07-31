import pandas as pd
import logging
from multiprocessing import Pool, cpu_count

from CPTCorpus import CPTCorpus
from CPT_Gibbs import GibbsSampler


def calculate_perplexity(nTopics, nIter, beta, out_dir, nPerplexity):
    try:
        logger.info('started perplexity calculation for {} topics'.format(nTopics))
        data_dir = '/scratch/users/jvdzwaan/data/exp_num_topics/'
        corpus = CPTCorpus.load('{}corpus.json'.format(data_dir),
                                topicDict='{}/topicDict.dict'.format(data_dir),
                                opinionDict='{}/opinionDict.dict'.format(data_dir))
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
            logger.info('doing perplexity calculation ({}, {})'.format(nTopics, s))
            tw_perp, ow_perp = sampler.perplexity(index=s)
            #tw_perp, ow_perp = 2.34639912, 9.26547634 
            results.append((nTopics, s, tw_perp, ow_perp))
        logger.info('finished perplexity calculation for {} topics'.
                    format(nTopics))
        return results
    except Exception as e:
	logger.info('error')
	logger.info(str(e))
	return None


logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)

logging.getLogger('gensim').setLevel(logging.ERROR)
logging.getLogger('CPTCorpus').setLevel(logging.ERROR)
logging.getLogger('CPT_Gibbs').setLevel(logging.ERROR)

print '# CPUs', cpu_count() 

# load corpus
#data_dir = '/scratch/users/jvdzwaan/data/exp_num_topics/'
#corpus = CPTCorpus.load('{}corpus.json'.format(data_dir))
#corpus = CPTCorpus.load('{}corpus.json'.format(data_dir),
#                        topicDict='{}/topicDict.dict'.format(data_dir),
#                        opinionDict='{}/opinionDict.dict'.format(data_dir))

nIter = 200
beta = 0.02
out_dir = '/scratch/users/jvdzwaan/data/exp_num_topics/{}'

nTopics = range(20, nIter+1, 20)
nPerplexity = range(0, nIter+1, 10)

pool = Pool(processes=1)
results = [pool.apply_async(calculate_perplexity, args=(n, nIter, beta,
                            out_dir, nPerplexity))
           for n in nTopics[::-1]]
logger.info('finished initialisation')
pool.close()
logger.info('called pool.close()')
pool.join()
logger.info('called pool.join()')

data = [p.get() for p in results]
logger.info('created data')

topic_perp = pd.DataFrame(columns=nTopics, index=nPerplexity)
opinion_perp = pd.DataFrame(columns=nTopics, index=nPerplexity)

for result in data:
    if result is None:
        logger.error('None result')
    else:
        for n, s, tw_perp, ow_perp in result:
            topic_perp.set_value(s, n, tw_perp)
            opinion_perp.set_value(s, n, ow_perp)

            #logger.info('nTopics: {}, nPerplexity: {}, topic perplexity: {}, '
            #            'opinion perplexity: {}'.format(n, s, tw_perp, ow_perp))

topic_perp.to_csv(out_dir.format('perplexity_topic.csv'))
opinion_perp.to_csv(out_dir.format('perplexity_opinion.csv'))
