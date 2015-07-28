import logging
import glob
from multiprocessing import Pool

from CPTCorpus import CPTCorpus
from CPT_Gibbs import GibbsSampler


def run_sampler(corpus, nTopics, nIter, beta, out_dir):
    alpha = 50.0/nTopics
    logger.info('running Gibbs sampler (nTopics: {}, nIter: {}, alpha: {}, '
                'beta: {})'.format(nTopics, nIter, alpha, beta))
    sampler = GibbsSampler(corpus, nTopics=nTopics, nIter=nIter,
                           alpha=alpha, beta=beta, beta_o=beta,
                           out_dir=out_dir.format(nTopics))
    sampler._initialize()
    sampler.run()


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
#logger.setLevel(logging.INFO)

files = glob.glob('/home/jvdzwaan/data/dilipad/20112012/gov_opp/*')

out_dir = '/home/jvdzwaan/data/dilipad/res_20112012/{}'

corpus = CPTCorpus(files, testSplit=20)
corpus.filter_dictionaries(minFreq=5, removeTopTF=100, removeTopDF=100)
corpus.save_dictionaries(directory=out_dir.format(''))
corpus.save(out_dir.format('corpus.json'))

#corpus = CPTCorpus.load(out_dir.format('corpus.json'),
#                        topicDict=out_dir.format('topicDict.dict'),
#                        opinionDict=out_dir.format('opinionDict.dict'))

nIter = 200
beta = 0.02

nTopics = range(20, 201, 20)
logger.info('running Gibbs sampler for {} configurations'.format(len(nTopics)))

pool = Pool(processes=3)
results = [pool.apply_async(run_sampler, args=(corpus, n, nIter, beta,
                                               out_dir))
           for n in nTopics]
pool.close()
pool.join()
