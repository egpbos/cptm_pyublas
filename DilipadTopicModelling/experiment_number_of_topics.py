import logging
import glob
from multiprocessing import Process

from CPTCorpus import CPTCorpus
from CPT_Gibbs import GibbsSampler


def run_sampler(corpus, nTopics, nIter, beta, out_dir):
    sampler = GibbsSampler(corpus, nTopics=nTopics, nIter=nIter,
                           alpha=(50.0/n), beta=beta, beta_o=beta,
                           out_dir=out_dir.format(nTopics))
    sampler._initialize()
    sampler.run()


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
#logger.setLevel(logging.INFO)

#files = glob.glob('/home/jvdzwaan/data/tmp/dilipad/gov_opp/*')
files = glob.glob('/home/jvdzwaan/data/tmp/test/*')

corpus_dir = '/home/jvdzwaan/data/tmp/generated/test_exp/corpus'
out_dir = '/home/jvdzwaan/data/tmp/generated/test_exp/{}'
#out_dir = '/home/jvdzwaan/data/tmp/dilipad/test_perplexity/'

corpus = CPTCorpus(files, testSplit=20)
corpus.filter_dictionaries(minFreq=5, removeTopTF=100, removeTopDF=100)
corpus.save_dictionaries(directory=out_dir.format(''))
corpus.save(out_dir.format('corpus.json'))

#corpus = CPTCorpus.CPTCorpus.load('{}corpus.json'.format(out_dir),
#                                  topicDict='{}/topicDict.dict'.format(out_dir),
#                                  opinionDict='{}/opinionDict.dict'.format(out_dir))

nIter = 200
beta = 0.02

nTopics = range(20, 201, 20)
logger.info('running Gibbs sampler for {} configurations'.format(len(nTopics)))

processes = [Process(target=run_sampler,
                     args=(corpus, n, nIter, beta, out_dir))
             for n in nTopics]

# Run processes
for p in processes:
    p.start()

# Exit the completed processes
for p in processes:
    p.join()
