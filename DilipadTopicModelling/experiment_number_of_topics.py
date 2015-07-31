import logging
import glob
from multiprocessing import Pool
import argparse
import json

from CPTCorpus import CPTCorpus
from CPT_Gibbs import GibbsSampler


def run_sampler(corpus, nTopics, nIter, beta, beta_o, out_dir):
    alpha = 50.0/nTopics
    logger.info('running Gibbs sampler (nTopics: {}, nIter: {}, alpha: {}, '
                'beta: {}, beta_o: {})'.format(nTopics, nIter, alpha, beta,
                                               beta_o))
    sampler = GibbsSampler(corpus, nTopics=nTopics, nIter=nIter,
                           alpha=alpha, beta=beta, beta_o=beta_o,
                           out_dir=out_dir.format(nTopics))
    sampler._initialize()
    sampler.run()


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('json', help='json file containing experiment '
                    'configuration.')
args = parser.parse_args()

with open(args.json) as f:
    config = json.load(f)

logger.debug('configuration of experiment: ')
params = ['{}: {}'.format(p, v) for p, v in config.iteritems()]
for p in params:
    logger.debug(p)

files = glob.glob(config.get('input_data'))

out_dir = config.get('out_dir', '/{}')
testSplit = config.get('testSplit', 20)
minFreq = config.get('minFreq', 5)
removeTopTF = config.get('removeTopTF', 100)
removeTopDF = config.get('removeTopDF', 100)
nIter = config.get('nIter', 200)
beta = config.get('beta', 0.02)
beta_o = config.get('beta_o', 0.02)
nTopics = config.get('nTopics', range(20, 201, 20))
nProcesses = config.get('nProcesses', None)
loadDictionaries = config.get('loadDictionaries', 0)

if not loadDictionaries:
    corpus = CPTCorpus(files, testSplit=testSplit)
    corpus.filter_dictionaries(minFreq=minFreq, removeTopTF=removeTopTF,
                               removeTopDF=removeTopDF)
    corpus.save_dictionaries(directory=out_dir.format(''))
    corpus.save(out_dir.format('corpus.json'))
else:
    corpus = CPTCorpus.load(out_dir.format('corpus.json'),
                            topicDict=out_dir.format('topicDict.dict'),
                            opinionDict=out_dir.format('opinionDict.dict'))

logger.info('running Gibbs sampler for {} configurations'.format(len(nTopics)))

pool = Pool(processes=nProcesses)
results = [pool.apply_async(run_sampler, args=(corpus, n, nIter, beta, beta_o,
                                               out_dir))
           for n in nTopics]
pool.close()
pool.join()
