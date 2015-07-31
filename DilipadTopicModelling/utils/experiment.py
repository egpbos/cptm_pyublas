"""Functions for experiments."""
import logging
import json
import glob
import os

from CPTCorpus import CPTCorpus
from CPT_Gibbs import GibbsSampler

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_config(fName):
    with open(fName) as f:
        config = json.load(f)

    logger.debug('configuration of experiment: ')
    params = ['{}: {}'.format(p, v) for p, v in config.iteritems()]
    for p in params:
        logger.debug(p)

    params = {}
    params['inputData'] = config.get('input_data')
    params['outDir'] = config.get('out_dir', '/{}')
    params['testSplit'] = config.get('testSplit', 20)
    params['minFreq'] = config.get('minFreq', 5)
    params['removeTopTF'] = config.get('removeTopTF', 100)
    params['removeTopDF'] = config.get('removeTopDF', 100)
    params['nIter'] = config.get('nIter', 200)
    params['beta'] = config.get('beta', 0.02)
    params['beta_o'] = config.get('beta_o', 0.02)
    params['expNumTopics'] = config.get('expNumTopics', range(20, 201, 20))
    params['nTopics'] = config.get('nTopics')
    params['nProcesses'] = config.get('nProcesses', None)

    return params


def get_corpus(params):
    out_dir = params.get('outDir')
    files = glob.glob(params.get('input_data'))

    if not os.path.isfile(os.path.join(out_dir, 'corpus.json')):
        corpus = CPTCorpus(files,
                           testSplit=params.get('testSplit'))
        corpus.filter_dictionaries(minFreq=params.get('minFreq'),
                                   removeTopTF=params.get('removeTopTF'),
                                   removeTopDF=params.get('removeTopDF'))
        corpus.save_dictionaries(directory=out_dir.format(''))
        corpus.save(out_dir.format('corpus.json'))
    else:
        corpus = CPTCorpus.load(out_dir.format('corpus.json'),
                                topicDict=out_dir.format('topicDict.dict'),
                                opinionDict=out_dir.format('opinionDict.dict'))
    return corpus


def get_sampler(params, corpus, nTopics):
    out_dir = params.get('outDir')
    nIter = params.get('nIter')
    alpha = 50.0/nTopics
    beta = params.get('beta')
    beta_o = params.get('beta_o')
    logger.info('running Gibbs sampler (nTopics: {}, nIter: {}, alpha: {}, '
                'beta: {}, beta_o: {})'.format(nTopics, nIter, alpha, beta,
                                               beta_o))
    sampler = GibbsSampler(corpus, nTopics=nTopics, nIter=nIter,
                           alpha=alpha, beta=beta, beta_o=beta_o,
                           out_dir=out_dir.format(nTopics))
    sampler._initialize()
    return sampler
