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
    params['inputData'] = config.get('inputData')
    params['outDir'] = config.get('outDir', '/{}')
    params['testSplit'] = config.get('testSplit', 20)
    params['minFreq'] = config.get('minFreq')
    params['removeTopTF'] = config.get('removeTopTF')
    params['removeTopDF'] = config.get('removeTopDF')
    params['nIter'] = config.get('nIter', 200)
    params['beta'] = config.get('beta', 0.02)
    params['beta_o'] = config.get('beta_o', 0.02)
    params['expNumTopics'] = config.get('expNumTopics', range(20, 201, 20))
    params['nTopics'] = config.get('nTopics')
    params['nProcesses'] = config.get('nProcesses', None)
    params['topicLines'] = config.get('topicLines', [0])
    params['opinionLines'] = config.get('opinionLines', [1])
    params['sampleEstimateStart'] = config.get('sampleEstimateStart')
    params['sampleEstimateEnd'] = config.get('sampleEstimateEnd')

    return params


def add_parameter(name, value, fName):
    with open(fName) as f:
        config = json.load(f)
    config[name] = value
    with open(fName, 'w') as f:
        json.dump(config, f)


def get_corpus(params):
    out_dir = params.get('outDir')
    files = glob.glob(params.get('inputData'))

    if not os.path.isfile(out_dir.format('corpus.json')):
        corpus = CPTCorpus(files,
                           testSplit=params.get('testSplit'),
                           topicLines=params.get('topicLines'),
                           opinionLines=params.get('opinionLines'))
        minFreq = params.get('minFreq')
        removeTopTF = params.get('removeTopTF')
        removeTopDF = params.get('removeTopDF')
        if (not minFreq is None) or (not removeTopTF is None) or \
           (not removeTopDF is None):
            corpus.filter_dictionaries(minFreq=minFreq,
                                       removeTopTF=removeTopTF,
                                       removeTopDF=removeTopDF)
        corpus.save_dictionaries(directory=out_dir.format(''))
        corpus.save(out_dir.format('corpus.json'))
    else:
        corpus = CPTCorpus.load(file_name=out_dir.format('corpus.json'),
                                topicLines=params.get('topicLines'),
                                opinionLines=params.get('opinionLines'),
                                topicDict=out_dir.format('topicDict.dict'),
                                opinionDict=out_dir.format('opinionDict.dict'))
    return corpus


def get_sampler(params, corpus, nTopics=None):
    if nTopics is None:
        nTopics = params.get('nTopics')
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


def thetaFileName(params):
    nTopics = params.get('nTopics')
    return os.path.join(params.get('outDir').format(''),
                        'theta_{}.csv'.format(nTopics))


def topicFileName(params):
    nTopics = params.get('nTopics')
    return os.path.join(params.get('outDir').format(''),
                        'topics_{}.csv'.format(nTopics))


def opinionFileName(params, name):
    nTopics = params.get('nTopics')
    return os.path.join(params.get('outDir').format(''),
                        'opinions_{}_{}.csv'.format(name, nTopics))


def experimentName(params):
    fName = params.get('outDir')
    fName = fName.replace('/{}', '')
    _p, name = os.path.split(fName)
    return name


def tarFileName(params):
    nTopics = params.get('nTopics')
    name = experimentName(params)
    return os.path.join(params.get('outDir').format(''),
                        '{}_{}.tgz'.format(name, nTopics))
