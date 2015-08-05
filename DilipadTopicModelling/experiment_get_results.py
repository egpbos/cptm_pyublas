import logging
import os
import pandas as pd
import argparse
import sys

from utils.experiment import load_config, get_corpus, get_sampler


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('json', help='json file containing experiment '
                    'configuration.')
args = parser.parse_args()

config = load_config(args.json)
corpus = get_corpus(config)

nTopics = config.get('nTopics')
start = config.get('sampleEstimateStart')
end = config.get('sampleEstimateEnd')

if nTopics is None or start is None or end is None:
    logger.error('nTopics ({}), sampleEstimateStart ({}), and '
                 'sampleEstimateEnd ({}) cannot be None'.
                 format(nTopics, start, end))
    logger.error('Please update {} to include these parameters'.
                 format(args.json))
    sys.exit()

sampler = get_sampler(config, corpus)
sampler.estimate_parameters(start=start, end=end)

thetaFile = os.path.join(config.get('outDir'), 'theta_{}.csv'.format(nTopics))
pd.DataFrame(sampler.theta).to_csv(thetaFile)

topicFile = os.path.join(config.get('outDir'), 'topics_{}.csv'.format(nTopics))
topics = sampler.topics_to_df(phi=sampler.topics, words=corpus.topic_words())
topics.to_csv(topicFile, encoding='utf8')

for i, p in enumerate(sampler.corpus.perspectives):
    opinionFile = os.path.join(config.get('outDir'),
                               'opinions_{}_{}.csv'.format(p.name, nTopics))
    opinions = sampler.topics_to_df(phi=sampler.opinions[i],
                                    words=corpus.opinion_words())
    opinions.to_csv(opinionFile, encoding='utf8')
