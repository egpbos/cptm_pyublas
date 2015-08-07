import logging
import argparse
import pandas as pd
import numpy as np

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
sampler = get_sampler(config, corpus, nTopics)

logger.info('loading opinions')
outDir = config.get('outDir').format(nTopics)
phi_opinion = []
for p in sampler.corpus.perspectives:
    f = '{}/opinions_{}_{}.csv'.format(outDir, p.name, nTopics)
    phi_opinion.append(pd.read_csv(f, index_col=0, encoding='utf-8'))

logger.info('calculating jsd')
# combine opinions from different perspectives and calculate jsd
co = np.zeros((sampler.VO, sampler.nPerspectives))
jsd = np.zeros(nTopics)
for t in range(nTopics):
    for i in range(len(phi_opinion)):
        co[:, i] = phi_opinion[i][str(t)].values
    jsd[t] = sampler.jsd_opinions(co)

fName = '{}/jsd_{}.csv'.format(outDir, nTopics)
logger.info('saving {} to disk'.format(fName))
df = pd.DataFrame({'jsd': jsd})
df.to_csv(fName)
