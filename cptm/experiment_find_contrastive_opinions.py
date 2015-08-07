import logging
import pandas as pd

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
nTopics = 100
out_dir = '/home/jvdzwaan/data/tmp/generated/test_exp/{}'

sampler = GibbsSampler(corpus, nTopics=nTopics, nIter=nIter,
                       alpha=(50.0/nTopics), beta=beta, beta_o=beta,
                       out_dir=out_dir.format(nTopics))
sampler._initialize()

words = corpus.topic_words()

results = pd.DataFrame(index=words, columns=['jsd'])

for word in words:
    co = sampler.contrastive_opinions(word)
    jsd = sampler.jsd_opinions(co)
    results.set_value(word, 'jsd', jsd)

results.to_csv(out_dir.format('jsd.csv'))

print 'top 20 words with most contrastive opinions'
top = pd.Series(results['jsd'])
top.sort(ascending=False)
print top[0:20]
