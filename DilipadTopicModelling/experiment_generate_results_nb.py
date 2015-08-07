import logging
import argparse
from IPython import nbformat as nbf
import codecs


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('resultsDir', help='directory containing experiment'
                    'result files.')
parser.add_argument('experimentName', help='name of the experiment (used as '
                    'file name for output).')
parser.add_argument('nTopics', help='the number of topics to print results '
                    'for.')
parser.add_argument('outDir', help='directory to write IPython Notebook to')
args = parser.parse_args()

resultsDir = args.resultsDir
experimentName = args.experimentName
nTopics = args.nTopics
outDir = args.outDir

with open('data/CPT_results_template.ipynb') as f:
    nb = nbf.read(f, 4)

# cell 0 = title
nb['cells'][0]['source'] = nb['cells'][0]['source'].format(experimentName)
# cell 4 = set results dir
nb['cells'][4]['source'] = nb['cells'][4]['source'].format(resultsDir)
# cell 11 = set nTopics
nb['cells'][11]['source'] = nb['cells'][11]['source'].format(nTopics)
# cell 13 = set nTopics
nb['cells'][13]['source'] = nb['cells'][13]['source'].format(nTopics)
# cell 22 = set nTopics
nb['cells'][22]['source'] = nb['cells'][22]['source'].format(nTopics)

# save notebook
fName = '{}/{}_{}.ipynb'.format(outDir, experimentName, nTopics)
logger.info('writing notebook {}'.format(fName))
with codecs.open(fName, 'wb', encoding='utf8') as f:
    nbf.write(nb, f)
