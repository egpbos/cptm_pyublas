import logging
import argparse
from IPython import nbformat as nbf
import codecs


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('resultsDir', help='directory containing experiment'
                    'result files.')
parser.add_argument('experimentName', help='name of the experiment')
parser.add_argument('outFile', help='file name to write IPython Notebook to '
                    '(*.ipynb)')
args = parser.parse_args()

resultsDir = args.resultsDir
experimentName = args.experimentName
outFile = args.outFile

with open('data/CPT_results_template.ipynb') as f:
    nb = nbf.read(f, 4)

# cell 0 = title
nb['cells'][0]['source'] = nb['cells'][0]['source'].format(experimentName)
# cell 4 = set results dir
nb['cells'][4]['source'] = nb['cells'][4]['source'].format(resultsDir)

# save notebook
logger.info('writing notebook {}'.format(outFile))
with codecs.open(outFile, 'wb', encoding='utf8') as f:
    nbf.write(nb, f)
