import os
import gzip
from lxml import etree
import logging
import codecs

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logger.setLevel(logging.DEBUG)


NUMBER = 100


class Perspective():
    def __init__(self, name):
        self.name = name
        self.topic_words = []
        self.opinion_words = []

    def __str__(self):
        return 'Perspective: {} - {} topic words; {} opinion words'.format(
            self.name, len(self.topic_words), len(self.opinion_words))

    def write2file(self, out_dir, file_name):
        # create dir (if not exists)
        directory = os.path.join(out_dir, self.name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # write words to file
        out_file = os.path.join(directory, file_name)
        with codecs.open(out_file, 'wb', 'utf8') as f:
            f.write(u'{}\n'.format(' '.join(self.topic_words)))
            f.write(u'{}\n'.format(' '.join(self.opinion_words)))

word_tag = '{http://ilk.uvt.nl/FoLiA}w'
pos_tag = '{http://ilk.uvt.nl/FoLiA}pos'
t_tag = '{http://ilk.uvt.nl/FoLiA}t'
lemma_tag = '{http://ilk.uvt.nl/FoLiA}lemma'
speech_tag = '{http://www.politicalmashup.nl}speech'
party_tag = '{http://www.politicalmashup.nl}party'

pos_topic_words = ['N']
pos_opinion_words = ['WW', 'ADJ', 'BW']

out_dir = '/home/jvdzwaan/data/dilipad/perspectives/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

data_dir = '/home/jvdzwaan/data/dilipad'
data_files = [d for d in os.listdir(data_dir) if d.endswith('.xml.gz')]

for i, data_file in enumerate(data_files):
    logger.debug('{} ({} of {})'.format(data_file, i+1, len(data_files)))
    if i % NUMBER == 0:
        logger.info('{} ({} of {})'.format(data_file, i+1, len(data_files)))

    xml_file = os.path.join(data_dir, data_file)
    f = gzip.open(xml_file)
    context = etree.iterparse(f, events=('end',), tag=speech_tag, huge_tree=True)

    data = {}
    num_speech = 0
    num_speech_without_party = 0

    for event, elem in context:
        num_speech += 1
        party = elem.attrib.get(party_tag)
        if party:
            if not data.get(party):
                data[party] = Perspective(party)
            # find all words
            word_elems = elem.findall('.//{}'.format(word_tag))
            for w in word_elems:
                pos = w.find(pos_tag).attrib.get('class')
                l = w.find(lemma_tag).attrib.get('class')
                if pos in pos_topic_words:
                    data[party].topic_words.append(l)
                if pos in pos_opinion_words:
                    data[party].opinion_words.append(l)
        else:
            num_speech_without_party += 1
    del context
    f.close()

    logger.debug('{}: # speech: {} - # speech without party: {}'.format(data_file, num_speech, num_speech_without_party))
    if i % NUMBER == 0:
        for p, persp in data.iteritems():
            logger.info('{}: {}'.format(data_file, persp))

    # write data to file
    min_words = 100
    for p, persp in data.iteritems():
        if len(persp.topic_words) > min_words and \
           len(persp.opinion_words) > min_words:
            persp.write2file(out_dir, '{}.txt'.format(data_file))
