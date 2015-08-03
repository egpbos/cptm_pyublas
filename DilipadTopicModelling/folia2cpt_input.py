import os
import gzip
from lxml import etree
import logging
import codecs
from fuzzywuzzy import process, fuzz
import argparse
import glob
import datetime
import pandas as pd
from multiprocessing import Pool


NUMBER = 100


class Perspective():
    def __init__(self, name):
        print self.word_types()
        self.name = name
        self.words = {}
        for w in self.word_types():
            self.words[w] = []

    def __str__(self):
        len_topic_words, len_opinion_words = self.word_lengths()
        return 'Perspective: {} - {} topic words; {} opinion words'.format(
            self.name, len_topic_words, len_opinion_words)

    @classmethod
    def pos_topic_words(self):
        return ['N']

    @classmethod
    def pos_opinion_words(self):
        return ['ADJ', 'BW', 'WW']

    @classmethod
    def word_types(self):
        return self.pos_topic_words() + self.pos_opinion_words()

    def write2file(self, out_dir, file_name):
        # create dir (if not exists)
        directory = os.path.join(out_dir, self.name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        # write words to file
        out_file = os.path.join(directory, file_name)
        logger.debug('Writing file {} for perspective {}'.format(out_file,
                     self.name))
        with codecs.open(out_file, 'wb', 'utf8') as f:
            for w in self.word_types():
                f.write(u'{}\n'.format(' '.join(self.words[w])))

    def word_lengths(self):
        len_topic_words = sum([len(self.words[w])
                               for w in Perspective.pos_topic_words()])
        len_opinion_words = sum([len(self.words[w])
                                for w in Perspective.pos_opinion_words()])
        return len_topic_words, len_opinion_words


def extract_words(data_file, nFile, nFiles, coalitions):
    logger.debug('{} ({} of {})'.format(data_file, nFile, nFiles))
    if nFile % NUMBER == 0:
        logger.info('{} ({} of {})'.format(data_file, nFile, nFiles))

    word_tag = '{http://ilk.uvt.nl/FoLiA}w'
    pos_tag = '{http://ilk.uvt.nl/FoLiA}pos'
    lemma_tag = '{http://ilk.uvt.nl/FoLiA}lemma'
    speech_tag = '{http://www.politicalmashup.nl}speech'
    party_tag = '{http://www.politicalmashup.nl}party'
    date_tag = '{http://purl.org/dc/elements/1.1/}date'

    known_parties = ['CDA', 'D66', 'GPV', 'GroenLinks', 'OSF', 'PvdA', 'RPF',
                     'SGP', 'SP', 'VVD', '50PLUS', 'AVP', 'ChristenUnie',
                     'Leefbaar Nederland', 'LPF', 'PvdD', 'PVV']

    f = gzip.open(data_file)
    context = etree.iterparse(f, events=('end',), tag=(speech_tag, date_tag),
                              huge_tree=True)

    # parties
    data = {}
    for party in known_parties:
        data[party] = Perspective(party)
    num_speech = 0
    num_speech_without_party = 0

    # Government vs. opposition
    go_data = {}
    go_data['g'] = Perspective('Government')
    go_data['o'] = Perspective('Opposition')

    for event, elem in context:
        if elem.tag == date_tag:
            d = datetime.datetime.strptime(elem.text, "%Y-%m-%d").date()
            i = coalitions.index.searchsorted(d)
            c = coalitions.ix[coalitions.index[i-1]].tolist()
            coalition_parties = [p for p in c if str(p) != 'nan']
        if elem.tag == speech_tag:
            num_speech += 1
            party = elem.attrib.get(party_tag)
            if party:
                # prevent unwanted subdirectories to be created (happens
                # when there is a / in the party name)
                party = party.replace('/', '-')

                if not data.get(party):
                    p, score1 = process.extractOne(party, known_parties)
                    score2 = fuzz.ratio(party, p)
                    logger.debug('Found match for "{}" to known party "{}" '
                                 '(scores: {}, {})'.format(party, p, score1,
                                                           score2))
                    if score1 >= 90 and score2 >= 90:
                        # change party to known party
                        logger.debug('Change party "{}" to known party "{}"'.
                                     format(party, p))
                        party = p
                        if not data.get(party):
                            data[party] = Perspective(party)
                    else:
                        # add new Perspective
                        logger.debug('Add new perspective for party "{}"'.
                                     format(party))
                        data[party] = Perspective(party)

                if party in coalition_parties:
                    go_perspective = 'g'
                else:
                    go_perspective = 'o'
                logger.debug('date: {}, party: {}, government or opposition: '
                             '{}'.format(str(d), party, go_perspective))

                # find all words
                word_elems = elem.findall('.//{}'.format(word_tag))
                for w in word_elems:
                    pos = w.find(pos_tag).attrib.get('class')
                    l = w.find(lemma_tag).attrib.get('class')
                    if pos in Perspective.word_types():
                        data[party].words[pos].append(l)
                        go_data[go_perspective].words[pos].append(l)
            else:
                num_speech_without_party += 1
    del context
    f.close()
    return data, go_data


def process_file(data_file, nFile, nFiles, coalitions):
        data, go_data = extract_words(data_file, nFile, nFiles, coalitions)

        min_words = 1
        for p, persp in data.iteritems():
            len_topic_words, len_opinion_words = persp.word_lengths()
            if len_topic_words >= min_words and len_opinion_words >= min_words:
                fpath, fname = os.path.split(data_file)
                persp.write2file('{}/parties'.format(dir_out),
                                 '{}.txt'.format(fname))
        for p, persp in go_data.iteritems():
            len_topic_words, len_opinion_words = persp.word_lengths()
            if len_topic_words >= min_words and len_opinion_words >= min_words:
                fpath, fname = os.path.split(data_file)
                persp.write2file('{}/gov_opp'.format(dir_out),
                                 '{}.txt'.format(fname))


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(levelname)s : %(message)s',
                        level=logging.INFO)
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('dir_in', help='directory containing the data '
                        '(gzipped FoLiA XML files)')
    parser.add_argument('dir_out', help='the name of the dir where the '
                        'CPT corpus should be saved.')
    args = parser.parse_args()

    coalitions = pd.read_csv('data/dutch_coalitions.csv', header=None,
                             names=['Date', '1', '2', '3', '4'],
                             index_col=0, parse_dates=True)
    coalitions.sort_index(inplace=True)

    dir_in = args.dir_in
    dir_out = args.dir_out

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    pool = Pool()
    data_files = glob.glob('{}/*/data_folia/*.xml.gz'.format(dir_in))
    results = [pool.apply_async(process_file, args=(data_file, i+1,
                                len(data_files), coalitions))
               for i, data_file in enumerate(data_files)]
    pool.close()
    pool.join()
