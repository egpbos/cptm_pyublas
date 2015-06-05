"""Script that generates a (synthetic) corpus to test the CPT model.

A text document contains the topic words on the first line and the opinion
words on the second line.
The document generation process is described in the CPT paper.

The corpus consists of two perspectives with each 5 documents containing fixed
topics and opinions:

Topic                           Perspective /P0/    Perspective /P1/
sun, ice_cream, beach           warm                swimming, sunny
ice_cream, vanilla, chocolate   cold                bad
broccoli, carrot                bad                 warm, good

The documents for each perspective are stored in different directories.

Usage: python generateCPTCorpus.py <out dir>
"""
import argparse
import numpy as np
from collections import Counter
import codecs
import os


def generate_opinion_words(topic_counter, num_topics, phi, vocabulary):
    words = []
    # select opinion (index) based on topic occurrence
    om = np.array([float(topic_counter[i]) for i in range(num_topics)])
    om /= sum(om)
    for i in range(length_opinion):
        # opinion words
        topic = np.random.multinomial(1, om).argmax()
        word = np.random.multinomial(1, phi[topic]).argmax()
        words.append(vocabulary[word])
    return words


parser = argparse.ArgumentParser()
#parser.add_argument('num_doc', help='the number of documents to be generated')
#parser.add_argument('num_topic_words', help='the number of topic words per '
#                    'document')
#parser.add_argument('num_opinion_words', help='the number of opinion words '
#                    'per document')
parser.add_argument('out_dir', help='the directory where the generated '
                    'documents should be saved.')
args = parser.parse_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

topic_vocabulary = np.array(['sun',
                             'ice_cream',
                             'beach',
                             'vanilla',
                             'chocolate',
                             'broccoli',
                             'carrot'])
opinion_vocabulary = np.array(['warm',
                               'swimming',
                               'sunny',
                               'cloudy',
                               'bad',
                               'good',
                               'cold'])

real_theta_topic = np.array([[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0],
                             [0.7, 0.3, 0.0],
                             [0.0, 0.5, 0.5]])
real_phi_topic1 = np.array([[0.4, 0.2, 0.4, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.3, 0.0, 0.35, 0.35, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5]])
real_phi_topic2 = np.array([[0.4, 0.2, 0.4, 0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.3, 0.0, 0.35, 0.35, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5]])
real_phi_opinion1 = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
real_phi_opinion2 = np.array([[0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                              [0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0]])

num_topics = real_theta_topic.shape[1]
phi_opinion_perspectives = [real_phi_opinion1, real_phi_opinion2]
phi_topic_perspectives = [real_phi_topic1, real_phi_topic2]
num_perspectives = len(phi_opinion_perspectives)
length_topic = 50
length_opinion = 20


for p in range(num_perspectives):
    p_dir = os.path.join(args.out_dir, 'p{}'.format(p))
    print p_dir
    if not os.path.exists(p_dir):
        os.makedirs(p_dir)

    for m, tm in enumerate(real_theta_topic):
        out_file = os.path.join(p_dir, 'document{}.txt'.format(m+1))
        print out_file
        with codecs.open(out_file, 'wb', 'utf8') as f:
            topic_words = []
            topic_counter = Counter()
            for i in range(length_topic):
                # topic words
                topic = np.random.multinomial(1, tm).argmax()
                topic_counter[topic] += 1
                word = np.random.multinomial(1, phi_topic_perspectives[p][topic]).argmax()
                topic_words.append(topic_vocabulary[word])
            #print topic_counter
            f.write('{}\n'.format(' '.join(topic_words)))

            opinion_words = generate_opinion_words(topic_counter, num_topics,
                                                   phi_opinion_perspectives[p],
                                                   opinion_vocabulary)
            f.write(' '.join(opinion_words))
