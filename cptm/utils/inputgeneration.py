"""Helpers to generate input data for cross-perspective topic modeling."""
import os
import logging
import codecs


logger = logging.getLogger('inputgeneration')

class Perspective():
    def __init__(self, name, posTopic, posOpinion):
        """Initialize inputgeneration Perspective.

        Parameters:
            name : str
                The perspective name. Used as directory name to store the data.
            posTopic : list of strings
                List of strings specifying the pos-tags for topic words.
            posOpinion : list of strings
                List of strings specifying the pos-tags for opinion words.
        """"
        self.name = name
        self.wordTypes = posTopic + posOpinion
        self.posTopic = posTopic
        self.posOpinion = posOpinion
        self.words = {}
        for w in self.wordTypes:
            self.words[w] = []

    def __str__(self):
        len_topic_words, len_opinion_words = self.word_lengths()
        return 'Perspective: {} - {} topic words; {} opinion words'.format(
            self.name, len_topic_words, len_opinion_words)

    def add(self, tag, word):
        self.words[tag].append(word)

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
            for w in self.wordTypes:
                f.write(u'{}\n'.format(' '.join(self.words[w])))

    def word_lengths(self):
        len_topic_words = sum([len(self.words[w])
                               for w in self.posTopic])
        len_opinion_words = sum([len(self.words[w])
                                for w in self.posOpinion])
        return len_topic_words, len_opinion_words
