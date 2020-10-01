# -*- coding: utf8 -*-

import logging
import sys
from collections import Counter
from collections import OrderedDict

import src.tools.fileAccess as fa
from src.tools.Timer import Timer
from src.tools.line.LANLLine import LANLLine

class WordDictionary:
    """ Used to manage lines cut into words
    """

    """ -------------------------------
                Constants
        -------------------------------
    """
    # The number representing padding.
    # If its value change, the filling of wordIndex dictionary has to be modified
    PADDING_VALUE = 0

    """ -------------------------------
                    Constructor
        -------------------------------
    """
    def __init__(self, minOccWord, corpus, fileStruct="txt"):
        """
        :param minOccWord: int
            Minimum number of occurrences for a word in vocabulary to be indexed as independent word
        """

        # Minimum number of occurrences for a word in vocabulary to be indexed as independent word
        self.minOccWord = minOccWord

        # Vocabulary extract from lines words.
        # It's a Counter, looks like a dictionary : key = word ; value = word counts
        self.voc = Counter()

        # Storage of numerical value for each word
        # It's a dictionary : key = word ; value = numerical value for the word
        # It will be used to transform a word into a number to be used in neural network
        self.wordIndex = {}

        # Storage of word value for each number
        # It's the mirror of wordIndex (created at the same time)
        # It's a dictionary : key = numerical value for the word ; value = word
        # It will be used to transform a number into a word
        self.indexToWord = {}

        # Number of read bytes used to construct vocabulary
        self.bytesProcessedForVoc = 0

        # Total number of lines used to construct vocabulary
        self.linesCount = 0

        # Total number of words used to construct vocabulary
        self.wordsCount = 0

        # Number of read bytes used to encode a line
        self.bytesProcessedForEncode = 0

        # Number of bytes resulting of encoding
        self.bytesAfterEncode = 0

        # Dictionary to store number of different words (dic value) for a number of parsed lines (dic index)
        self.statDifWords = OrderedDict()

        # Indicates the structures of files used to create vocabulary and encode lines
        self.fileStruct = fileStruct

        self.corpus = corpus


    def createVocabulary(self, *paths):
        """
        Fill vocabulary from several paths
        :param paths: a list of path
        """
        # Init timers
        vocCreation = Timer()
        wordIndexCreation = Timer()

        # We count the number of different words
        numberLineCountWords = 10000
        stepNumberLine = 20000

        # Init
        self.linesCount = 0
        self.wordsCount = 0
        self.bytesProcessedForVoc = 0

        # Parsing lines to create vocabulary
        vocCreation.start()
        for path in paths:
            for line in fa.lines_iterator_directory(path, self.fileStruct, recursive=True):
                if self.corpus == "LANL":
                    preprocessedLine = LANLLine(line, 0)
                else:
                    raise ValueError("Corpus ", self.corpus, " is incorrect")

                self.updateVoc(preprocessedLine)

                # Store statistics about vocabulary
                if self.linesCount == numberLineCountWords:
                    self.statDifWords[numberLineCountWords] = len(self.voc)
                    numberLineCountWords = round(numberLineCountWords + stepNumberLine)

                # Display information
                if self.linesCount % 100000 == 0:
                    print("Lines treated : " + str(self.linesCount))

        vocCreation.stop()

        # Convert dictionary to index
        wordIndexCreation.start()
        self.createWordIndex()
        wordIndexCreation.stop()
        logging.log(logging.INFO, "Voc most common = %s", self.voc.most_common())
        logging.log(logging.INFO, "\nWord index = %s", self.wordIndex)

        # Final info display
        print("\n******************************")
        print("  End of vocabulary creation  ")
        print("******************************")

        print("\n==============================")
        print("         PARAMETERS           ")
        print("==============================")
        print("Minimum word occurrence : ", self.minOccWord)

        print("\n==============================")
        print("         STATISTICS           ")
        print("==============================")
        print("All words vocabulary size : ", len(self.voc))
        print("Index words size : ", len(self.wordIndex))
        print("Number of lines processed : ", self.linesCount)
        print("Size of data processed : ", "{:,}".format(self.bytesProcessedForVoc), " bytes")
        print("Average words per line : ", self._getAverageWords())
        logger = logging.getLogger()
        if logger.getEffectiveLevel() <= logging.INFO:
            print("Number of different words with specific amount of lines parsed : ")
            for numberLine, difWordCount in self.statDifWords.items():
                print("{:,}".format(numberLine), " lines : ", "{:,}".format(difWordCount), " words")

        print("\n==============================")
        print("             TIMERS           ")
        print("==============================")
        print("Vocabulary creation : ", vocCreation.totalElapsedTime, " seconds")
        print("Vocabulary creation speed : ", "{:,}".format(self.bytesProcessedForVoc / vocCreation.totalElapsedTime),
              " B/s")
        print("Index creation : ", wordIndexCreation.totalElapsedTime, " seconds")


    def updateVoc(self, line):
        """
        Update vocabulary with a line as word list

        :param line: LineComponent
            The line to add in vocabulary
        """

        # Update vocabulary
        self.voc.update(line.preprocessedLine)

        # Calculate statistics
        self.bytesProcessedForVoc += sys.getsizeof(line)
        self.linesCount += 1
        self.wordsCount += len(line)

    def createWordIndex(self):
        """
        Create the word index that can be used to convert word lines into number lines
        Uses vocabulary filled with updateVoc
        """
        self.wordIndex = {}
        self.indexToWord = {}

        # Creating special words

        # Padding is index 0
        self.wordIndex["[pad]"] = self.PADDING_VALUE
        # Unknown word is index 1
        self.wordIndex["[uknw]"] = 1


        # Index starts at 2 for normal words (not padding and unknown)
        index = 2


        # Creating other words from most used words in corpus
        # most_common returns a tuple of (key, count) sorted in descending order
        for word, count in self.voc.most_common():
            # Adding the new word only if not previously added
            if word not in self.wordIndex:
                if count >= self.minOccWord:
                    self.wordIndex[word] = index
                    index += 1
                else:
                    # Leave loop because most_common return in occurrence number in descending order so no other word will have count >= self.minOccWord
                    break

        for word, index in self.wordIndex.items():
            self.indexToWord[index] = word


    """ -------------------------------
                Attributes accessors
        -------------------------------
    """

    def _getAverageWords(self):
        try:
            return float(self.wordsCount) / float(self.linesCount)
        except:
            return 0

    """ -------------------------------
                Properties definition
        -------------------------------
    """
    averageWordsPerLine = property(_getAverageWords)
