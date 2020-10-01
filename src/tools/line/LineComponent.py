# -*- coding: utf8 -*-
import re

import torch


class LineComponent:
    """
    Abstract class
    Takes a string line as input, stores all the words of the line separately and can returns the line as list of word after preprocessing each word
    """

    def __init__(self, inputStrLine, outputLineLength, printLineSeparator=" "):
        """
        :param inputStrLine: Raw string line to process
        :param outputLineLength: Desired length of output preprocessed line
        :param printLineSeparator: Separator of words when printing the line for visualization
        """

        # /!\ This constructor MUST be called in children AFTER declaration of specific line attributes in the child

        # Variables for line management
        self.keepLine = True # Indicates that the line is without errors and can be used for processing
        self.printLineSeparator = printLineSeparator

        # Line data
        self.preprocessedLine = []
        self.inputRawStrLine = inputStrLine # Input raw line for information only. Do not use in methods
        self.inputNormalizedLine = "" # Str normalized input line. Must be used in methods to process the line
        self.outputLineLength = outputLineLength

        # Preprocess the new line
        self.preprocessLine()


    """
    Common methods for all line type
    """
    def preprocessLine(self):
        self.normalizeInputLine()
        self.controlInputLine()
        if self.keepLine:
            self.extractLineWords()
            self.calculatePreprocessedLine()
            self.outputLineSizeNormalization()

    def normalizeInputLine(self):
        # Lowercase all characters in str and remove carriage return character
        self.inputNormalizedLine = re.sub('\n', '', self.inputRawStrLine).lower()

    def outputLineSizeNormalization(self):
        """
        Normalize the output line by cutting it if too long or adding padding if too short
        """

        # Output length == 0 signify that we want to keep the original line length
        if self.outputLineLength > 0:
            paddedLine = ["[pad]"] * self.outputLineLength
            if len(self.preprocessedLine) > self.outputLineLength:
                paddedLine[0:self.outputLineLength] = self.preprocessedLine[0:self.outputLineLength]
            else:
                paddedLine[0:len(self.preprocessedLine)] = self.preprocessedLine[:]

            self.preprocessedLine = paddedLine


    def getEncodedLine(self, voc):
        """
        Encode the preprocessed line into numbers using the given vocabulary
        :return: list[int] : Encoded preprocessed line
        """
        return [voc.wordIndex.get(word, voc.wordIndex["[uknw]"]) for word in self.preprocessedLine]

    def getTensorEncodedLine(self, voc, dtype, device):
        """
        Encode the preprocessed line using a vocabulary and convert it into tensor

        :param voc: Vocabulary used to encode the line
        :param dtype: Type of the values in the tensor
        :param device: Device where the tensor has to be stored
        :return: Tensor representing the encoded line
        """
        return torch.tensor(self.getEncodedLine(voc), dtype=dtype, device=device)

    def __eq__(self, lineToCompare):
        """
        Compare this preprocessed line with another line
        :param lineToCompare: LineComponent
        :return: True if the two lines are equals (same words at the same position in the list), false otherwise
        """
        if isinstance(lineToCompare, self.__class__):
            return self.preprocessedLine == lineToCompare.preprocessedLine
        else:
            return False

    def __len__(self):
        return len(self.preprocessedLine)

    def __iter__(self):
        for word in self.preprocessedLine:
            yield word

    def __str__(self):
        """
        Convert the preprocessed into string with space as separator
        :return: str : The preprocessed line with all words separated by space
        """
        return self.lineToStr()

    def lineToStr(self, lineType="preprocessed"):
        if lineType == "preprocessed":
            return self.printLineSeparator.join(self.preprocessedLine)
        elif lineType == "raw":
            return self.inputNormalizedLine

    """
    Abstract methods
    """
    def extractLineWords(self):
        """
        Extract relevant words in a string line and stores them into class variables
        """
        raise NotImplementedError


    def calculatePreprocessedLine(self):
        """
        Construct a list of words with attributes of the object and stores in attribute the preprocessed line as a list(str)
        """
        raise NotImplementedError

    def controlInputLine(self):
        """
        Test if line syntax is correct. Stores result in keepLine attribute
        """
        raise NotImplementedError

    def getLineSeparator(self):
        """

        :return:
        """