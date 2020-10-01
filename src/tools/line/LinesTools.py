# -*- coding: utf8 -*-

import os

import torch

import src.tools.fileAccess as fa
from src.tools.line.LANLLine import LANLLine
from src.tools.line.LineList import LineList


class LinesTools:
    """
    Class containing tools to load batch, encode lines, etc.
    """

    def __init__(self, corpus, voc, lineLength):

        # Statistics variables
        self.bytesProcessedForEncode = 0
        self.bytesAfterEncode = 0

        self.voc = voc
        self.lineLength = lineLength

        self.corpus = corpus
        if self.corpus != "LANL":
            raise ValueError("Corpus ", self.corpus, " invalid")


        self.previousTs = None
        self.previousLine = None


    def loadBatch(self, path, oneBatchOneFile, batchSize=0, linesCountInBatchUnit=0, slidingWindowRenewRate=0, useAllLinesInFile=True):
        """
        Load a batch of lines from a list of files
        :param corpus: Name of the corpus from which the files originate
        :param path: Path of the directory containing the files to load or the path of the single file to load in one single batch
        :param oneBatchOneFile: If True, each file corresponds to one batch (all lines of the file go to a single batch) and return only one batch each time (simplfy management of different line length)
                                All other parameters are unused because batch parameters depend only of the file
                                If False, using the other parameters to load the batch
        :param batchSize: Size of the returned batch. If 0, only one batch is returned (with respect of linesCountInBatchUnit) with all data
        :param linesCountInBatchUnit: Lines count in each unit of the batch
        :param slidingWindowRenewRate: If > 0, count of new lines between two batch unit (first n lines of the previous unit will be deleted in the current unit and replaced by n new lines).
                                      If = 0, no lines will be the same between two batch unit
        :param useAllLinesInFile: If True, last lines of a file will be returned separately in one batch with batch size = 1 if file length is not a multiple of linesCountInBatchUnit
                                  If False, all batch units will have the same length and last lines of the file will be skipped if not enough remaining (=lines from the last multiple of linesCountInBatchUnit)
        :return: A batch composed of a list of tuple (lineList, file)
        """

        # Set a flag to decide if duplicates have to be removed or not
        removeDuplicates = False

        corpusIterator = fa.files_iterator(path, True)
        outputBatch = [] # List of tuple (lineList, file) where file is the line source file
        endDirectoryIterator = False
        while not endDirectoryIterator:
            if os.path.isfile(path):
                # We just process a single file
                # First time we process the file, just use it and remember this first pass
                file = path
                endDirectoryIterator = True
            else:
                # Iteration over directory containing all the files
                try:
                    file = next(corpusIterator)
                except StopIteration:
                    if len(outputBatch) > 0:
                        # No more file in the directory.
                        # We yield the last outputBatch if not empty
                        yield outputBatch
                    # Directly break the directory loop because there is no more file to process, it's useless to go to the file iterator
                    break

            # New file = reset all variables because we don't want to mix lines of different files
            lineList = LineList(removeDuplicates)
            self.previousTs = None
            fileIterator = fa.lines_iterator_file(file)
            while True:
                # Iteration over the current file
                try:
                    line = next(fileIterator)
                except StopIteration:
                    # No more line in the file. Go through end of file process
                    if oneBatchOneFile:
                        # All the file is loaded in one batch, append current line list to output, yield the batch and reset batch output list for next call
                        if len(lineList) > 0:
                            outputBatch.append((lineList, file))
                            yield outputBatch
                        outputBatch = []
                    else:
                        # We don't encode all the file in one batch
                        # If linelist reached the batch unit size, it already has been added to the outputBatch list.
                        # Only remains the check of the last lineList if we need to use all lines
                        if useAllLinesInFile and len(lineList) > 0 and lineList.atLeastOneLineAdded and batchSize > 0:
                            # We need to load all lines even if last lineList is not full.
                            # We yield the current line list in one single batch if not empty and one new line has been added to the list
                            # Current outputBatch list will be filled with the next file
                            yield [(lineList, file)]

                    # Leaving file iterator loop to return to the directory iterator loop
                    break

                # Adding current line to the line list
                if self.corpus == "LANL":
                    lineList.addLine(LANLLine(line, self.lineLength))
                else:
                    raise ValueError("Corpus ", self.corpus, " invalid")

                if not oneBatchOneFile:
                    # We don't encode all the file in one batch => checking if batch parameters are reached after adding the new line list to the current batch
                    if len(lineList) == linesCountInBatchUnit:
                        outputBatch.append((lineList, file))
                        # Reset lineList
                        if slidingWindowRenewRate > 0:
                            lineList = LineList(removeDuplicates, lineList[-1], lineList[slidingWindowRenewRate:])
                        else:
                            lineList = LineList(removeDuplicates, lineList[-1])

                        # Checking if batch size is reached
                        if len(outputBatch) == batchSize:
                            # Yield the current batch output and reset the list for the next call
                            yield outputBatch
                            outputBatch = []


    def convertBatchIntoTensor(self, batch, dtype, device):
        """
        Convert a batch containing multiple tuple (LineList, file) into a single tensor of the lines
        :param batch: list((LineList, file(str))) : Contains all lines for the batch
        :return: Batch tensor with len(batch) as the first dimension
        """
        return torch.stack([lineList.convertListToTensor(self.voc, dtype, device) for lineList, fileName in batch], 0)


    """
    Conversion from number list (=encoded line) to string line
    """
    def convertLineIntoWords(self, numberList):
        return [self.voc.indexToWord.get(number) for number in numberList]

    def convertNumberLineIntoString(self, numberList):
        return " ".join(self.convertLineIntoWords(numberList))