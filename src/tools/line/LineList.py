# -*- coding: utf8 -*-

import torch


class LineList:

    def __init__(self, removeDuplicates=False, previousLastLine=None, initLineList=None):
        """
        :param removeDuplicates: bool: Indicates if we want to remove duplicate lines in the list
        :param previousLastLine: LineComponent: The last line of previous LineList (in the case of batch chaining). Only used if removeDuplicates = True
        :param initLineList: list(LineComponent) : Initialize the current line list with the given list of lines. If None, current line list is initialized empty
        """
        if initLineList is None or len(initLineList) == 0:
            self.lineList = [] # List of LineComponent
        else:
            self.lineList = initLineList  # List of LineComponent
        self.removeDuplicates = removeDuplicates
        self.previousLastLine = previousLastLine

        self.atLeastOneLineAdded = False # Indicates if at least one line has been added to the list using addLine method (lines added with initLineList doesn't count)

    def addLine(self, newLine):
        """
        Add a LineComponent to the list
        :newLine: LineComponent: The new line to add to the list
        :deleteDuplicateLines: Indicates if there is a need to avoid duplicate lines in the list
        """

        appendLine = True  # By default, we append the new line

        # Test if line is correctly formatted
        if appendLine and not newLine.keepLine:
            appendLine = False

        # Control of duplicates if needed
        if appendLine and self.removeDuplicates:
            # No element in the list. Checking duplicates with the last line of the previous list
            if len(self.lineList) == 0:
                # If no line from previous list, no test is done and the line is not marked as duplicate
                # If there is a line from previous list, we test the duplicate
                if self.previousLastLine is not None and newLine == self.previousLastLine:
                    appendLine = False
            # Elements exist in the list, comparison is done with the current last line of the list
            else:
                if newLine == self.lineList[-1]:
                    appendLine = False


        # Append the line if all controls are OK
        if appendLine:
            self.lineList.append(newLine)
            self.atLeastOneLineAdded = True

    def addLineList(self, lineListToAdd):
        """
        Add a line list (python list or LineList) to this list
        :param lineListToAdd: The list (python list or LineList) with lines to add
        """
        for line in lineListToAdd:
            self.addLine(line)

    def convertListToTensor(self, voc, dtype, device):
        """
        Convert all preprocessed lines into a tensor using vocabulary to index each word
        :return: Tensor
        """

        # All lines have save size, so all tensors for each line have same size too.
        tensorLineList = [line.getTensorEncodedLine(voc, dtype, device) for line in self.lineList]
        outputTensor = torch.stack(tensorLineList, 0)

        return outputTensor

    def __iter__(self):
        for line in self.lineList:
            yield line

    def __len__(self):
        return len(self.lineList)

    def __getitem__(self, item):
        return self.lineList[item]

    def __delitem__(self, key):
        del self.lineList[key]

    def __str__(self):
        """
        Convert all preprocessed lines of the list into string
        :return: str : All preprocessed lines converted into string separated by carriage return
        """
        return "\n".join([str(line) for line in self.lineList])

    def lineListToStr(self, lineType="preprocessed"):
        return "\n".join(self.lineListToList(lineType))

    def lineListToList(self, lineType="preprocessed"):
        return [line.lineToStr(lineType) for line in self.lineList]