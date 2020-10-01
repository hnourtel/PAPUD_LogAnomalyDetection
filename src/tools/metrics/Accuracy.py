# -*- coding: utf8 -*-

import torch

class Accuracy:


    def __init__(self):

        # Count of good comparison between previous list of predicted lines and previous list of real lines
        self.okComparison = 0
        # Count of comparison done between previous list of predicted lines and previous list of real lines
        self.comparisonCount = 0

        # Total count of good comparison
        self.totalOkComparison = 0
        # Total count of comparison done
        self.totalComparisonCount = 0

        # List to store accuracy each time calculateAccuracy is called
        self.accuracyHistory = []




    def calculateAccuracy(self, predictedLinesList, realLinesList):

        # Retrieve number of lines and length of a line (number of items compounding the line)
        numOfLines = len(predictedLinesList)
        lineLength = len(predictedLinesList[0])

        self.okComparison = 0
        self.comparisonCount = 0

        for lineNum in range(numOfLines):
            for itemNum in range(lineLength):
                self.comparisonCount += 1
                # For each item of the two lines, evaluate prediction
                if predictedLinesList[lineNum][itemNum] == realLinesList[lineNum][itemNum]:
                    self.okComparison += 1


        # Store calculated accuracy
        self.accuracyHistory.append(self.getAccuracy())

        # Change totals count for global accuracy
        self.totalComparisonCount += self.comparisonCount
        self.totalOkComparison += self.okComparison

    def calculateAccuracyTensors(self, targetTensor, predictedTensor):

        self.comparisonCount = targetTensor.numel()
        self.okComparison = torch.eq(targetTensor, predictedTensor).nonzero().size(0)

        self.totalComparisonCount += self.comparisonCount
        self.totalOkComparison += self.okComparison



    def getAccuracy(self):
        try:
            return round(float(self.okComparison) / float(self.comparisonCount), 6)
        except ZeroDivisionError:
            return 0

    def getTotalAccuracy(self):
        try:
            return round(float(self.totalOkComparison) / float(self.totalComparisonCount), 6)
        except ZeroDivisionError:
            return 0