# -*- coding: utf8 -*-

import torch


class Fmeasure():

    def __init__(self, calculationType):
        """

        :param calculationType: Must be Tensor or List. Used to know which calculation method use for probability count
        """

        # Model prediction data
        self.truePositive = 0
        self.falsePositive = 0
        self.falseNegative = 0
        self.trueNegative = 0 #/!\ Works only if there are two classes (True negative is calculated by subtraction between total element count and all other data (TP, FP and FN)

        # Model prediction metrics. Value can be "NaN" if calculation of a value is not possible
        self.fmeasure = 0
        self.precision = 0
        self.recall = 0
        self.specificity = 0

        # Parameters
        if calculationType == "Tensor" or calculationType == "List":
            self.calculationType = calculationType
        else:
            raise ValueError("Calculation type not allowed")


    def metricsCalculation(self, prediction, gold, positiveValue):
        """
        Calculate all relevant metrics. Prediction and gold must have the same number of samples
        :param prediction: Predicted values from the model. Can be Tensor or List
        :param gold: Target values. Can be Tensor or List (but same type as prediction)
        :param positiveValue: The number considered to be the positive value (A true positive is detected if this value is in the prediction and the gold)
        """
        self.retrievePredictionData(prediction, gold, positiveValue)

        # Calculate all metrics
        self.calculateMetricsFromData()


    def retrievePredictionData(self, prediction, gold, positiveValue):
        """
        Retrieve all model prediction data used to calculate metrics
        :param prediction: Predicted values from the model. Can be Tensor or List
        :param gold: Target values. Can be Tensor or List (but same type as prediction)
        :param positiveValue: The number considered to be the positive value (A true positive is detected if this value is in the prediction and the gold)
        """
        if self.calculationType == "Tensor":
            self.predictionDataTensor(prediction, gold, positiveValue)
        elif self.calculationType == "List":
            self.predictionDataList(prediction, gold, positiveValue)



    def predictionDataTensor(self, predictionTensor, goldTensor, positiveValue):

        if predictionTensor.numel() != goldTensor.numel():
            raise ValueError("Prediction and gold tensor have different size. Pred : ", predictionTensor.numel(), " / Gold : ", goldTensor.numel())

        goldPositiveMask = torch.eq(goldTensor, positiveValue)
        predictionPositiveMask = torch.eq(predictionTensor, positiveValue)

        # True positive calculation
        self.truePositive = (torch.masked_select(goldTensor, goldPositiveMask) == torch.masked_select(predictionTensor, goldPositiveMask)).nonzero().numel()

        # False negative calculation
        self.falseNegative = goldPositiveMask.nonzero().size(0) - self.truePositive

        # False positive calculation
        self.falsePositive = (torch.masked_select(goldTensor, predictionPositiveMask) != torch.masked_select(predictionTensor, predictionPositiveMask)).nonzero().numel()

        #  True negative calculation
        self.trueNegative = predictionTensor.numel() - (self.truePositive + self.falseNegative + self.falsePositive)




    def predictionDataList(self, predictionList, goldList, positiveValue):

        # Control input lists
        if len(predictionList) != len(goldList):
            raise ValueError("Prediction and gold list have different size. Pred : ", len(predictionList), " / Gold : ", len(goldList))

        self.truePositive = 0
        self.falsePositive = 0
        self.falseNegative = 0

        for i in range(len(predictionList)):
            if predictionList[i] == positiveValue:
                if predictionList[i] == goldList[i]:
                    self.truePositive += 1
                else:
                    self.falsePositive += 1
            else:
                if goldList[i] == positiveValue:
                    self.falseNegative += 1


        #  True negative calculation
        self.trueNegative = len(predictionList) - (self.truePositive + self.falseNegative + self.falsePositive)


    def calculateMetricsFromData(self):
        """
        Calculate precision, recall and fmeasure from true/false positive/negative values in attributes' class
        :return: fmeasure, precision, recall. Value can be "NaN" if calculation of a value is not possible
        """

        # Calculate precision
        try:
            self.precision = self.truePositive / (self.truePositive + self.falsePositive)
        except ZeroDivisionError:
            self.precision = "NaN"

        # Calculate recall
        try:
            self.recall = self.truePositive / (self.truePositive + self.falseNegative)
        except ZeroDivisionError:
            self.recall = "NaN"

        # Calculate fmeasure
        if self.precision != "NaN" and self.recall != "NaN":
            try:
                self.fmeasure = 2 * ((self.precision * self.recall) / (self.precision + self.recall))
            except ZeroDivisionError:
                self.fmeasure = "NaN"
        else:
            self.fmeasure = "NaN"

        # Calculate specificity (True negative rate)
        try:
            self.specificity = self.trueNegative / (self.trueNegative + self.falsePositive)
        except ZeroDivisionError:
            self.specificity = "NaN"





















