# -*- coding: utf8 -*-
import matplotlib.pyplot as plt
import torch
from scipy.integrate import simps, trapz

from model.tools.metrics.Fmeasure import Fmeasure


class Graphs:

    def __init__(self, probabilityList, positiveValue, negativeValue, thresholdStart, thresholdEnd, thresholdStep, fmeasureMode=None, onGC=False):
        """
        :param probabilityList: A list of tuples with probability as first value and gold value as second value
        :param positiveValue: The value considered as positive value for metrics calculation
        :param negativeValue: The value considered as negative value for metrics calculation
        :param thresholdStart: Start value for the probability threshold
        :param thresholdEnd: End value for the probability threshold
        :param thresholdStep: Step between two threshold value
        """
        # Save all arguments
        self.probabilityList = probabilityList
        self.positiveValue = positiveValue
        self.negativeValue = negativeValue
        self.thresholdStart = thresholdStart
        self.thresholdEnd = thresholdEnd
        self.thresholdStep = thresholdStep

        # Calculate on graphic card ?
        self.onGC = onGC

        # Calculate fmeasure list with model probabilities
        if fmeasureMode is None:
            self.fmeasureList = self.calculateFmeasureList(self.probabilityList)
        elif fmeasureMode == "sort":
            self.fmeasureList = self.calculateFmeasureListSort(self.probabilityList)
        else:
            raise ValueError("fmeasureMode not supported")


    def drawPrecisionRecallCurve(self):
        """
        Plot the precision-recall curve to evaluate a model
        """
        modelPointsPrecision, modelPointsRecall = self.precisionRecallCurve(self.fmeasureList)

        # Baseline. Predicts always the ratio of the positive case (it's a horizontal line)
        positiveCount = 0
        negativeCount = 0
        for prob, gold in self.probabilityList:
            if gold == self.positiveValue:
                positiveCount += 1
            else:
                negativeCount += 1
        baselinePoints = positiveCount / (positiveCount + negativeCount)

        concatList = list(zip(modelPointsRecall, modelPointsPrecision))
        concatList.sort(key=lambda elem: elem[0])
        unconcatList = list(zip(*concatList))
        modelPointsRecall = list(unconcatList[0])
        modelPointsPrecision = list(unconcatList[1])

        # Compute and print the AUC
        realPrecRecAucSimps = simps(modelPointsPrecision, modelPointsRecall)
        baselinePrecRecAucSimps = simps([baselinePoints, baselinePoints], [self.thresholdStart, self.thresholdEnd])
        realPrecRecAucTrapz = trapz(modelPointsPrecision, modelPointsRecall)
        baselinePrecRecAucTrapz = trapz([baselinePoints, baselinePoints], [self.thresholdStart, self.thresholdEnd])
        print("Real Precision-Recall AUC Simps : ", realPrecRecAucSimps)
        print("Real Precision-Recall AUC Trapz : ", realPrecRecAucTrapz)
        print("Baseline Precision-Recall AUC Simps : ", baselinePrecRecAucSimps)
        print("Baseline Precision-Recall AUC Trapz : ", baselinePrecRecAucTrapz)

        #  Plot the graph and display it
        plt.plot(modelPointsRecall, modelPointsPrecision, marker=".", label="Model")
        plt.plot([self.thresholdStart, self.thresholdEnd], [baselinePoints, baselinePoints], linestyle="--", label="Baseline")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.gca().legend()
        plt.show()



    def precisionRecallCurve(self, fmeasureList):
        """
        Outputs the points used to print a Precision-Recall curve
        :param fmeasureList: The list of all fmeasure to construct points of the curve
        :return: List of precision points, list of recall points
        """

        plotPointsPrecision = []
        plotPointsRecall = []
        for fmeasure in fmeasureList:
            if fmeasure.precision != "NaN":
                plotPointsPrecision.append(fmeasure.precision)
            else:
                plotPointsPrecision.append(0)

            if fmeasure.recall != "NaN":
                plotPointsRecall.append(fmeasure.recall)
            else:
                plotPointsRecall.append(0)

        return plotPointsPrecision, plotPointsRecall


    def drawRocCurve(self):
        """
        Plot the ROC curve to evaluate a model
        :param modelOutputProbList: A list of tuples with probability as first value and gold value as second value
        :param thresholdStart: Start value for the probability threshold
        :param thresholdEnd: End value for the probability threshold
        :param thresholdStep: Step between two threshold value
        """

        modelPointsRecall, modelPointsFPRate = self.rocCurve(self.fmeasureList)
        # Baseline. It's a random choose between the two classes, so the probability for each point is 0.5
        # Calculate custom fmeasure list (beacause self.fmeasureList is created with model output probabilities)
        baselineProb = [(0.5, gold) for prob, gold in self.probabilityList]
        baselineFmeasureList = self.calculateFmeasureList(baselineProb)
        baselinePointsRecall, baselinePointsFPRate = self.rocCurve(baselineFmeasureList)

        # Sort points lists
        concatListReal = list(zip(modelPointsRecall, modelPointsFPRate))
        concatListReal.sort(key=lambda elem: elem[0])
        unconcatListReal = list(zip(*concatListReal))
        modelPointsRecall = list(unconcatListReal[0])
        modelPointsFPRate = list(unconcatListReal[1])

        concatListBaseline = list(zip(baselinePointsRecall, baselinePointsFPRate))
        concatListBaseline.sort(key=lambda elem: elem[0])
        unconcatListBaseline = list(zip(*concatListBaseline))
        baselinePointsRecall = list(unconcatListBaseline[0])
        baselinePointsFPRate = list(unconcatListBaseline[1])

        #  Compute and print the AUC
        realRocAucSimps = simps(modelPointsFPRate, modelPointsRecall)
        baselineRocAucSimps = simps(baselinePointsFPRate, baselinePointsRecall)
        realRocAucTrapz = trapz(modelPointsFPRate, modelPointsRecall)
        baselineRocAucTrapz = trapz(baselinePointsFPRate, baselinePointsRecall)
        print("Real ROC AUC Simps : ", realRocAucSimps)
        print("Real ROC AUC Trapz : ", realRocAucTrapz)
        print("Baseline ROC AUC Simps : ", baselineRocAucSimps)
        print("Baseline ROC AUC Trapz : ", baselineRocAucTrapz)

        #  Plot the graph and display it
        plt.plot(modelPointsRecall, modelPointsFPRate, marker=".", label="Model")
        plt.plot(baselinePointsRecall, baselinePointsFPRate, linestyle="--", label="Baseline")
        plt.xlabel("False Positive Rate")
        plt.ylabel("Recall")
        plt.title("ROC Curve")
        plt.text()
        plt.gca().legend()
        plt.show()


    def rocCurve(self, fmeasureList):
        """
        Outputs the points used to print a ROC curve
        :param fmeasureList: The list of all fmeasure to construct points of the curve
        :return: List of recall points, list of false positive rate points
        """

        plotPointsFalsePositiveRate = []
        plotPointsRecall = []

        for fmeasure in fmeasureList:
            if fmeasure.specificity != "NaN":
                plotPointsFalsePositiveRate.append(1 - fmeasure.specificity)
            else:
                plotPointsFalsePositiveRate.append(0)

            if fmeasure.recall != "NaN":
                plotPointsRecall.append(fmeasure.recall)
            else:
                plotPointsRecall.append(0)

        return plotPointsRecall, plotPointsFalsePositiveRate


    def calculateFmeasureList(self, probabilityList):
        """
        :param probabilityList: A list of tuples with probability as first value and gold value as second value
        :return: The list of all fmeasure calculated
        """
        currentThreshold = self.thresholdStart
        fmeasureList = []
        while currentThreshold <= self.thresholdEnd:
            fmeasure = Fmeasure("List")
            predictionList = []
            goldList = []
            for prob, gold in probabilityList:
                if float(prob) >= currentThreshold:
                    predictionList.append(self.positiveValue)
                else:
                    predictionList.append(self.negativeValue)

                goldList.append(gold)

            fmeasure.metricsCalculation(predictionList, goldList, self.positiveValue)
            fmeasureList.append(fmeasure)
            currentThreshold += self.thresholdStep

        return fmeasureList

    def calculateFmeasureListSort(self, probabilityList):
        """
        :param probabilityList: A list of tuples with probability as first value and gold value as second value
        :return: The list of all fmeasure calculated
        """

        # Sort input list by increasing probability
        probabilityList.sort(key=lambda elem: elem[0])

        # Calculate gold value list
        if self.onGC:
            goldList = torch.tensor([currentGold for currentProb, currentGold in probabilityList], device=torch.device('cuda:0'))
            predictionList = torch.tensor([self.positiveValue] * len(probabilityList), device=torch.device('cuda:0'))
            fmeasureMode = "Tensor"
        else:
            goldList = [currentGold for currentProb, currentGold in probabilityList]
            predictionList = [self.positiveValue] * len(probabilityList)
            fmeasureMode = "List"

        fmeasureList = []
        fmeasure = Fmeasure(fmeasureMode)
        fmeasure.metricsCalculation(predictionList, goldList, self.positiveValue)
        fmeasureList.append(fmeasure)
        currentIdx = 0
        while currentIdx < len(probabilityList):
            fmeasure = Fmeasure(fmeasureMode)
            saveProbValue = probabilityList[currentIdx][0]
            while probabilityList[currentIdx][0] == saveProbValue:
                predictionList[currentIdx] = self.negativeValue
                currentIdx += 1
                if currentIdx >= len(probabilityList):
                    break

            fmeasure.metricsCalculation(predictionList, goldList, self.positiveValue)
            fmeasureList.append(fmeasure)

        return fmeasureList
