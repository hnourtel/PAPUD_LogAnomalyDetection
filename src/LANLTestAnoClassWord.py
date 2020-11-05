# -*- coding: utf8 -*-

"""
Test classifier trained in anomalyClassification.py
"""

import logging
import os
import re
from collections import Counter

import torch
import torch.nn as nn

from src.model.LANLWordModel import LANLWordModel
from src.tools.Graphs import Graphs
from src.tools.Paths import Paths
from src.tools.ProgramArguments import ProgramArguments
from src.tools.Timer import Timer
from src.tools.line.LinesTools import LinesTools


def testAnomalyClassification(corpusName, pathAllData, anoClassModelFilename, desiredBatchSize, desiredLinesPerBatch, slidingWindowRenewRate, redteamFilePath, testFilePath=""):

    # Retrieving paths
    paths = Paths(pathAllData, corpusName)

    # Set logging level
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)

    if testFilePath == "":
        testFilePath = paths.testPath

    # Decide if the graphs are drawn or if we save only values on the disk
    drawGraph = False

    """ ==========================
             Process the lines 
        ==========================     
    """

    #  Tensors parameters
    dtype = torch.long
    torch.set_printoptions(precision=6)
    if torch.cuda.is_available():
        print("Testing anomaly classification on GPU")
        device = torch.device("cuda")
        cudaOK = True
    else:
        print("Testing anomaly classification on CPU")
        device = torch.device("cpu")
        cudaOK = False

    #  Construct the DL model
    # Load dictionary with encoder, R and c
    savedAnoClassWordModel = torch.load(paths.modelPath + anoClassModelFilename, map_location=device)

    wordModelList = []
    RList = []
    cList = []
    for i in range(len(savedAnoClassWordModel)):
        wordModel = LANLWordModel(device, dtype, paths.vocabularyCachePath, corpusName, savedAnoClassWordModel["word" + str(i)]["model"])
        wordModel.eval()
        if cudaOK:
            wordModel.to(device)
        wordModelList.append(wordModel)
        RList.append(savedAnoClassWordModel["word" + str(i)]["R"])
        cList.append(savedAnoClassWordModel["word" + str(i)]["c"])

    # Load lines parameters
    linesParam = LinesTools(corpusName, wordModelList[0].voc, wordModelList[0].lineLength)

    #  Special loss function for anomaly detection
    lossFunc = nn.CrossEntropyLoss(reduction="none")
    lossRepeat = 1000

    # Load a redteam file in a list to calculate true positive and false positive
    redteamFile = open(redteamFilePath, "r")
    redLineList = []
    for redLine in redteamFile:
        redLineList.append(re.sub('\n', '', redLine).lower())

    # Test files
    with torch.no_grad():
        totalBatchCount = 0
        batchOutCount = 0
        batchInCount = 0
        print("R**2 = ")
        for idx, R in enumerate(RList):
            print("word" + str(idx) + " : " + str(R))

        datasetIterator = linesParam.loadBatch(testFilePath, False, desiredBatchSize, desiredLinesPerBatch, slidingWindowRenewRate, False)

        outLineList = []

        timerTotal = Timer()
        timerModel = Timer()

        scoresListMetrics = [] # Stores all scores for metrics calculation

        timerTotal.start()
        for batch in datasetIterator:

            timerModel.start()

            inputTensor = linesParam.convertBatchIntoTensor(batch, dtype, device)
            inputTensor = inputTensor.view(-1, wordModelList[0].lineLength)
            currentBatchSize = inputTensor.size(0)
            distList = []
            scoresList = []
            for i in range(inputTensor.size(1)):

                trainInputTensor = torch.cat((inputTensor[:, 0:i], inputTensor[:, i+1:]), 1)
                trainTargetTensor = inputTensor[:, i]

                wordModelOutput = wordModelList[i](trainInputTensor)
                targetTensorLoss = trainTargetTensor.view(-1)

                # Adding loss to the last hidden layer output
                currentLoss = lossFunc(wordModelOutput, targetTensorLoss)
                repeatLoss = currentLoss.unsqueeze(1).repeat(1, lossRepeat)
                catOutLoss = torch.cat((wordModelList[i].lastHiddenLayer, repeatLoss), 1)

                # Calculate distance
                dist = torch.sum((catOutLoss - cList[i]) ** 2, dim=1)

                distList.append(dist)
                scores = distList[i] - (RList[i] ** 2)
                scoresList.append(scores)

            
            # Calculate score for a line (= sum of score of each word)
            # If it's strictly positive => anomaly. If not => no anomaly
            scoresListTensor = torch.stack(scoresList)
            scoresSum = torch.sum(scoresListTensor, 0)
            for lineIdx in range(scoresSum.size(0)):
                currentLine = batch[lineIdx][0][0]
                lineExtract = currentLine.ts + "," + currentLine.sourceUser + "," + currentLine.sourceComputer + "," + currentLine.destComputer
                if lineExtract in redLineList:
                    scoresListMetrics.append((scoresSum[lineIdx], 1))
                else:
                    scoresListMetrics.append((scoresSum[lineIdx], 0))

            scoresSumPositive = (torch.gt(scoresSum, 0) == 1).nonzero().squeeze()
            if scoresSumPositive.dim() > 0:
                # More than 2 distance are greater than radius
                listIndexOut = scoresSumPositive.tolist()
            else:
                # 1 or 0 distance greater than radius
                listIndexOut = [scoresSumPositive.item()]

            totalBatchCount += currentBatchSize
            batchInCount += currentBatchSize - len(listIndexOut)
            batchOutCount += len(listIndexOut)
            if len(listIndexOut) > 0:
                # At least one batch has a larger distance than radius. Save all these batch
                for i in listIndexOut:
                    outLineList.append((batch[i][0], batch[i][1], scoresSum[i]))
                """
                print("New batch out hypersphere")
                print("R**2 = " + str(R ** 2))
                print("Dist = " + str(dist))
                print("Index out = " + str(listIndexOut))
                """

            if totalBatchCount % 1000 == 0:
                timerTotal.stop()
                print("Batch processed : " + str(totalBatchCount))
                print("Batch in hypersphere : " + str(batchInCount))
                print("Batch out hypersphere : " + str(batchOutCount))
                print("Timers : ")
                print("  Total time : " + str(timerTotal.totalElapsedTime))
                print("  Model time : " + str(timerModel.totalElapsedTime))
                timerTotal.start()

        timerTotal.stop()

    outputFileOutEx = open("LANLanoClassOutEx.txt", 'w')
    # Analyze batch out hypersphere
    truePositive = 0
    falsePositive = 0

    outputFileOutEx.close()

    print(Counter(elem[1] for elem in scoresListMetrics))
    graphs = Graphs(scoresListMetrics, 1, 0, 0, 1, 0.01, "sort", cudaOK)
    if drawGraph:
        graphs.drawPrecisionRecallCurve()
    else:
        # Save graphs value on the disk
        torch.save(graphs, os.path.join(pathAllData, "graphValue.obj"))

    print("End of anomaly classification")
    print("Total batch tested : " + str(totalBatchCount))
    print("Batch in hypersphere : " + str(batchInCount))
    print("Batch out hypersphere : " + str(batchOutCount))
    print("Timers : ")
    print("  Total time : " + str(timerTotal.totalElapsedTime))
    print("  Model time : " + str(timerModel.totalElapsedTime))
    print("Results : ")
    print("True positive : ", truePositive)
    print("False positive : ", falsePositive)


if __name__ == "__main__":
    print("Beginning of program")
    # execute only if run as a script
    try:
        # Parsing command lines option
        progArg = ProgramArguments(withModel=True)

        corpusName = progArg.corpusName
        pathAllData = progArg.pathData
        anoClassModelFilename = progArg.modelFile

        desiredBatchSize = 128
        desiredLinesPerBatch = 1
        slidingWindowRenewRate = 1

        redteamFilePath = pathAllData + "redteam_example"

        testAnomalyClassification(corpusName, pathAllData, anoClassModelFilename, desiredBatchSize, desiredLinesPerBatch, slidingWindowRenewRate, redteamFilePath)


    finally:
        print("=============== End of program ===============")
