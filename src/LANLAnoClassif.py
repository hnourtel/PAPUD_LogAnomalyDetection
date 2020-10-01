# -*- coding: utf8 -*-

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.network.LANLWordModel import LANLWordModel
from src.tools.ModelSave import ModelSave
from src.tools.Paths import Paths
from src.tools.ProgramArguments import ProgramArguments
from src.tools.Timer import Timer
from src.tools.line.LinesTools import LinesTools


def LANLAnoClassif(corpusName, pathAllData, wordModelFilename, desiredBatchSize, desiredLinesPerBatch, slidingWindowRenewRate, nu, eps):

    paths = Paths(pathAllData, corpusName)

    #Set logging level
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)


    """ ==========================
             Process the lines 
        ==========================     
    """

    #  Tensors parameters
    dtype = torch.long
    if torch.cuda.is_available():
        print("Cuda OK")
        device = torch.device("cuda")
        cudaOK = True
    else:
        print("Cuda KO")
        device = torch.device("cpu")
        cudaOK = False

    #  Construct the DL model
    modelsToSave = {} # Contains model (encoder, R, c) for each word
    for i in range(8):
        modelsToSave["word" + str(i)] = {}

    wordModelStateDict = torch.load(paths.modelPath + wordModelFilename, map_location=device)
    wordModelList = []
    for i in range(8):
        wordModel = LANLWordModel(device, dtype, paths.vocabularyCachePath, corpusName, wordModelStateDict["word" + str(i)])
        wordModel.eval()
        if cudaOK:
            wordModel.to(device)
        wordModelList.append(wordModel)

    # Special loss function for anomaly detection
    lossFunc = nn.CrossEntropyLoss(reduction="none")
    lossRepeat = 1000

    # Load lines parameters
    linesParam = LinesTools(corpusName, wordModelList[0].voc, wordModelList[0].lineLength)

    # Hypersphere construction parameters
    cList = []
    RList = []
    for i in range(wordModelList[0].lineLength):
        cList.append(torch.zeros(wordModelList[0].lastLinearOutSize + lossRepeat, device=device)) # Center is the size of last encoder hidden layer + the loss value
        RList.append(torch.tensor(0.0, device=device))

    print("Hypersphere parameters : nu = ", nu)
    modelSaving = ModelSave(corpusName, paths.modelPath, "wordAno_" + str(nu), nameFormat="short", fixedTs=True)

    # Calibrating hypersphere center c. It's the mean of the dataset (calculated with a forward pass on the train dataset)
    timerC = Timer()

    if modelsToSave.get("c") is not None:
        print("Start loading c from disk")
        c = modelsToSave.get("c")
        print("End loading c from disk")
        print("c = " + str(c))
    else:
        print("Start calibrating c")
        with torch.no_grad():

            nbBatch = 0
            nbSamples = 0
            trainDatasetIterator = linesParam.loadBatch(paths.trainPath, False, desiredBatchSize, desiredLinesPerBatch, slidingWindowRenewRate, True)

            for batch in trainDatasetIterator:
                timerC.start()
                nbBatch += 1

                inputTensor = linesParam.convertBatchIntoTensor(batch, dtype, device)
                inputTensor = inputTensor.view(-1, wordModelList[0].lineLength)

                for i in range(inputTensor.size(1)):

                    # Convert batchLog into input tensor for the model
                    trainInputTensor = torch.cat((inputTensor[:, 0:i], inputTensor[:, i + 1:]), 1)
                    trainTargetTensor = inputTensor[:, i]

                    # Encoder pass
                    wordModelOutput = wordModelList[i](trainInputTensor)
                    targetTensorLoss = trainTargetTensor.view(-1)

                    # Used to print vector representing the file
                    #print("Enc vec : " + ' '.join([str(x) for x in encoderOutput.squeeze().tolist()]))
                    currentLoss = lossFunc(wordModelOutput, targetTensorLoss)
                    repeatLoss = currentLoss.unsqueeze(1).repeat(1, lossRepeat)
                    encOut = torch.cat((wordModelList[i].lastHiddenLayer, repeatLoss), 1)
                    encOutSum = torch.sum(encOut, 0)
                    cList[i] += encOutSum
                timerC.stop()

                nbSamples += wordModelOutput.size(0)

                if nbBatch % 1000 == 0:
                    print(str(nbBatch) + " batch processed in " + str(timerC.totalElapsedTime) + " seconds")


            for idx in range(len(cList)):
                c = cList[idx]

                c /= nbSamples
                # If c is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
                print("c" + str(idx) + " before non zero normalization : " + str(cList[idx]))
                c[(abs(c) < eps) & (c < 0)] = -eps
                c[(abs(c) < eps) & (c >= 0)] = eps
                print("c" + str(idx) + " = " + str(c))
                cList[idx] = c

            print("End calibrating c")

    for idx, c in enumerate(cList):
        modelsToSave["word" + str(idx)]["c"] = c

    # Calibration hypersphere radius R
    print("Start calibrating R")
    timerR = Timer()
    optimizerList = []
    for wordModel in wordModelList:
        wordModel.train()
        optimizerList.append(optim.Adam(wordModel.parameters(), lr=0.0001))

    backprogCalculationStep = 50
    batchNum = 0

    for epoch in range(1):
        distList = []
        scoresList = []
        for i in range(wordModelList[0].lineLength):
            distList.append([])
            scoresList.append([])

        for optimizer in optimizerList:
            optimizer.zero_grad()

        trainDatasetIterator = linesParam.loadBatch(paths.trainPath, False, desiredBatchSize, desiredLinesPerBatch, slidingWindowRenewRate, True)

        for batch in trainDatasetIterator:
            timerR.start()
            batchNum += 1

            inputTensor = linesParam.convertBatchIntoTensor(batch, dtype, device)
            inputTensor = inputTensor.view(-1, wordModelList[0].lineLength)

            # Encoder pass
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

                distList[i].append(dist)
                scores = distList[i][-1] - RList[i] ** 2
                scoresList[i].append(scores)

            # Backpropagation and calculation of R
            if len(distList[0]) == backprogCalculationStep:
                for idx, wordModel in enumerate(wordModelList):
                    distCat = torch.cat(distList[idx])
                    scoresCat = torch.cat(scoresList[idx])
                    loss = ((RList[idx]) ** 2) + ((1 / nu) * torch.mean(torch.max(torch.zeros_like(scoresCat), scoresCat)))
                    loss.backward()
                    optimizerList[idx].step()
                    RList[idx] = torch.tensor(np.quantile(np.sqrt(distCat.detach().cpu().numpy()), 1 - nu), device=device)

                    distList[idx] = []
                    scoresList[idx] = []
                    optimizerList[idx].zero_grad()
            timerR.stop()

            if batchNum % 10000 == 0:
                print(str(batchNum) + " batch processed in " + str(timerR.totalElapsedTime) + " seconds")
                print(RList)
                for idx, R in enumerate(RList):
                    print("R" + str(idx) + " = " + str(R))



        # Last backpropagation and calculation of R if needed
        if len(distList) > 0:
            for idx, wordModel in enumerate(wordModelList):
                distCat = torch.cat(distList[idx])
                scoresCat = torch.cat(scoresList[idx])
                loss = ((RList[idx]) ** 2) + ((1 / nu) * torch.mean(torch.max(torch.zeros_like(scoresCat), scoresCat)))
                loss.backward()
                optimizerList[idx].step()
                RList[idx] = torch.tensor(np.quantile(np.sqrt(distCat.detach().cpu().numpy()), 1 - nu), device=device)

                distList[idx] = []
                scoresList[idx] = []
                optimizerList[idx].zero_grad()

    print("End calibrating R")
    for idx, R in enumerate(RList):
        print("R" + str(idx) + " = " + str(R))
        print("R" + str(idx) + "**2 = " + str(R**2))

    # Save R
    for idx, R in enumerate(RList):
        modelsToSave["word" + str(idx)]["R"] = R

    # Save encoder
    for idx, wordModel in enumerate(wordModelList):
        modelsToSave["word" + str(idx)]["model"] = wordModel.state_dict()

    # Save the objects on disk
    modelPathOnDisk = modelSaving.saveObject(modelsToSave)

if __name__ == "__main__":
    print("Beginning of program")
    # execute only if run as a script
    try:
        # Parsing command lines option
        progArg = ProgramArguments(withModel=True)

        #  Calculate all data that depend on input arguments
        corpusName = progArg.corpusName
        pathAllData = progArg.pathData
        seq2seqModelFilename = progArg.modelFile

        desiredBatchSize = 32
        desiredLinesPerBatch = 1
        slidingWindowRenewRate = 0
        nu = 0.005
        eps = 0.01

        LANLAnoClassif(corpusName, pathAllData, seq2seqModelFilename, desiredBatchSize, desiredLinesPerBatch, slidingWindowRenewRate, nu, eps)

    finally:
        print("=============== End of program ===============")
