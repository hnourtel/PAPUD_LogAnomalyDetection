# -*- coding: utf8 -*-
import argparse
import logging

import torch
import torch.optim as optim

import src.tools.misc as miscTool
from src.model.LANLWordModel import LANLWordModel
from src.tools.ModelSave import ModelSave
from src.tools.Paths import Paths
from src.tools.Timer import Timer
from src.tools.line.LinesTools import LinesTools
from src.tools.metrics.Accuracy import Accuracy

def LANLTrainWord(corpusName, pathAllData, desiredBatchSize, desiredLinesPerBatch, slidingWindowRenewRate, devCalculStep, learningRate, epochNumber):

    # Uncomment to simplify debugging on graphic card
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    # Calculate path of input and output data
    paths = Paths(pathAllData, corpusName)

    #Set logging level
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)

    #--- Vocabulary creation/loading ---
    print("Start vocabulary loading")

    useVocabularyCache = True
    forceRefreshVocabularyCache = False
    stopAfterVocCreation = False

    loadVocabularyTimer = Timer()
    loadVocabularyTimer.start()
    wordDic = miscTool.loadVocabulary(paths.corpusPath, useVocabularyCache, paths.vocabularyCachePath, corpusName, forceRefreshVocabularyCache)
    loadVocabularyTimer.stop()
    print("Vocabulary loading time : " + str(loadVocabularyTimer.totalElapsedTime) + " seconds")

    # Stop program there if you want to only create the vocabulary
    if stopAfterVocCreation:
        print("Request program stop after vocabulary creation")
        exit()

    print("Vocabulary length : " + str(len(wordDic.wordIndex)))


    """ ==========================
             Process the lines 
        ==========================     
    """

    # Tensors parameters
    cudaDevice = "0"
    dtype = torch.long
    forceCPU = False
    if torch.cuda.is_available() and not forceCPU:
        print("Cuda OK")
        device = torch.device("cuda:" + cudaDevice)
        cudaOK = True
    else:
        print("Cuda KO")
        device = torch.device("cpu")
        cudaOK = False

    # Run parameters
    # Batch parameters for corpus that need them

    # For contextSize and embeddingDim see model arguments

    # Construct the DL model
    modelSaving = ModelSave(corpusName, paths.modelPath, "Word", 20000)
    wordModelList = []
    for i in range(8):
        wordModelList.append(LANLWordModel(device, dtype, paths.vocabularyCachePath, corpusName))

    print("Parameters : ")
    totalParam = 0
    currentModelIdx = 0
    for wordModel in wordModelList:
        currentModelIdx += 1
        nbParam = sum(param.numel() for param in wordModel.parameters())
        print("Model ", currentModelIdx, " : ", str(nbParam))
        totalParam += nbParam

    optimizerList = []
    for wordModel in wordModelList:
        if cudaOK:
            wordModel.cuda(device)

        optimizerList.append(optim.Adam(wordModel.parameters(), lr=learningRate))

    # Load lines parameters
    linesParam = LinesTools(corpusName, wordModelList[0].voc, wordModelList[0].lineLength)


    # Load data_dev directory, encode the lines and transfer them to GPU for dev dataset
    print("Encoding data_dev")
    devIterator = linesParam.loadBatch(paths.devPath, False, 0, desiredLinesPerBatch)
    # As we load all devdata in one batch, we need only one iteration over the iterator
    devBatch = next(devIterator)
    devInputTensorList = torch.stack([lineList.convertListToTensor(wordDic, dtype, device) for lineList, fileName in devBatch], 0).view(-1, wordModelList[0].lineLength)

    # Start training process
    batchNum = 0
    totalTime = Timer()
    devTime = Timer()
    saveModelTime = Timer()
    batchTime = Timer()
    batchEncodingTime = Timer()
    batchTrainTime = Timer()

    print("Start training")
    for epoch in range(epochNumber):

        trainDatasetIterator = linesParam.loadBatch(paths.trainPath, False, desiredBatchSize, desiredLinesPerBatch,
                                                        slidingWindowRenewRate, False)

        while True:
            totalTime.start()

            batchTime.start()
            batchEncodingTime.start()

            # Load a batch
            try:
                batch = next(trainDatasetIterator)
            except StopIteration:
                break

            # Convert batch into input and target tensors
            inputTensor = linesParam.convertBatchIntoTensor(batch, dtype, device)
            inputTensor = inputTensor.view(-1, wordModelList[0].lineLength) # Only 1 line per batch => Removing number of lines dimension
            batchEncodingTime.stop()

            batchTrainTime.start()
            accuTrainList = []
            for i in range(wordModelList[0].lineLength):
                accuTrainList.append(Accuracy())

            lossTrainList = [0] * wordModelList[0].lineLength
            for i in range(inputTensor.size(1)):
                optimizerList[i].zero_grad()
                trainInputTensor = torch.cat((inputTensor[:, 0:i], inputTensor[:, i+1:]), 1)
                trainTargetTensor = inputTensor[:, i]

                # Forward pass
                wordModelOutput = wordModelList[i](trainInputTensor)
                targetTensorLoss = trainTargetTensor.view(-1)

                lossTrainList[i] = wordModelList[i].lossFunc(wordModelOutput, targetTensorLoss)
                lossTrainList[i].backward()
                optimizerList[i].step()

                # Retrieve predicted numbers
                predictedWordTensor = torch.argmax(wordModelOutput, 1)
                accuTrainList[i].calculateAccuracyTensors(trainTargetTensor, predictedWordTensor)



            batchTrainTime.stop()
            batchTime.stop()

            # ===== Dev dataset processing =====
            # Each devCalculStep times, run the forward pass on dev dataset
            if batchNum % devCalculStep == 0:

                # Calculate trainLoss and accuracy for dev dataset
                devTotalLoss = 0
                devTime.start()
                accuDevList = []
                for i in range(wordModelList[0].lineLength):
                    accuDevList.append(Accuracy())
                lossDevList = [0] * wordModelList[0].lineLength
                # Set eval mode with model.eval and no_grad to improve computation speed
                with torch.no_grad():
                    for i in range(devInputTensorList.size(1)):
                        wordModelList[i].eval()
                        devInputTensor = torch.cat((devInputTensorList[:, 0:i], devInputTensorList[:, i + 1:]), 1)
                        devTargetTensor = devInputTensorList[:, i]
                        # Forward pass
                        wordModelOutput = wordModelList[i](devInputTensor)
                        targetTensorLoss = devTargetTensor.view(-1)

                        lossDevList[i] = wordModelList[i].lossFunc(wordModelOutput, targetTensorLoss)

                        # Retrieve predicted numbers
                        devPredictedWordTensor = torch.argmax(wordModelOutput, 1)
                        accuDevList[i].calculateAccuracyTensors(devTargetTensor, devPredictedWordTensor)

                        # Reactivate train mode
                        wordModelList[i].train()

                devTime.stop()

                # Display informations about the execution only when dev dataset is passed to the model
                # EPOCH : current epoch count (starts at 0)
                # BATCH : current batch count (starts at 0, not reset when epoch changes)
                # LTR : Loss TRain dataset
                # ACCTR : ACCuracy TRain dataset
                # LDV : Loss DeV dataset
                # ACCDV : ACCuracy DeV dataset
                for i in range(devInputTensorList.size(1)):
                    print("PLOTLOSS " + str(i) + " EPOCH " + str(epoch) + " BATCH " + str(batchNum) +
                          " LTR " + str(round(lossTrainList[i].item(), 6)) + " ACCTR " + str(accuTrainList[i].getTotalAccuracy()) +
                          " LDV " + str(round(lossDevList[i].item(), 6)) + " ACCDV " + str(accuDevList[i].getTotalAccuracy())
                          )

            # End of the batch, save model
            saveModelTime.start()
            wordModelStateDict = {}
            for idx, wordModel in enumerate(wordModelList):
                wordModelStateDict["word" + str(idx)] = wordModel.state_dict()
            modelSaving.saveModel(wordModelStateDict, epoch, batchNum)
            saveModelTime.stop()

            totalTime.stop()

            # Display time information
            if batchNum % devCalculStep == 0:
                print("===== Timers =====")
                print("Last time : ")
                print("  - Total : " + str(totalTime.lastElapsedTime))
                print("  - Batch : " + str(batchTime.lastElapsedTime))
                print("    * Batch encoding : " + str(batchEncodingTime.lastElapsedTime))
                print("    * Batch training : " + str(batchTrainTime.lastElapsedTime))
                print("  - Dev : " + str(devTime.lastElapsedTime))
                print("  - Save : " + str(saveModelTime.lastElapsedTime))

                print("Total time : ")
                print("  - Total : " + str(totalTime.totalElapsedTime))
                print("  - Batch : " + str(batchTime.totalElapsedTime))
                print("    * Batch encoding : " + str(batchEncodingTime.totalElapsedTime))
                print("    * Batch training : " + str(batchTrainTime.totalElapsedTime))
                print("  - Dev : " + str(devTime.totalElapsedTime))
                print("  - Save : " + str(saveModelTime.totalElapsedTime))

                # Reset last elapsed time for save to display 0 if no save is done when batch information are displayed
                saveModelTime.resetLastElapsedTime()

            # Next iteration
            batchNum += 1

        # End of the epoch, save model
        saveModelTime.start()
        wordModelStateDict = {}
        for idx, wordModel in enumerate(wordModelList):
            wordModelStateDict["word" + str(idx)] = wordModel.state_dict()
        modelSaving.saveModel(wordModelStateDict, epoch, batchNum, True)
        saveModelTime.stop()

    print("===== Timers =====")
    print("Last time : ")
    print("  - Total : " + str(totalTime.lastElapsedTime))
    print("  - Batch : " + str(batchTime.lastElapsedTime))
    print("    * Batch encoding : " + str(batchEncodingTime.lastElapsedTime))
    print("    * Batch training : " + str(batchTrainTime.lastElapsedTime))
    print("  - Dev : " + str(devTime.lastElapsedTime))
    print("  - Save : " + str(saveModelTime.lastElapsedTime))

    print("Total time : ")
    print("  - Total : " + str(totalTime.totalElapsedTime))
    print("  - Batch : " + str(batchTime.totalElapsedTime))
    print("    * Batch encoding : " + str(batchEncodingTime.totalElapsedTime))
    print("    * Batch training : " + str(batchTrainTime.totalElapsedTime))
    print("  - Dev : " + str(devTime.totalElapsedTime))
    print("  - Save : " + str(saveModelTime.totalElapsedTime))

    return modelSaving.lastFileSave


if __name__ == "__main__":
    print("Beginning of program")
    # execute only if run as a script
    try:
        desiredBatchSize = 64
        desiredLinesPerBatch = 1
        slidingWindowRenewRate = 0

        devCalculStep = 200
        learningRate = 0.0001
        epochNumber = 1

        # Parsing command lines option
        parser = argparse.ArgumentParser()
        parser.add_argument("corpus_name", choices={"LANL"}, help="Accept LANL")
        parser.add_argument("path_data", help="Path to data directory.")
        args = parser.parse_args()

        savePath = LANLTrainWord(args.corpus_name, args.path_data, desiredBatchSize, desiredLinesPerBatch, slidingWindowRenewRate, devCalculStep, learningRate, epochNumber)

    finally:
        print("=============== End of program ===============")
