# -*- coding: utf8 -*-

import argparse
import os
import LANLTrainWord as WordModelScript
import LANLAnoClassifWord as AnoClassifScript
import LANLTestAnoClassWord as AnoClassifTest

"""
Script example of the running pipeline of the anomaly detection in system logs tool
It has 3 parts :
  - Training the LANL Word model
  - Training the anomaly classifier model (DeepSVDD)
  - Testing the anomaly classifier model
"""

if __name__ == "__main__":
    print("=============== Beginning of program ===============")

    # Retrieving command line parameters
    argParser = argparse.ArgumentParser()
    argParser.add_argument("path_data", help="Path to data directory.")
    args = argParser.parse_args()

    # General parameters for all parts
    corpusName = "LANL"

    """
    First part : training the LANL Word model
    This model aims to predict a word in a log line with the other word given as the input
    """
    # Training parameters
    desiredBatchSize = 64 # Number of minibatchs in one call of the model
    desiredLinesPerBatch = 1 # Number of lines loaded in one minibatch
    slidingWindowRenewRate = 0 # If 2 or more lines are selected, determine the window selection sliding between two minibatch
    devCalculStep = 200 # Step for the forward pass on dev dataset
    learningRate = 0.0001 # Learning rate for the model
    epochNumber = 1 # Number of epochs pass on the training dataset

    # Run training
    encoderModelFilepath = WordModelScript.LANLTrainWord(corpusName, args.path_data, desiredBatchSize, desiredLinesPerBatch,
                                                      slidingWindowRenewRate, devCalculStep, learningRate, epochNumber)

    """
    Second part : training the the LANL anomaly classifier model
    This model is a DeepSVDD which classify a line in two class : "anomaly" or "no anomaly"
    """
    #  Training parameters
    desiredBatchSize = 32
    desiredLinesPerBatch = 1
    slidingWindowRenewRate = 0
    nu = 0.005
    eps = 0.01

    # Run training
    anoClassModelPath = AnoClassifScript.LANLAnoClassif(corpusName, args.path_data, os.path.basename(encoderModelFilepath), desiredBatchSize, desiredLinesPerBatch,
                   slidingWindowRenewRate, nu, eps)

    """
    Third part : testing the LANL anomaly classifier
    This part test the classifier using redteam annotation in LANL dataset to calculate metrics (true positives, false negatives, ...)
    """

    # Retrieving redteam file location
    redteamFilePath = os.path.join(args.path_data, "redteam_example")

    # Run testing. Parameters are the same than for training the anomaly classifier model
    AnoClassifTest.testAnomalyClassification(corpusName, args.path_data, os.path.basename(anoClassModelPath), desiredBatchSize, desiredLinesPerBatch,
                              slidingWindowRenewRate, redteamFilePath)

    print("=============== End of program ===============")