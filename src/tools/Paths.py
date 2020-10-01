# -*- coding: utf8 -*-
import logging
import os


class Paths:
    """
    Contains all useful paths from the softRoot data path

    Root path must contains at least the following tree :
        rootPath
            |-- corpusName_Corpus
                |-- train
                    |-- [some logs files]
                |-- dev
                    |-- [some logs files]
                |-- test
                    |-- [some logs files]

            |-- corpusName_Model
                |-- [some model files] (optional)

            |-- corpusName_Vocabulary.cache (optional file)

    Files have to be classics text files (non compressed files)
    """

    def __init__(self, rootDataPath, corpusName, create=False):
        """
        Compute tall tree paths from root.

        :param rootDataPath: The root path.
        :param corpusName: The corpus name.
        :param create: If true create missing directories, else raise exception when any directory is missing
        """



        # Corpus
        self.corpusPath = os.path.join(rootDataPath, corpusName + "_Corpus")
        if not os.path.isdir(self.corpusPath):
            if create and not os.path.exists(self.corpusPath):
                logging.info("Create directory: " + self.corpusPath)
                os.makedirs(self.corpusPath)
            else:
                raise ValueError(
                    "Not valid corpus path : " + self.corpusPath + "\nExpected softRoot/corpusName_Corpus/")

        # Train
        self.trainPath = os.path.join(self.corpusPath, 'train')
        if not os.path.isdir(self.trainPath):
            if create and not os.path.exists(self.trainPath):
                logging.info("Create directory: " + self.trainPath)
                os.mkdir(self.trainPath)
            else:
                raise ValueError(
                    "Not valid train path : " + self.trainPath + "\nExpected softRoot/corpusName_Corpus/train")

        # Dev
        self.devPath = os.path.join(self.corpusPath, 'dev')
        if not os.path.isdir(self.devPath):
            if create and not os.path.exists(self.devPath):
                logging.info("Create directory: " + self.devPath)
                os.mkdir(self.devPath)
            else:
                raise ValueError("Not valid dev path : " + self.devPath + "\nExpected softRoot/corpusName_Corpus/dev")

        # Test
        self.testPath = os.path.join(self.corpusPath, 'test')
        if not os.path.isdir(self.testPath):
            if create and not os.path.exists(self.testPath):
                logging.info("Create directory: " + self.testPath)
                os.mkdir(self.testPath)
            else:
                raise ValueError(
                    "Not valid test path : " + self.testPath + "\nExpected softRoot/corpusName_Corpus/test")

        # Model
        self.modelPath = os.path.join(rootDataPath, corpusName + "_Model/")
        if not os.path.isdir(self.modelPath):
            if create and not os.path.exists(self.modelPath):
                logging.info("Create directory: " + self.modelPath)
                os.mkdir(self.modelPath)
            else:
                raise ValueError("Not valid model path : " + self.modelPath + "\nExpected softRoot/corpusName_Model/")

        #  Vocabulary
        self.vocabularyCachePath = os.path.join(rootDataPath, corpusName + "_Vocabulary.cache")
