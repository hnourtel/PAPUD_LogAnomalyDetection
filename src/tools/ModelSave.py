# -*- coding: utf8 -*-

import os
from datetime import datetime

import torch


class ModelSave:

    def __init__(self, corpus, saveDirectory, prefix, saveFrequency=1, nameFormat="normal", fixedTs=False):

        self.lastFileSave = ""
        self.keepLastSave = False
        self.corpus = corpus
        self.saveDirectory = saveDirectory
        self.prefix = prefix
        self.saveFrequency = saveFrequency
        self.nameFormat = nameFormat # "short" = without epoch or batch
        if fixedTs:
            self.ts = "{:%Y-%m-%d_%H:%M:%S}".format(datetime.now())
        else:
            self.ts = None


    def saveModel(self, modelStateDict, currentEpoch=0, currentBatch=0, forceSaving=False):
        """
        Save current model when :
          - batch count reach save frequency
          - forceSaving is enabled
        :param modelStateDict: Model parameters to save
        :param currentEpoch: Current epoch number in the training process. Starts from 0
        :param currentBatch: Current batch number in the training process. Starts from 0
        :param forceSaving: If true, save model whatever other parameters value
        """

        if forceSaving or currentBatch % self.saveFrequency == 0 :
            # Save current model
            savePath = self.saveObject(modelStateDict, currentEpoch, currentBatch)

            # Delete previous model except is flag keepLastSave (turned True from previous call) is True
            if self.keepLastSave:
                # Disabling keepLastSave for the freshly saved model. It will turn True if necessary just below
                self.keepLastSave = False
            else:
                if self.lastFileSave != "":
                    os.remove(self.lastFileSave)

            # Save parameters for next call
            if forceSaving:
                # We keep the model freshly saved when saving was forced
                # keepLastSave will be evaluated on the next call
                self.keepLastSave = True
            self.lastFileSave = savePath


    def saveObject(self, object, currentEpoch=0, currentBatch=0):
        if self.ts is None:
            ts = "{:%Y-%m-%d_%H:%M:%S}".format(datetime.now())
        else:
            ts = self.ts

        if self.nameFormat == "short":
            saveFileName = self.corpus + "_" + self.prefix + "_" + ts + ".pt"
        else:
            saveFileName = self.corpus + "_" + self.prefix + "_" + ts + "_e" + str(currentEpoch) + "b" + str(
                currentBatch) + ".pt"

        savePath = self.saveDirectory + saveFileName
        torch.save(object, savePath)
        return savePath