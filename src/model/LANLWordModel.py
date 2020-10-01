# -*- coding: utf8 -*-

from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as nnFunc

import src.tools.misc as miscTool

MODEL_VERSION = '2.0.0'

class LANLWordModel(nn.Module):
    def __init__(self, device, dtype, cachePathVocabulary, corpus, modelStateDict=None):

        super(LANLWordModel, self).__init__()

        # Vocabulary parameters
        self.voc = miscTool.loadVocabularyFromCache(cachePathVocabulary, corpus)
        self.vocSize = len(self.voc.wordIndex)


        self.lineLength = 8

        # Loss function
        self.lossFunc = nn.CrossEntropyLoss()

        # Other
        self.device = device
        self.dtype = dtype
        self.corpus = corpus

        # Layers definition
        #Embeddings
        self.embeddingSize = 100
        self.embeddings = nn.Embedding(self.vocSize, self.embeddingSize)
        #MLP
        """self.linear1 = nn.Linear((self.lineLength - 1) * self.embeddingSize, 600)
        self.linear2 = nn.Linear(600, 400)
        self.linear3 = nn.Linear(400, 800)
        self.lastLinearOutSize = 1000
        self.linearLastHidden = nn.Linear(800, self.lastLinearOutSize)
        self.linearOut = nn.Linear(self.lastLinearOutSize, self.vocSize) # Output is only one word"""
        self.linear1 = nn.Linear((self.lineLength - 1) * self.embeddingSize, 1600)
        self.linear2 = nn.Linear(1600, 800)
        self.lastLinearOutSize = 600
        self.linearLastHidden = nn.Linear(800, self.lastLinearOutSize)
        self.linearOut = nn.Linear(self.lastLinearOutSize, self.vocSize)  # Output is only one word
        # Stores the last hidden layer to be reused
        self.lastHiddenLayer = None

        # Loading pretrained model
        stateDict = None

        if modelStateDict is not None:
            stateDict = modelStateDict


        # Load parameters from given state_dict if valued
        if stateDict is not None:
            try:
                self.load_state_dict(stateDict)
            except RuntimeError:
                # Loading state_dict that was saved with a DataParallel model.
                # Removing "module." in the OrderedDict to load the model without DataParallel
                newStateDict = OrderedDict()
                for k, v in stateDict.items():
                    name = k[7:]  # remove module.
                    newStateDict[name] = v

                self.load_state_dict(newStateDict)



    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        batchSize, wordsPerLine, embeddingDim = embeds.size()

        embeds = embeds.view(batchSize, wordsPerLine * embeddingDim)
        out1 = nnFunc.relu(self.linear1(embeds))
        out2 = nnFunc.relu(self.linear2(out1))
        lastHidden = nnFunc.relu(self.linearLastHidden(out2))
        self.lastHiddenLayer = lastHidden
        out = nnFunc.relu(self.linearOut(lastHidden))
        out = out.view(batchSize, self.vocSize)
        return out