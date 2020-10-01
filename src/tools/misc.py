# -*- coding: utf8 -*-

import logging
import os
import pickle

from src.tools.WordDictionary import WordDictionary

"""
******************************************
                Vocabulary
******************************************
"""
def loadVocabulary(dataPath, useCache, cachePath, corpus, forceRefresh = False):
    """
    Extract vocabulary from dataset or from cache file

    :param dataPath: (str) Path to the dataset directory used to construct vocabulary
    :param useCache: (bool) If True, try to use vocabulary saved in cache
    :param cachePath: (str) Path to the vocabulary cache file
    :param forceRefresh: (bool) If True, force reloading vocabulary from dataset directory even if a cache file already exists
    :return: (WordDictionary) Vocabulary loaded in WordDictionary structure
    """

    # Remove the cache file if we want to refresh
    if forceRefresh:
        logging.info("Cache file removed (" + cachePath + ")")
        try:
            os.remove(cachePath)
        except FileNotFoundError:
            # If file is not found, there is no need to remove it
            logging.info("File " + cachePath + " not found then not removed")
            pass

    # Try to load from cache
    if useCache and os.path.isfile(cachePath):
        logging.info("Loading vocabulary from cache (" + cachePath + ") ...")
        f = open(cachePath, 'rb')
        try:
            wordDic = pickle.load(f)
        except UnicodeDecodeError:
            # Temporary for compatibility between Python2 and Python3
            wordDic = pickle.load(f, encoding='latin1')
        f.close()

        logging.info("... Vocabulary loaded from cache")

    else:
        logging.info("Loading vocabulary from log files (" + dataPath + "/[dev | test | train]) ...")
        # Instantiation of the word dictionary
        wordDic = WordDictionary(10, corpus)

        # Create vocabulary
        pathDev = os.path.join(dataPath, 'dev')
        pathTest = os.path.join(dataPath, 'test')
        pathTrain = os.path.join(dataPath, 'train')

        wordDic.createVocabulary(pathTrain)

        logging.info("... Vocabulary loaded from log files")

        if useCache:
            logging.info("Saving vocabulary to cache (" + cachePath + ") ...")

            f = open(cachePath, 'wb')
            pickle.dump(wordDic, f)
            f.close()

            logging.info("... Vocabulary saved in cache")

    return wordDic


def loadVocabularyFromCache(cachePath, corpus):
    """
    Extract vocabulary from cache file only. It sends an exception if the specified file doesn't exists
    :param cachePath: Path to the vocabulary cache file
    :return: (WordDictionary): Vocabulary loaded in WordDictionary structure
    """

    if not os.path.isfile(cachePath):
        raise FileNotFoundError("Vocabulary cache file doesn't exists : ", cachePath)

    return loadVocabulary("", True, cachePath, corpus, False)