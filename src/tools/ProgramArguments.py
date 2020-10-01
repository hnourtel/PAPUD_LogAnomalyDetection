# -*- coding: utf8 -*-

import argparse


class ProgramArguments:

    def __init__(self, withModel=False, withInputFile='N'):
        """

        :param withModel: bool
        :param withInputFile: 3 values : Y, N, D => (Y)es, (N)o, (D)efault accepted
        """

        self.withModel = withModel
        self.withInputFile = withInputFile

        parser = argparse.ArgumentParser()

        # Mandatory arguments for all programs
        parser.add_argument("corpus_name", choices={"LANL"}, help="Accept LANL")
        parser.add_argument("path_data", help="Path to data directory.")

        # Mandatory arguments for some programs
        if withModel:
            parser.add_argument("model_file", help="Name of the model file in model directory")

        if withInputFile == 'Y':
            parser.add_argument("file_name", help="Name of the input file in test directory")
        elif withInputFile == 'D':
            parser.add_argument("file_name", help="Name of the input file in test directory", nargs='?', default="0")

        # Optional argument for all programs
        parser.add_argument("cuda_device", help="Cuda device to use. 0 is default device.", nargs='?', default="0")

        args = parser.parse_args()



        # Mandatory arguments for all programs
        self._corpusName = args.corpus_name
        self._pathData = args.path_data

        # Mandatory arguments for some programs
        if withModel:
            self._modelFile = args.model_file
        else:
            self._modelFile = None

        if withInputFile == 'Y' or withInputFile == 'D':
            self._inputFile = args.file_name
        else:
            self._inputFile = None


        # Optional argument for all programs
        self._cudaDevice = args.cuda_device



    def _getCorpusName(self):
        return self._corpusName

    def _getPathData(self):
        return self._pathData

    def _getModelFile(self):
        if self._modelFile is not None:
            return self._modelFile
        else:
            raise ValueError("Model file not in program argument")

    def _getInputFile(self):
        if self._inputFile is not None:
            return self._inputFile
        else:
            raise ValueError("Input file not in program argument")

    def _getCudaDevice(self):
        return self._cudaDevice

    """ -------------------------------
                Properties definition
            -------------------------------
    """
    corpusName = property(_getCorpusName)
    pathData = property(_getPathData)
    modelFile = property(_getModelFile)
    inputFile = property(_getInputFile)
    cudaDevice = property(_getCudaDevice)