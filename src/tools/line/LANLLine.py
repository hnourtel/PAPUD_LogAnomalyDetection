# -*- coding: utf8 -*-

from src.tools.line.LineComponent import LineComponent


class LANLLine(LineComponent):

    def __init__(self, inputStrLine, lineLength):

        # Words definitions in a LANL log line
        self.ts = ""
        self.sourceUser = ""
        self.destUser = ""
        self.sourceComputer = ""
        self.destComputer = ""
        self.authType = ""
        self.logonType = ""
        self.authOrientation = ""
        self.authState = ""


        # Call parent's constructor to preprocess the line
        super(LANLLine, self).__init__(inputStrLine, lineLength, ",")


    def extractLineWords(self):

        # First split by comma to extract first columns
        commaSplit = self.inputNormalizedLine.split(",")

        # No extraction executed if line is not correct
        if self.keepLine:
            # Extract relevant words in the log
            self.ts = commaSplit[0]
            self.sourceUser = commaSplit[1]
            self.destUser = commaSplit[2]
            self.sourceComputer = commaSplit[3]
            self.destComputer = commaSplit[4]
            self.authType = commaSplit[5]
            self.logonType = commaSplit[6]
            self.authOrientation = commaSplit[7]
            self.authState = commaSplit[8]


    def calculatePreprocessedLine(self):
        # Preprocessed the line only if we keep it
        if self.keepLine:
            # If changing length, change line length in Seq2SeqParam
            self.preprocessedLine = [self.sourceUser, self.destUser, self.sourceComputer, self.destComputer,
                                     self.authType, self.logonType, self.authOrientation, self.authState]


    def controlInputLine(self):
        """
        Test if line syntax is correct. Stores result in keepLine attribute
        """

        # By default, line is OK
        self.keepLine = True

        # Split the line by comma
        lineToTest = self.inputNormalizedLine.split(",")

        # Ensure that a line is not empty
        if len(lineToTest) == 0:
            self.keepLine = False
            return