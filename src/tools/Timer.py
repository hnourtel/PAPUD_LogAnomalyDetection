import sys
import time


class Timer:

    def __init__(self):
        self._startTime = 0
        self._stopTime = 0
        self.lastElapsedTime = 0
        self.totalElapsedTime = 0

        if sys.version_info[0] <= 2:
            self._pythonVersion2 = True
        else:
            self._pythonVersion2 = False


    def start(self):
        if self._pythonVersion2:
            self._startTime = time.clock()
        else:
            self._startTime = time.perf_counter()

    def stop(self):
        if self._pythonVersion2:
            self._stopTime = time.clock()
        else:
            self._stopTime = time.perf_counter()
        self.lastElapsedTime = self._stopTime - self._startTime
        self.totalElapsedTime += self.lastElapsedTime

    def resetLastElapsedTime(self):
        self.lastElapsedTime = 0

