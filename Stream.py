import csv
import random
from math import *

# wczytanie pliku
class StreamDataFunctions:
    def loadCsv(self, filename):
        lines = csv.reader(open(filename, "rt"))
        dataset = list(lines)
        for i in range(len(dataset)):
            dataset[i] = [float(x) for x in dataset[i]]
        return dataset

class StreamData(StreamDataFunctions):
    def __init__(self, filename, chunk_size = 1, schuffle = False):
        self.source_samples = []    
        self.chunk_size = chunk_size
        self.chunk = []

        self.source_samples = self.loadCsv(filename)

        # pomieszanie próbek
        if (schuffle == True):
            random.shuffle(self.source_samples)

        self.rows = len(self.source_samples)

        # wyliczenie ile próbek można wygenerować z pliku
        self.chunks_amt = floor(self.rows / self.chunk_size) - 1

        for x in range(0, self.chunks_amt):  
            first = self.chunk_size * x
            last = self.chunk_size + (self.chunk_size * x)

            self.chunk.append(self.source_samples[first:last])