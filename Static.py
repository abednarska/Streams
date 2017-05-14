import random
import csv

class StaticDataFunctions:
	def loadCsv(self, filename, splitRatio):
		lines = csv.reader(open(filename, "rt"))
		dataset = list(lines)
		for i in range(len(dataset)):
			dataset[i] = [float(x) for x in dataset[i]]
		return dataset

	def splitDataset(self, dataset, splitRatio):
		trainSize = int(len(dataset) * splitRatio)
		trainSet = []
		copy = list(dataset)
		while len(trainSet) < trainSize:
			index = random.randrange(len(copy))
			trainSet.append(copy.pop(index))
		return [trainSet, copy]

class StaticData(StaticDataFunctions):
    def __init__(self, filename, splitRatio = 0.7):
        self.filename = filename
        self.splitRatio = splitRatio
        self.dataset = []
        self.trainingSet =[]
        self.testSet = []

        self.dataset = self.loadCsv(filename, splitRatio)

        self.trainingSet, self.testSet = self.splitDataset(self.dataset, splitRatio)