import csv
import random
import math
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

class modelFunctions:
	def separateLabel(self, dataset):
		self.chunkX = []
		for x in range (0, len(dataset)):
			self.chunkX.append(dataset[x][:-1])
		return(self.chunkX)

	def separateClass(self, dataset):
		self.chunkY = []
		for x in range (0, len(dataset)):
			self.chunkY.append(dataset[x][-1])
		return(self.chunkY)

class MLPTrain(modelFunctions):
	def __init__(self, trainingSet):
		trainingSet = trainingSet
		summaries = MLPClassifier(solver='lbgfs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
		trainX = self.separateLabel(trainingSet)
		trainY = self.separateClass(trainingSet)
		self.summaries = summaries.fit(trainX, trainY)

class MLPTest(modelFunctions):
	def __init__(self, summaries, testSet):
		testX = self.separateLabel(testSet)
		testY = self.separateClass(testSet)
		prediction = summaries.predict(testX)
		self.accuracy = metrics.f1_score(testY, prediction)
		print(self.accuracy)