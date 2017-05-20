import csv
import random
import math
from sklearn.naive_bayes import GaussianNB
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

class NaiveBayesSCITrain(modelFunctions):
	def __init__(self, trainingSet):
		trainingSet = trainingSet
		summaries = GaussianNB()
		trainX = self.separateLabel(trainingSet)
		trainY = self.separateClass(trainingSet)
		self.summaries = summaries.fit(trainX, trainY)

class NaiveBayesSCITest(modelFunctions):
	def __init__(self, summaries, testSet):
		testX = self.separateLabel(testSet)
		testY = self.separateClass(testSet)
		prediction = summaries.predict(testX)
		self.accuracy = metrics.f1_score(testY, prediction)
		print(self.accuracy)