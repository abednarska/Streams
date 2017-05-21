import csv
import random
import math

class modelFunctions:
	def separateByClass(self, dataset):
		separated = {}
		for i in range(len(dataset)):
			vector = dataset[i]
			if (vector[-1] not in separated):
				separated[vector[-1]] = []
			separated[vector[-1]].append(vector)
		return separated

	def mean(self, numbers):
		return sum(numbers)/float(len(numbers))
	 
	def stdev(self, numbers):
		avg = self.mean(numbers)
		if ((len(numbers))==1):
			variance = sum([pow(x-avg,2) for x in numbers])/1
		else:
			variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)

		return math.sqrt(variance)

	def summarize(self, dataset):
		summaries = [(self.mean(attribute), self.stdev(attribute)) for attribute in zip(*dataset)]
		del summaries[-1]
		return summaries

	def summarizeByClass(self, dataset):
		separated = self.separateByClass(dataset)
		summaries = {}
		for classValue, instances in separated.items():
			summaries[classValue] = self.summarize(instances)
		return summaries
	 
	def calculateProbability(self, x, mean, stdev):
		if (stdev == 0):
			stdev = 0.001
			exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
		else:
			exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
		return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
	 
	def calculateClassProbabilities(self, summaries, inputVector):
		probabilities = {}
		for classValue, classSummaries in summaries.items():
			probabilities[classValue] = 1
			for i in range(len(classSummaries)):
				mean, stdev = classSummaries[i]
				x = inputVector[i]
				probabilities[classValue] *= self.calculateProbability(x, mean, stdev)
		return probabilities
				
	def predict(self, summaries, inputVector):
		probabilities = self.calculateClassProbabilities(summaries, inputVector)
		bestLabel, bestProb = None, -1
		for classValue, probability in probabilities.items():
			if bestLabel is None or probability > bestProb:
				bestProb = probability
				bestLabel = classValue
		return bestLabel
	 
	def getPredictions(self, summaries, testSet):
		predictions = []
		for i in range(len(testSet)):
			result = self.predict(summaries, testSet[i])
			predictions.append(result)
		return predictions
	 
	def getAccuracy(self, testSet, predictions):
		TP = 0
		FP = 0
		FN = 0
		
		for i in range(len(testSet)):
			if predictions[i] == 1 and testSet[i][-1] == 1:
				TP += 1
			if predictions[i] == 1 and testSet[i][-1] == 0:
				FP += 1
			if predictions[i] == 0 and testSet[i][-1] == 1:
				FN += 1
		if ((2*TP+FP+FN) == 0):
			accuracy = 2*TP/1
		else:
			accuracy = 2*TP/(2*TP+FP+FN)
		return (accuracy)

class NaiveBayesTrain(modelFunctions):
	def __init__(self, trainingSet):
		separateByClass = self.separateByClass(trainingSet)
		self.summaries = self.summarizeByClass(trainingSet)

class NaiveBayesTest(modelFunctions):
	def __init__(self, summaries, testSet):
		self.accuracy = 0
		predictions = self.getPredictions(summaries, testSet)
		self.accuracy = self.getAccuracy(testSet, predictions)