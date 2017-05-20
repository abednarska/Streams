import csv
import random
import math
import matplotlib.pyplot as plt

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
		correct = 0
		for i in range(len(testSet)):
			if testSet[i][-1] == predictions[i]:
				correct += 1
		return (correct/float(len(testSet)))


class StreamPred(modelFunctions):
	def Train(self, trainingSet, limit):
		classes = []
		trainingSet = trainingSet

		if (len(trainingSet) > limit):
			for x in range (0,len(trainingSet)):
				label = trainingSet[x][-1]
				if (label not in classes): classes.append(label)

			classes_amt = len(classes)
			subset_size = math.floor(limit/classes_amt)

			subset =[]

			newTrainingSet = []

			#podzbiory o danej klasie
			for x in range (0, classes_amt):
				class_subset = []
				for y in range (0, len(trainingSet)):
					if (trainingSet[y][-1] == classes[x]):
						class_subset.append(trainingSet[y])
				subset.append(class_subset)

			#jeśli podzbiór próbki jest >= wyliczonemu rozmiarowi podzbioru
			for x in range (0, classes_amt):
				if (len(subset[x]) >= subset_size):
					newTrainingSubset = []
					for z in range(0, subset_size):
						newTrainingSubset.append(subset[x][z])
					newTrainingSet.extend(newTrainingSubset)
			
			#jeśli podzbiór próbki jest < wyliczonemu rozmiarowi podzbioru
			#dane niezbalansowane - losowanie
				if (len(subset[x]) < subset_size):
					newTrainingSubset = []
					for r in range (0, subset_size):
						rand = random.choice(subset[x])
						newTrainingSubset.append(rand)
					newTrainingSet.extend(newTrainingSubset)

			trainingSet = newTrainingSet

		summaries = self.summarizeByClass(trainingSet)
		return(summaries)

	def Test(self, summaries, testSet):
		predictions = self.getPredictions(summaries, testSet)
		accuracy = self.getAccuracy(testSet, predictions)
		return(accuracy)

class NaiveBayesStream(StreamPred):
	def __init__(self, stream, sensitivity = 0.1, limit = 50):
		current_accuracy = 0.

		self.acc = []
		self.acc_candidate = []
		self.acc_primary = []
		model_changes = []

		self.primary_model = self.Train(stream[0], limit) 

		samples_used_to_teach = stream[0]
		for i, chunk in enumerate(stream):
			if (i == len(stream) - 1):
				break
			self.current_model = self.Train(samples_used_to_teach, limit)
			candidate_model = self.Train(chunk, limit)

			current_accuracy = self.Test(self.current_model, stream[i+1])
			candidate_accuracy = self.Test(candidate_model, stream[i+1])
			primary_accuracy = self.Test(self.primary_model, stream[i+1])

			self.acc.append(current_accuracy)
			self.acc_candidate.append(candidate_accuracy)
			self.acc_primary.append(primary_accuracy)

			if (candidate_accuracy - sensitivity > current_accuracy):
				samples_used_to_teach = chunk
				self.current_model = candidate_model
			else:
				samples_used_to_teach.extend(chunk)

			model_changes.append(candidate_accuracy - sensitivity > current_accuracy)

				
		self.avg_acc = (sum(self.acc)/len(self.acc))
		self.avg_acc_candidate = (sum(self.acc_candidate)/len(self.acc_candidate))
		self.avg_acc_primary = (sum(self.acc_primary)/len(self.acc_primary))
		
		self.mcc = sum(model_changes)