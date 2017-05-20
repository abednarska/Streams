import csv
import random
import math
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import metrics
import statistics as stat

class modelFunctions:
	def separateLabel(self, chunk):
		self.chunkX = []
		for x in range (0, len(chunk)):
			self.chunkX.append(chunk[x][:-1])
		return(self.chunkX)

	def separateClass(self, chunk):
		self.chunkY = []
		for x in range (0, len(chunk)):
			self.chunkY.append(chunk[x][-1])
		return(self.chunkY)

class StreamPred(modelFunctions):
	def Train(self, trainingSet, limit, md):
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

		summaries = tree.DecisionTreeClassifier(max_depth=md, min_samples_split=2,random_state=0)
		trainX = self.separateLabel(trainingSet)
		trainY = self.separateClass(trainingSet)
		summaries.fit(trainX, trainY)
		return(summaries)

	def Test(self, summaries, testSet):
		testX = self.separateLabel(testSet)
		testY = self.separateClass(testSet)
		prediction = summaries.predict(testX)
		accuracy = metrics.f1_score(testY, prediction)
		return(accuracy)

class DecisionTreeStream(StreamPred):
	def __init__(self, stream, sensitivity = 0.1, limit = 50, max_depth = 2):
		current_accuracy = 0.

		self.acc = []
		self.acc_candidate = []
		self.acc_primary = []
		model_changes = []

		self.stability = []
		self.stability_candidate = []
		self.stability_primary = []

		self.primary_model = self.Train(stream[0], limit, max_depth) 

		samples_used_to_teach = stream[0]
		for i, chunk in enumerate(stream):
			if (i == len(stream) - 1):
				break
			self.current_model = self.Train(samples_used_to_teach, limit, max_depth)
			candidate_model = self.Train(chunk, limit, max_depth)

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

		self.stability = stat.variance(self.acc)
		self.stability_candidate = stat.variance(self.acc_candidate)
		self.stability_primary = stat.variance(self.acc_primary)
