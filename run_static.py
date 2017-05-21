from Static import *
from StaticNaiveBayes import *
from StaticNaiveBayesSCI import *
from StaticMLP import *
from StaticDecisionTree import *
from StaticSVM import *
from StaticKNN import *
import matplotlib.pyplot as plt
import numpy as np

acc_NB = []
acc_NBsci = []
acc_MLP = []
acc_DT = []
acc_SVM = []
acc_KNN = []

for x in range (0, 9):
	data = StaticData('SEA1.csv', splitRatio = 0.8)

	# NaiveBayes 
	train = NaiveBayesTrain(data.trainingSet)
	test = NaiveBayesTest(train.summaries, data.testSet)
	acc_NB.append(test.accuracy)

	# NaiveBayesSCI 
	train = NaiveBayesSCITrain(data.trainingSet)
	test = NaiveBayesSCITest(train.summaries, data.testSet)
	acc_NBsci.append(test.accuracy)

	# MLP 
	train = MLPTrain(data.trainingSet)
	test = MLPTest(train.summaries, data.testSet)
	acc_MLP.append(test.accuracy)

	# DecisionTree 
	train = DecisionTreeTrain(data.trainingSet)
	test = DecisionTreeTest(train.summaries, data.testSet)
	acc_DT.append(test.accuracy)

	# SVM
	train = SVMTrain(data.trainingSet)
	test = SVMTest(train.summaries, data.testSet)
	acc_SVM.append(test.accuracy)

	#KNN
	train = KNNTrain(data.trainingSet)
	test = KNNTest(train.summaries, data.testSet)
	acc_KNN.append(test.accuracy)

print(np.mean(acc_NB))
print(np.mean(acc_NBsci))
print(np.mean(acc_MLP))
print(np.mean(acc_DT))
print(np.mean(acc_SVM))
print(np.mean(acc_KNN))