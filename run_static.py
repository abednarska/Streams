from Static import *
from StaticNaiveBayes import *
from StaticNaiveBayesSCI import *
from StaticMLP import *
from StaticDecisionTree import *
from StaticSVM import *
from StaticKNN import *
import matplotlib.pyplot as plt
import numpy as np


# NaiveBayes 
data = StaticData('sea.csv', splitRatio = 0.8)
train = NaiveBayesTrain(data.trainingSet)
test = NaiveBayesTest(train.summaries, data.testSet)

# NaiveBayesSCI 
data = StaticData('sea.csv', splitRatio = 0.8)
train = NaiveBayesSCITrain(data.trainingSet)
test = NaiveBayesSCITest(train.summaries, data.testSet)

# MLP 
data = StaticData('sea.csv', splitRatio = 0.8)
train = MLPTrain(data.trainingSet)
test = MLPTest(train.summaries, data.testSet)

# DecisionTree 
data = StaticData('sea.csv', splitRatio = 0.8)
train = DecisionTreeTrain(data.trainingSet)
test = DecisionTreeTest(train.summaries, data.testSet)

# SVM
data = StaticData('sea.csv', splitRatio = 0.8)
train = SVMTrain(data.trainingSet)
test = SVMTest(train.summaries, data.testSet)

#KNN
data = StaticData('sea.csv', splitRatio = 0.8)
train = KNNTrain(data.trainingSet)
test = KNNTest(train.summaries, data.testSet)