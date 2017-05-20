from Static import *
from StaticNaiveBayes import *
import matplotlib.pyplot as plt
import numpy as np


# NaiveBayes - statyczne
data = StaticData('sea.csv', splitRatio = 0.8)
train = TrainModel(data.trainingSet)
test = TestModel(train.summaries, data.testSet)
