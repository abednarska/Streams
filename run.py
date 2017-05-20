from Static import *
from NaiveBayes import *
from Stream import *
import matplotlib.pyplot as plt
import numpy as np
from NaiveBayesStream import *
from NaiveBayesStreamSCI import *
from DecisionTreeStream import *
from MLPStream import *

"""
# NaiveBayes - statyczne
data = StaticData('sea.csv', splitRatio = 0.8)
train = TrainModel(data.trainingSet)
test = TestModel(train.summaries, data.testSet)
"""

# NaiveBayes - autorski
stream = StreamData('sea.csv', chunk_size = 500, schuffle = False)
clf1 = NaiveBayesStream(stream.chunk, sensitivity = 0.02, limit = 500)

fig = plt.figure()
plt.plot(clf1.acc, 'r', label = 'Jakosc z najlepszego modelu')
plt.plot(clf1.acc_candidate, 'b', label = 'Jakosc z modelu kandydujacego')
plt.plot(clf1.acc_primary, 'g', label = 'Jakosc z modelu pierwotnego')
fig.suptitle('Naive Bayes - autorski', fontsize=20)

plt.xlabel('Numer probki')
plt.ylabel('Uzyskana jakosc[%]')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)

plt.show()
print('NaiveBayes - autorski')
print(clf1.avg_acc)
print(clf1.avg_acc_candidate)
print(clf1.avg_acc_primary)
print(clf1.mcc)


# NaiveBayes - scikit
stream = StreamData('sea.csv', chunk_size = 500, schuffle = False)
clf2 = NaiveBayesStreamSCI(stream.chunk, sensitivity = 0.02, limit = 500)

fig = plt.figure()
plt.plot(clf2.acc, 'r', label = 'Jakosc z najlepszego modelu')
plt.plot(clf2.acc_candidate, 'b', label = 'Jakosc z modelu kandydujacego')
plt.plot(clf2.acc_primary, 'g', label = 'Jakosc z modelu pierwotnego')
fig.suptitle('Naive Bayes - scikit', fontsize=20)

plt.xlabel('Numer probki')
plt.ylabel('Uzyskana jakosc[%]')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)

plt.show()
print('NaiveBayes - scikit')
print(clf2.avg_acc)
print(clf2.avg_acc_candidate)
print(clf2.avg_acc_primary)
print(clf2.mcc)

# MLP - scikit
stream = StreamData('sea.csv', chunk_size = 500, schuffle = False)
clf3 = MLPStream(stream.chunk, sensitivity = 0.02, limit = 500)

fig = plt.figure()
plt.plot(clf3.acc, 'r', label = 'Jakosc z najlepszego modelu')
plt.plot(clf3.acc_candidate, 'b', label = 'Jakosc z modelu kandydujacego')
plt.plot(clf3.acc_primary, 'g', label = 'Jakosc z modelu pierwotnego')
fig.suptitle('MLP - scikit', fontsize=20)

plt.xlabel('Numer probki')
plt.ylabel('Uzyskana jakosc[%]')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)

plt.show()
print('MLP - scikit')
print(clf3.avg_acc)
print(clf3.avg_acc_candidate)
print(clf3.avg_acc_primary)
print(clf3.mcc)

# Decision Tree - scikit
stream = StreamData('sea.csv', chunk_size = 500, schuffle = False)
clf4 = DecisionTreeStream(stream.chunk, sensitivity = 0.02, limit = 500)

fig = plt.figure()
plt.plot(clf4.acc, 'r', label = 'Jakosc z najlepszego modelu')
plt.plot(clf4.acc_candidate, 'b', label = 'Jakosc z modelu kandydujacego')
plt.plot(clf4.acc_primary, 'g', label = 'Jakosc z modelu pierwotnego')
fig.suptitle('Decision Tree Stream - scikit', fontsize=20)

plt.xlabel('Numer probki')
plt.ylabel('Uzyskana jakosc[%]')

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)

plt.show()
print('DecisionTree - scikit')
print(clf4.avg_acc)
print(clf4.avg_acc_candidate)
print(clf4.avg_acc_primary)
print(clf4.mcc)