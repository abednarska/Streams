from Stream import *
from StreamNaiveBayes import *
from StreamNaiveBayesSCI import *
from StreamDecisionTree import *
from StreamMLP import *
from StreamKNN import *
from StreamSVM import *
import matplotlib.pyplot as plt

stream = StreamData('SEA1.csv', chunk_size = 500, schuffle = False)

# NaiveBayes - autorski
clf1 = NaiveBayesStream(stream.chunk, sensitivity = 0.02, limit = 500)
plt.ylim(0.6, 1)
plt.plot(clf1.acc, 'r', label = 'Jakosc z najlepszego modelu')
plt.plot(clf1.acc_candidate, 'b', label = 'Jakosc z modelu kandydujacego')
plt.plot(clf1.acc_primary, 'g', label = 'Jakosc z modelu pierwotnego')
plt.suptitle('Naive Bayes - autorski', fontsize=20)
plt.xlabel('Numer probki')
plt.ylabel('Uzyskana jakosc[%]')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
plt.show()

print('NaiveBayes - autorski')
print(clf1.avg_acc)
print(clf1.avg_acc_candidate)
print(clf1.avg_acc_primary)
print(clf1.mcc)
print(clf1.stability)
print(clf1.stability_candidate)
print(clf1.stability_primary)

# NaiveBayes - scikit
clf2 = NaiveBayesStreamSCI(stream.chunk, sensitivity = 0.02, limit = 300)
plt.ylim(0.6, 1)
plt.plot(clf2.acc, 'r', label = 'Jakosc z najlepszego modelu')
plt.plot(clf2.acc_candidate, 'b', label = 'Jakosc z modelu kandydujacego')
plt.plot(clf2.acc_primary, 'g', label = 'Jakosc z modelu pierwotnego')
plt.suptitle('Naive Bayes - scikit', fontsize=20)
plt.xlabel('Numer probki')
plt.ylabel('Uzyskana jakosc[%]')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
plt.show()

print('NaiveBayes - scikit')
print(clf2.avg_acc)
print(clf2.avg_acc_candidate)
print(clf2.avg_acc_primary)
print(clf2.mcc)
print(clf2.stability)
print(clf2.stability_candidate)
print(clf2.stability_primary)

# MLP - scikit
clf3 = MLPStream(stream.chunk, sensitivity = 0.02, limit = 500)
plt.ylim(0.6, 1)
plt.plot(clf3.acc, 'r', label = 'Jakosc z najlepszego modelu')
plt.plot(clf3.acc_candidate, 'b', label = 'Jakosc z modelu kandydujacego')
plt.plot(clf3.acc_primary, 'g', label = 'Jakosc z modelu pierwotnego')
plt.suptitle('MLP - scikit', fontsize=20)
plt.xlabel('Numer probki')
plt.ylabel('Uzyskana jakosc[%]')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
plt.show()

print('MLP - scikit')
print(clf3.avg_acc)
print(clf3.avg_acc_candidate)
print(clf3.avg_acc_primary)
print(clf3.mcc)
print(clf3.stability)
print(clf3.stability_candidate)
print(clf3.stability_primary)

# Decision Tree - scikit
clf4 = DecisionTreeStream(stream.chunk, sensitivity = 0.02, limit = 500)
plt.ylim(0.6, 1)
plt.plot(clf4.acc, 'r', label = 'Jakosc z najlepszego modelu')
plt.plot(clf4.acc_candidate, 'b', label = 'Jakosc z modelu kandydujacego')
plt.plot(clf4.acc_primary, 'g', label = 'Jakosc z modelu pierwotnego')
plt.suptitle('Decision Tree Stream - scikit', fontsize=20)
plt.xlabel('Numer probki')
plt.ylabel('Uzyskana jakosc[%]')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
plt.show()

print('DecisionTree - scikit')
print(clf4.avg_acc)
print(clf4.avg_acc_candidate)
print(clf4.avg_acc_primary)
print(clf4.mcc)
print(clf4.stability)
print(clf4.stability_candidate)
print(clf4.stability_primary)

# SVM - scikit
clf5 = SVMStream(stream.chunk, sensitivity = 0.02, limit = 500)
plt.ylim(0.6, 1)
plt.plot(clf5.acc, 'r', label = 'Jakosc z najlepszego modelu')
plt.plot(clf5.acc_candidate, 'b', label = 'Jakosc z modelu kandydujacego')
plt.plot(clf5.acc_primary, 'g', label = 'Jakosc z modelu pierwotnego')
plt.suptitle('SVM - scikit', fontsize=20)
plt.xlabel('Numer probki')
plt.ylabel('Uzyskana jakosc[%]')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
plt.show()

print('SVM - scikit')
print(clf5.avg_acc)
print(clf5.avg_acc_candidate)
print(clf5.avg_acc_primary)
print(clf5.mcc)
print(clf5.stability)
print(clf5.stability_candidate)
print(clf5.stability_primary)

# KNN - scikit
clf6 = KNNStream(stream.chunk, sensitivity = 0.02, limit = 500)
plt.ylim(0.6, 1)
plt.plot(clf6.acc, 'r', label = 'Jakosc z najlepszego modelu')
plt.plot(clf6.acc_candidate, 'b', label = 'Jakosc z modelu kandydujacego')
plt.plot(clf6.acc_primary, 'g', label = 'Jakosc z modelu pierwotnego')
plt.suptitle('KNN - scikit', fontsize=20)
plt.xlabel('Numer probki')
plt.ylabel('Uzyskana jakosc[%]')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
plt.show()

print('KNN - scikit')
print(clf6.avg_acc)
print(clf6.avg_acc_candidate)
print(clf6.avg_acc_primary)
print(clf6.mcc)
print(clf6.stability)
print(clf6.stability_candidate)
print(clf6.stability_primary)