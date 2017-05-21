from Static import *
from collections import Counter
import matplotlib.pyplot as plt

"""data = StaticData('RBF1.csv', splitRatio = 0.8)

# proporcje klas
classes = []
for x in range (0, len(data.dataset)):
	classes.append(data.dataset[x][-1])
print(Counter(classes))

data = StaticData('RBF2.csv', splitRatio = 0.8)

# proporcje klas
classes = []
for x in range (0, len(data.dataset)):
	classes.append(data.dataset[x][-1])
print(Counter(classes))

data = StaticData('RBF3.csv', splitRatio = 0.8)

# proporcje klas
classes = []
for x in range (0, len(data.dataset)):
	classes.append(data.dataset[x][-1])
print(Counter(classes))

data = StaticData('sea.csv', splitRatio = 0.8)

# proporcje klas
classes = []
for x in range (0, len(data.dataset)):
	classes.append(data.dataset[x][-1])
print(Counter(classes))"""

data = StaticData('SEA3.csv', splitRatio = 0.8)

# proporcje klas
classes = []
for x in range (0, len(data.dataset)):
	classes.append(data.dataset[x][-1])
print(Counter(classes))