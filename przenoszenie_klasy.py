import csv

lines = csv.reader(open('Heart.csv', "rt"))
dataset = list(lines)

new_dataset = []

for r in range(0, len(dataset)):
	listone = dataset[r][1:] 
	listtwo = dataset[r][0]

	mergedlist = []
	mergedlist.extend(listone)
	mergedlist.extend(listtwo)

	new_dataset.append(mergedlist)

with open("output.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(new_dataset)