import csv

file_path = "data/intervale_alergat_mariana.csv"

intervals = []

with open(file_path, 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        print(row)
        intervals.append([int(row[0]), int(row[1])])

file.close()

file_path = "data/alergat_mariana.csv"

data = []
with open(file_path, 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        arr = []
        for el in row:
            arr.append(float(el))
        data.append(arr)

file.close()

indx = 5
device = 1.0
samples = []
f100 = []

c = 0

for arr in data:
    if intervals[indx][0] <= int(arr[1]) and int(arr[1]) <= intervals[indx][1] and arr[0] == device:
        c += 1
        if c <= 100:
            f100.append(int(arr[1]))
        samples.append(int(arr[1]))

samples = sorted(samples)

last = 0
cnt = 0
diffs = []
for el in samples:
    if el - last > 5:
        cnt += 1
        diffs.append(el - last)
    last = el

sumy = 0
for x in diffs:
    sumy += x
sumy -= diffs[0]

print(diffs)
print(len(samples))
print(cnt)

print(intervals[indx][1] - intervals[indx][0])
print(sumy)
    


