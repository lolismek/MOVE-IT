import extract
import random
import csv

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

file_path = "data/features_data_set.csv"

X = []
Y = []

with open(file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

    for row in csv_reader:
        aux = row[:-2]
        i = 0
        while i < len(aux):
            aux[i] = float(aux[i])
            i += 1
        X.append(aux)
        Y.append(int(float(row[len(row) - 1]))) #!!!
csv_file.close()

plt.ion()  # Turn on interactive mode
fig = plt.figure()
ax = fig.add_subplot(111) #!!!
ax.set_xlabel('WL')
ax.set_ylabel('Skewness')
#ax.set_zlabel('Z-axis')

colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow']

i = 0
while i < 10000:
    color = colors[Y[i]]

    x = X[i][0]
    y = X[i][1]
    #z = X[i][4]

    #ax.scatter(x, y, z, c=color)
    ax.scatter(x, y, c=color)

    i += 1

plt.ioff()  # Turn off interactive mode at the end
plt.show()