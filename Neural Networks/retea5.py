import extract
import random
import csv

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

file_path = "data/data_set3.csv"

X_train = []
Y_train = []

X_test = []
Y_test = []


take = [False] * 24

for i in range(0, 24):
    prob = random.uniform(0, 1)
    if prob <= 0.8:
        take[i] = True

subject_id = []

with open("data/data_set3.csv", 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

    for row in csv_reader:
        subject_id.append(int(float(row[len(row) - 1])))
csv_file.close()

test = 0
with open(file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

    for row in csv_reader:
        test += 1
        aux = row[:-2] #!!!
        i = 0
        while i < len(aux):
            aux[i] = float(aux[i])
            i += 1
        if take[subject_id[test - 1]]:
            X_train.append(aux)
            Y_train.append(int(float(row[len(row) - 2]))) #!!!
        else:
            X_test.append(aux)
            Y_test.append(int(float(row[len(row) - 2]))) #!!!

csv_file.close()

X_train = np.array(X_train)
Y_train = np.array(Y_train)

print(np.shape(X_train))

X_test = np.array(X_test)
Y_test = np.array(Y_test)

perm_indices = np.random.permutation(len(X_train))
X_train = X_train[perm_indices]
Y_train = Y_train[perm_indices]

Y_train = tf.one_hot(Y_train, depth = 6)
Y_test = tf.one_hot(Y_test, depth = 6)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
Y_train = scaler.fit_transform(Y_train)

X_test = scaler.fit_transform(X_test) #?
Y_test = scaler.fit_transform(Y_test) #?

def get_model(input_shape):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    #layer0 = tf.keras.layers.Dense(16384, activation='relu')(input_layer)
    layer1 = tf.keras.layers.Dense(8192, activation='relu')(input_layer)
    layer2 = tf.keras.layers.Dense(4096, activation='relu')(layer1)
    layer3 = tf.keras.layers.Dense(1024, activation='relu')(layer2)
    layer4 = tf.keras.layers.Dense(512, activation='relu')(layer3)
    drop1 = tf.keras.layers.Dropout(0.4)(layer4)
    layer5 = tf.keras.layers.Dense(256, activation='relu')(drop1)
    layer6 = tf.keras.layers.Dense(24, activation='relu')(layer5)
    layer7 = tf.keras.layers.Dense(6, activation='softmax')(layer6)

    return tf.keras.models.Model(inputs = [input_layer], outputs = [layer7])

def train_model(X_train, Y_train, X_test, Y_test):
    model_path = "best_model2.hdf5"
    get_best_model = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, monitor="val_loss", save_best_only = True, verbose = 1)

    optimizer = tf.keras.optimizers.legacy.Adam() 
    loss_function = tf.keras.losses.CategoricalCrossentropy()

    model = get_model(X_train.shape[1])
    model.compile(optimizer = optimizer, loss = [loss_function], metrics = ["accuracy"])
    model.fit(X_train, Y_train, validation_data = (X_test, Y_test), verbose = 1, epochs = 10, batch_size = 32, callbacks=[get_best_model])

def use_best_model(X, Y):
    model_location = "best_model2.hdf5"

    best_model = tf.keras.models.load_model(model_location)
    return best_model.evaluate(x = X, y = Y)

train_model(X_train, Y_train, X_test, Y_test)

#idee: categorisire prin turnament?



