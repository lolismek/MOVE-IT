import extract
import random
import csv

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

file_path = "data/features_data_set3.csv"

X = []
Y = []

test = 0
f = [0, 0, 0, 0, 0, 0, 0]

with open(file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

    for row in csv_reader:
        test += 1
        if test % 1000 == 0:
            print(test)

        aux = row[:-1] #!!!
        i = 0
        while i < len(aux):
            aux[i] = float(aux[i])
            i += 1
        X.append(aux)
        Y.append(int(float(row[len(row) - 1]))) #!!!
        f[int(float(row[len(row) - 1]))] += 1
csv_file.close()

for i in range(0, 7):
    print(f[i])

X = np.array(X)
Y = np.array(Y)

print(np.shape(X))

perm_indices = np.random.permutation(len(X))
X = X[perm_indices]
Y = Y[perm_indices]

Y = tf.one_hot(Y, depth = 6)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
#Y = scaler.fit_transform(Y)

split_index = int(len(X) * 0.8)
X_train = X[:split_index]
X_test = X[split_index:]
Y_train = Y[:split_index]
Y_test = Y[split_index:]

def get_model(input_shape):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    # layer0 = tf.keras.layers.Dense(16384, activation='relu')(input_layer)
    # layer1 = tf.keras.layers.Dense(8192, activation='relu')(layer0)
    # layer2 = tf.keras.layers.Dense(4096, activation='relu')(layer1)
    # layer3 = tf.keras.layers.Dense(1024, activation='relu')(input_layer)
    # layer4 = tf.keras.layers.Dense(512, activation='relu')(layer3)
    # drop1 = tf.keras.layers.Dropout(0.4)(layer4)
    layer5 = tf.keras.layers.Dense(1024, activation='relu')(input_layer)
    layer6 = tf.keras.layers.Dense(24, activation='relu')(layer5)
    layer7 = tf.keras.layers.Dense(6, activation='softmax')(layer6)

    return tf.keras.models.Model(inputs = [input_layer], outputs = [layer7])

def train_model(X_train, Y_train, X_test, Y_test):
    model_path = "best_model.hdf5"
    get_best_model = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, monitor="val_loss", save_best_only = True, verbose = 1)

    optimizer = tf.keras.optimizers.legacy.Adam() 
    loss_function = tf.keras.losses.CategoricalCrossentropy()

    model = get_model(X_train.shape[1])
    model.compile(optimizer = optimizer, loss = [loss_function], metrics = ["accuracy"])
    model.fit(X_train, Y_train, validation_data = (X_test, Y_test), verbose = 1, epochs = 10, batch_size = 32, callbacks=[get_best_model])

def use_best_model(X, Y):
    model_location = "best_model3.hdf5"

    best_model = tf.keras.models.load_model(model_location)
    return best_model.evaluate(x = X, y = Y)

train_model(X_train, Y_train, X_test, Y_test)

#idee: categorisire prin turnament?



