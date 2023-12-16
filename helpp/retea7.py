import extract
import random
import csv

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

file_path = "data/data_set3.csv"

X = []
Y = []

test = 0

with open(file_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

    for row in csv_reader:
        aux = row[:-2] 
        i = 0
        while i < len(aux):
            aux[i] = float(aux[i])
            i += 1
        X.append(aux)
        Y.append(int(float(row[len(row) - 2])))
        # if test > 1000:
        #     break
        # test += 1
    
csv_file.close()

X = np.array(X)
Y = np.array(Y)

perm_indices = np.random.permutation(len(X))
X = X[perm_indices]
Y = Y[perm_indices]

Y = tf.one_hot(Y, depth = 6)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
#Y = scaler.fit_transform(Y)

print(np.shape(X))

new_X = []
for row in X:
    ind = 0

    vi = []
    for i in range(0, 4):
        vj = []
        for j in range(0, 6):
            vk = []
            for k in range(0, 500):
                vk.append(row[ind])
                ind += 1
            vj.append(vk)
        vi.append(vj)

    new_X.append(vi)

X = new_X

print(np.shape(X))

split_index = int(len(X) * 0.8)
X_train = X[:split_index]
X_test = X[split_index:]
Y_train = Y[:split_index]
Y_test = Y[split_index:]

print(np.shape(X[1]))

def get_model(input_shape):
    input_layer = tf.keras.Input(shape=input_shape)
    Z1 = tf.keras.layers.Conv2D(filters = 128, kernel_size = (1, 100), padding = 'same', strides = 1)(input_layer)
    A1 = tf.keras.layers.ReLU()(Z1)
    Z2 = tf.keras.layers.Conv2D(filters = 128, kernel_size = (1, 100), padding = 'same', strides = 1)(A1)
    A2 = tf.keras.layers.ReLU()(Z2)

    Z3 = tf.keras.layers.Conv2D(filters = 64, kernel_size = (1, 50), padding = 'same', strides = 1)(A2)
    A3 = tf.keras.layers.ReLU()(Z3)
    Z4 = tf.keras.layers.Conv2D(filters = 64, kernel_size = (1, 50), padding = 'same', strides = 1)(A3)
    A4 = tf.keras.layers.ReLU()(Z4)

    F = tf.keras.layers.Flatten()(A4)

    drop1 = tf.keras.layers.Dropout(0.2)(F)
    layer1 = tf.keras.layers.Dense(units = 512, activation = 'relu')(drop1)
    drop2 = tf.keras.layers.Dropout(0.1)(layer1)
    layer2 = tf.keras.layers.Dense(units = 512, activation = 'relu')(drop2)

    output_layer = tf.keras.layers.Dense(units = 6, activation = 'softmax')(layer2)

    return tf.keras.models.Model(inputs = [input_layer], outputs = [output_layer])

def train_model(X_train, Y_train, X_test, Y_test):
    optimizer = tf.keras.optimizers.legacy.Adam() 
    loss_function = tf.keras.losses.CategoricalCrossentropy()

    model = get_model(np.shape(X_train[1]))
    model.compile(optimizer = optimizer, loss = [loss_function], metrics = ["accuracy"])
    #print(np.shape(X_train))
    X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
    model.fit(X_train_tensor, Y_train, validation_data = (X_test_tensor, Y_test), verbose = 1, epochs = 10, batch_size = 32)

train_model(X_train, Y_train, X_test, Y_test)