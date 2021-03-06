from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from numpy.random import shuffle

raw_data = np.loadtxt('./pima-indians-diabetes.csv', delimiter=',')


def split_data(x, y):
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for index, row in enumerate(x):
        if index % 10 == 0:
            x_test.append(row)
            y_test.append(y[index])
        else:
            x_train.append(row)
            y_train.append(y[index])

    if len(x_train) != len(y_train):
        raise RuntimeError(
            "Split Failed, Row Mismatch: {} expected, received {}".format(x_train, y_train))
    elif len(x_test) != len(y_test):
        raise RuntimeError(
            "Split Failed, Row Mismatch: {} expected, received {}".format(x_test, y_test))
    return x_train, y_train, x_test, y_test


x = raw_data[:, 0:8]
y = raw_data[:, 8]

x_train, y_train, x_test, y_test = split_data(x, y)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(64, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(32, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adagrad', metrics=['accuracy'])
model.fit(x_train, y_train, nb_epoch=95, batch_size=10)
scores = model.evaluate(x_train, y_train)

y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred)
print(cm)
tp = cm[0][0]
tn = cm[1][1]
fp = cm[1][0]
fn = cm[0][1]
accuracy = (tp + tn) / (tp + tn + fp + fn)
print("Accuracy: %.2f%%" % (accuracy * 100))


# print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
