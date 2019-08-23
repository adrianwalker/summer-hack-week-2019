import pandas as pd
from keras import Sequential, optimizers
from keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from statistics import mean
from time import sleep


class DataProcess():
    def __init__(self, input_file):
        # Read CSV into Class
        self._data = pd.read_csv(input_file)
        print("Successfully loaded: {}".format(input_file))
        self._model = None
        self._layers = []

    def split_data(self, key="glyhb", test_size=0.2, scale=True):
        self._data.drop(["location"], axis=1)
        x = self._data.drop([key], axis=1).values
        y = self._data[key].values
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size)
        if scale:
            x_train, x_test = self.standard_scale(x_train, x_test)
        selfx_test, y_test = self._test_data
        print(x_test)
        y_pred = self._model.predict(x_test)
        y_pred = (y_pred > 0.5)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        tp = cm[0][0]
        tn = cm[1][1]
        fp = cm[1][0]
        fn = cm[0][1]
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print("Accuracy: {}".format(accuracy))
        return accuracy


._training_data = x_train, y_train
        self._test_data = x_test, y_test
        return self._training_data, self._test_data

    def add_dense(self, units, **kwargs):
        self._layers.append(Dense(units, **kwargs))

    @staticmethod
    def standard_scale(train, test):
        sc = StandardScaler()
        print(train)
        train = sc.fit_transform(train)
        print(train)
        test = sc.transform(test)
        return train, test

    def compile(self, **kwargs):
        classifier = Sequential()
        for layer in self._layers:
            classifier.add(layer)
        classifier.compile(**kwargs)
        self._model = classifier
        return classifier

    def train(self, **kwargs):
        x_train, y_train = self._training_data
        self._model.fit(x_train, y_train, **kwargs)

    def run_predictions(self):
        x_test, y_test = self._test_data
        print(x_test)
        y_pred = self._model.predict(x_test)
        y_pred = (y_pred > 0.5)
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        tp = cm[0][0]
        tn = cm[1][1]
        fp = cm[1][0]
        fn = cm[0][1]
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print("Accuracy: {}".format(accuracy))
        return accuracy

    def add_dropout(self, weight, **kwargs):
        self._layers.append(Dropout(weight, **kwargs))

    def save(self):
        self._model.save("./keras_model.h5")


if __name__ == "__main__":
    results = []

    dp = DataProcess("diabetes.csv")
    dp.add_dropout(0.2, input_shape=(8,))
    dp.add_dense(3, kernel_initializer="uniform",
                 activation="relu", input_dim=8)
    dp.add_dense(1, kernel_initializer="uniform", activation="sigmoid")
    # optimizer = optimizers.SGD(lr=0.03575, momentum=0.9,
                            #    decay=1e-5, nesterov=True)

    # optimizer = optimizers.Adagrad(lr=0.03575, epsilon=None, decay=0.0)
    optimizer = optimizers.Adadelta(lr=1.0, epsilon=None, decay=0.0)

    dp.compile(optimizer="adadelta",
               loss="binary_crossentropy", metrics=["accuracy"])
    dp.split_data(test_size=0.2)

    # dp.train(batch_size=8, epochs=8) SGD

    largest_result = 0
    while True:
        dp.split_data(test_size=0.2)
        dp.train(batch_size=8, epochs=20)
        result = dp.run_predictions()
        if result > largest_result:
            dp.save()
            print("NEW LARGEST: {}".format(result))
            largest_result = result
            with open("acc.txt", "w") as acc:
                acc.write(str(result))
                acc.close()
            sleep(5)
