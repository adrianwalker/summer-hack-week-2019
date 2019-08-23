from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd
from numpy import random as np_rand
import random


model = load_model("93percent_accurate_model.dat")


def scale(train):
    sc = StandardScaler()
    train = sc.fit_transform(train)
    # test = sc.transform(test)
    return train


if __name__ == "__main__":
    data = pd.read_csv("pima-indians-diabetes.csv")

    for _ in range(0, 1000):
        working_data = data

        x = scale(working_data.drop(["outcome"], axis=1).values)
        y = working_data["outcome"].values

        predictions = model.predict(x, y)
        print(predictions)
