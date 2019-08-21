import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PREGNANCIES_COLUMN = "PREGNANCIES"
GLUCOSE_COLUMN = "GLUCOSE"
BLOOD_PRESSURE_COLUMN = "BLOOD_PRESSURE"
SKIN_THICKNESS_COLUMN = "SKIN_THICKNESS"
INSULIN_COLUMN = "INSULIN"
BMI_COLUMN = "BMI"
DIABETES_FUNC_COLUMN = "DIABETES_FUNC"
AGE_COLUMN = "AGE"
OUTCOME_COLUMN = "OUTCOME"

dataset = pd.read_csv('./pima-indians-diabetes.csv')
dataset.columns = [
    PREGNANCIES_COLUMN, GLUCOSE_COLUMN, BLOOD_PRESSURE_COLUMN,
    SKIN_THICKNESS_COLUMN, INSULIN_COLUMN, BMI_COLUMN,
    DIABETES_FUNC_COLUMN, AGE_COLUMN, OUTCOME_COLUMN
]

print(dataset.shape)
print(dataset.head())

# Clean up any zeros which shouldn't be there
dataset[GLUCOSE_COLUMN] = dataset[GLUCOSE_COLUMN].replace(
    to_replace=0, value=dataset[GLUCOSE_COLUMN].median())
dataset[BLOOD_PRESSURE_COLUMN] = dataset[BLOOD_PRESSURE_COLUMN].replace(
    to_replace=0, value=dataset[BLOOD_PRESSURE_COLUMN].median())
dataset[SKIN_THICKNESS_COLUMN] = dataset[SKIN_THICKNESS_COLUMN].replace(
    to_replace=0, value=dataset[SKIN_THICKNESS_COLUMN].median())
dataset[INSULIN_COLUMN] = dataset[INSULIN_COLUMN].replace(
    to_replace=0, value=dataset[INSULIN_COLUMN].median())
dataset[BMI_COLUMN] = dataset[BMI_COLUMN].replace(
    to_replace=0, value=dataset[BMI_COLUMN].median())
dataset[DIABETES_FUNC_COLUMN] = dataset[DIABETES_FUNC_COLUMN].replace(
    to_replace=0, value=dataset[DIABETES_FUNC_COLUMN].median())

# Split the training dataset 80% / 20%
train_set, test_set = train_test_split(dataset, test_size=0.1)

# Separate outcomes from the rest of the dataset
train_set_outcomes = train_set[OUTCOME_COLUMN].copy().values
train_set = train_set.drop(OUTCOME_COLUMN, axis=1)
test_set_outcomes = test_set[OUTCOME_COLUMN].copy().values
test_set = test_set.drop(OUTCOME_COLUMN, axis=1)

# Scale input values
scaler = StandardScaler()
scaler.fit(train_set)
train_set_scaled = scaler.transform(train_set)
test_set_scaled = scaler.transform(test_set)

model = Sequential()
model.add(Dense(64, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(32, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
model.fit(train_set_scaled, train_set_outcomes, nb_epoch=150, batch_size=10, verbose=0)

scores = model.evaluate(train_set_scaled, train_set_outcomes)
print("Train set %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

scores = model.evaluate(test_set_scaled, test_set_outcomes)
print("Test set %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
