# https://www.kaggle.com/uciml/pima-indians-diabetes-database#diabetes.csv
# https://arxiv.org/ftp/arxiv/papers/1707/1707.08386.pdf

import pandas as pd
from keras.initializers import RandomUniform
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adadelta, Adagrad
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer, \
    Normalizer

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
train_set, test_set = train_test_split(dataset, test_size=0.2)

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

# Create model

initial_weight = 0.5
n_input = 8

n_hidden_1 = 64
drp_hidden_1 = 0.25

n_hidden_2 = 32
drp_hidden_2 = 0.25

n_output = 1
drp_output = 0.25

learning_rate = 1
decay = 0.0
epochs = 100
batch_size = 64

random_weights = RandomUniform(minval=-initial_weight, maxval=initial_weight)

model = Sequential()
model.add(Dense(n_hidden_1, input_dim=n_input, activation='elu', kernel_initializer=random_weights))
model.add(Dropout(drp_hidden_1))
model.add(Dense(n_hidden_2, activation='elu', kernel_initializer=random_weights))
model.add(Dropout(drp_hidden_2))
model.add(Dense(n_output, activation='softplus', kernel_initializer=random_weights))
model.add(Dropout(drp_output))

model.compile(
    metrics=['accuracy'],
    loss='mean_squared_error',
    optimizer=Adadelta(lr=learning_rate, decay=decay))

# Train model
model.fit(train_set_scaled, train_set_outcomes, batch_size=batch_size, epochs=epochs, verbose=0)

# Print accuracy
predicted_outcomes = model.predict_classes(test_set_scaled)
accuracy = accuracy_score(test_set_outcomes, predicted_outcomes)
print("accuracy: %s" % (accuracy * 100))
