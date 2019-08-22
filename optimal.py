import os
import time
from pathlib import Path

import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine.saving import load_model
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD, Adagrad
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

PREGNANCIES_COLUMN = "PREGNANCIES"
GLUCOSE_COLUMN = "GLUCOSE"
BLOOD_PRESSURE_COLUMN = "BLOOD_PRESSURE"
SKIN_THICKNESS_COLUMN = "SKIN_THICKNESS"
INSULIN_COLUMN = "INSULIN"
BMI_COLUMN = "BMI"
DIABETES_FUNC_COLUMN = "DIABETES_FUNC"
AGE_COLUMN = "AGE"
OUTCOME_COLUMN = "OUTCOME"

DATA_FILE = './pima-indians-diabetes.csv'
MODEL_FILE = './model.dat'
SCALER_FILE = './scaler.dat'

# Remove previous saved model
if os.path.exists(MODEL_FILE):
    os.remove(MODEL_FILE)

# Remove previous saved scaler
if os.path.exists(SCALER_FILE):
    os.remove(SCALER_FILE)

# Read data file
dataset = pd.read_csv(DATA_FILE)
dataset.columns = [
    PREGNANCIES_COLUMN, GLUCOSE_COLUMN, BLOOD_PRESSURE_COLUMN,
    SKIN_THICKNESS_COLUMN, INSULIN_COLUMN, BMI_COLUMN,
    DIABETES_FUNC_COLUMN, AGE_COLUMN, OUTCOME_COLUMN
]

# Clean up any zeros which shouldn't be there
clean_columns = [GLUCOSE_COLUMN, BLOOD_PRESSURE_COLUMN, SKIN_THICKNESS_COLUMN, INSULIN_COLUMN, BMI_COLUMN, DIABETES_FUNC_COLUMN]
for clean_column in clean_columns:
    dataset[clean_column] = dataset[clean_column].replace(to_replace=0, value=dataset[clean_column].median())

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

# Save the scaler
joblib.dump(scaler, SCALER_FILE)

# Build the model
model = Sequential()
model.add(Dense(32, input_dim=8, activation='sigmoid', init='uniform'))
model.add(Dense(16, activation='sigmoid', init='uniform'))
model.add(Dense(1, activation='sigmoid', init='uniform'))

optimizer = SGD(lr=0.25, momentum=0.5, decay=1e-6)
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
mc = ModelCheckpoint(MODEL_FILE, monitor='acc', mode='max', verbose=1, save_best_only=True)

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)
model.fit(train_set_scaled, train_set_outcomes, nb_epoch=1000, verbose=1, callbacks=[mc, es])

# Evaluate training set
train_set_scores = model.evaluate(train_set_scaled, train_set_outcomes)
print("Train set %s: %.2f" % (model.metrics_names[0], train_set_scores[0]))
print("Train set %s: %.2f%%" % (model.metrics_names[1], train_set_scores[1] * 100))

# Evaluate test set
test_set_scores = model.evaluate(test_set_scaled, test_set_outcomes)
print("Test set %s: %.2f" % (model.metrics_names[0], test_set_scores[0]))
print("Test set %s: %.2f%%" % (model.metrics_names[1], test_set_scores[1] * 100))

# Load and test saved model with test set
while not os.path.exists(MODEL_FILE):
    time.sleep(1)

model = load_model(MODEL_FILE)
predicted_outcomes = model.predict_classes(test_set_scaled)
accuracy = accuracy_score(test_set_outcomes, predicted_outcomes)
print("Loaded model accuracy: %s" % (accuracy * 100))