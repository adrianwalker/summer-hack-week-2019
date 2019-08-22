import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

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
    DIABETES_FUNC_COLUMN, AGE_COLUMN, OUTCOME_COLUMN]

print(dataset.shape)
print(dataset.head())

dataset[GLUCOSE_COLUMN] = dataset[GLUCOSE_COLUMN].replace(to_replace=0, value=dataset[GLUCOSE_COLUMN].median())
dataset[BLOOD_PRESSURE_COLUMN] = dataset[BLOOD_PRESSURE_COLUMN].replace(to_replace=0,
                                                                        value=dataset[BLOOD_PRESSURE_COLUMN].median())
dataset[SKIN_THICKNESS_COLUMN] = dataset[SKIN_THICKNESS_COLUMN].replace(to_replace=0,
                                                                        value=dataset[SKIN_THICKNESS_COLUMN].median())
dataset[INSULIN_COLUMN] = dataset[INSULIN_COLUMN].replace(to_replace=0, value=dataset[INSULIN_COLUMN].median())
dataset[BMI_COLUMN] = dataset[BMI_COLUMN].replace(to_replace=0, value=dataset[BMI_COLUMN].median())
dataset[DIABETES_FUNC_COLUMN] = dataset[DIABETES_FUNC_COLUMN].replace(to_replace=0,
                                                                      value=dataset[DIABETES_FUNC_COLUMN].median())

# Split the training dataset 80% / 20%
train_set, test_set = train_test_split(dataset, test_size=0.2)

# Separate outcomes from the rest of the dataset
train_set_outcomes = train_set[OUTCOME_COLUMN].copy()
train_set = train_set.drop(OUTCOME_COLUMN, axis=1)
test_set_outcomes = test_set[OUTCOME_COLUMN].copy()
test_set = test_set.drop(OUTCOME_COLUMN, axis=1)

# Scale values
scaler = StandardScaler()
scaler.fit(train_set)
train_set_scaled = scaler.transform(train_set)
test_set_scaled = scaler.transform(test_set)

consensus = 2
models = [
    SVC(),
    LogisticRegression(),
    MLPClassifier(hidden_layer_sizes=(64, 32, 1))
]

predicted_outcomes = []

# Train the model
for model in models:

    model.fit(train_set_scaled, train_set_outcomes)

    # Predict the outcome for test dataset
    predicted_outcome = model.predict(test_set_scaled)
    predicted_outcomes.append(predicted_outcome)

predicted_outcomes = list(zip(*predicted_outcomes))
print(list(predicted_outcomes))
predicted_outcomes = list(map(lambda x: 1 if x.count(1) >= consensus else 0, predicted_outcomes))
print(list(predicted_outcomes))

accuracy = accuracy_score(test_set_outcomes, predicted_outcomes)
print(accuracy * 100)