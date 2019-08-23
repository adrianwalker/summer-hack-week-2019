
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from statistics import mean, variance


results = []

for _ in range(0, 500):
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

    dataset[GLUCOSE_COLUMN] = dataset[GLUCOSE_COLUMN].replace(
        to_replace=0, value=dataset[GLUCOSE_COLUMN].median())
    dataset[BLOOD_PRESSURE_COLUMN] = dataset[BLOOD_PRESSURE_COLUMN].replace(to_replace=0,
                                                                            value=dataset[BLOOD_PRESSURE_COLUMN].median())
    dataset[SKIN_THICKNESS_COLUMN] = dataset[SKIN_THICKNESS_COLUMN].replace(to_replace=0,
                                                                            value=dataset[SKIN_THICKNESS_COLUMN].median())
    dataset[INSULIN_COLUMN] = dataset[INSULIN_COLUMN].replace(
        to_replace=0, value=dataset[INSULIN_COLUMN].median())
    dataset[BMI_COLUMN] = dataset[BMI_COLUMN].replace(
        to_replace=0, value=dataset[BMI_COLUMN].median())
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
    scaler = MinMaxScaler()
    scaler.fit(train_set)
    train_set_scaled = scaler.transform(train_set)
    test_set_scaled = scaler.transform(test_set)

    determine_best_params = False
    if determine_best_params:
        # Determine best parameters for an SVC...
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'shrinking': [True, False],
            'gamma': ['auto', 10, 1, 0.1],
            'coef0': [0.0, 0.1, 0.5]
        }

        model_svc = SVC()
        grid_search = GridSearchCV(
            model_svc, param_grid, cv=10, scoring='accuracy')
        grid_search.fit(train_set_scaled, train_set_outcomes)
        svc = grid_search.best_estimator_
        print(svc)
    else:
        # ... or just create one
        svc = SVC(C=10, gamma='auto')

    # Train the model
    svc.fit(train_set_scaled, train_set_outcomes)

    # Predict the outcome for test dataset
    predicted_outcomes = svc.predict(test_set_scaled)
    accuracy = accuracy_score(test_set_outcomes, predicted_outcomes)

    print(accuracy * 100)
    results.append(accuracy * 100)
    # Test some inputs
    test_input = pd.DataFrame([[6, 168, 72, 35, 0, 43.6, 0.627, 65]])
    test_input_scaled = scaler.transform(test_input)
    predicted_outcome = svc.predict(test_input_scaled)

    print(predicted_outcome)


print(mean(results))
print(variance(results))
