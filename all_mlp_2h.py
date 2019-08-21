# https://www.kaggle.com/uciml/pima-indians-diabetes-database#diabetes.csv

import tensorflow as tf
import random
import statistics

# Column indexes
PREGNANCIES_COLUMN = 0
GLUCOSE_COLUMN = 1
BLOOD_PRESSURE_COLUMN = 2
SKIN_THICKNESS_COLUMN = 3
INSULIN_COLUMN = 4
BMI_COLUMN = 5
DIABETES_FUNC_COLUMN = 6
AGE_COLUMN = 7
OUTCOME_COLUMN=8

def column(lines, column, type):
    """
    get a column and convert to a type
    """
    return [type(line[column]) for line in lines]

def median_clean(l):
    """
    replace any zero values with median values
    """
    median = statistics.median(l)
    return [median if x == 0 else x for x in l]

def minmax_scaling(l):
    """
    MinMax scaling
    """
    xMin = min(l)
    xMax = max(l)
    return [(x - xMin) / (xMax - xMin) for x in l]

def cross_entropy(y, p):
    """
    cross entropy cost function
    """
    return tf.reduce_mean(-y * tf.log(p) - (1 - y) * tf.log(1 - p))

def mean_squared_error(y, p):
    """
    mean squared error cost function
    """
    return tf.reduce_mean(tf.square(tf.subtract(y, p)))

def dataset(lines):

    random.shuffle(lines)
    split_lines = [line.strip().split(',') for line in lines]

    scale = minmax_scaling
    clean = median_clean

    pregnancies_col = scale(column(split_lines, PREGNANCIES_COLUMN, float))
    glucose_col = scale(clean(column(split_lines, GLUCOSE_COLUMN, float)))
    blood_pressure_col = scale(clean(column(split_lines, BLOOD_PRESSURE_COLUMN, float)))
    skin_thickness_col = scale(clean(column(split_lines, SKIN_THICKNESS_COLUMN, float)))
    insulin_col = scale(clean(column(split_lines, INSULIN_COLUMN, float)))
    bmi_col = scale(clean(column(split_lines, BMI_COLUMN, float)))
    diabetes_func_col = scale(clean(column(split_lines, DIABETES_FUNC_COLUMN, float)))
    age_col = scale(column(split_lines, AGE_COLUMN, float))
    outcome_col = column(split_lines, OUTCOME_COLUMN, float)

    input_cols = list(map(list, zip(pregnancies_col, glucose_col, blood_pressure_col, skin_thickness_col, insulin_col,
                                    bmi_col, diabetes_func_col, age_col)))
    output_cols = list(map(list, zip(outcome_col)))

    return input_cols, output_cols

# load file
csv = open("./pima-indians-diabetes.csv", 'r')
lines = csv.readlines()

# Dataset
(input_data, output_data) = dataset(lines)
head = int(len(input_data) * .8) # 80%
tail = int(len(input_data) * .2) # 20%
x_data = input_data[0: head]
y_data = output_data[0: head]
x_data_test = input_data[-tail:]
y_data_test = output_data[-tail:]

# Network Parameters
n_input = len(input_data[0])
n_hidden_1 = 4 * n_input
n_hidden_2 = 4 * n_input
n_output = len(output_data[0])

learning_rate = 0.95
epochs = 10000
print_epoch = 100

# Placeholders
X = tf.placeholder(tf.float32) # input
Y = tf.placeholder(tf.float32) # output

# Weights
W1 = tf.Variable(tf.random_uniform([n_input, n_hidden_1], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([n_hidden_1, n_hidden_2], -1.0, 1.0))
W3 = tf.Variable(tf.random_uniform([n_hidden_2, n_output], -1.0, 1.0))

# Bias
B1 = tf.Variable(tf.zeros([n_hidden_1]))
B2 = tf.Variable(tf.zeros([n_hidden_2]))
B3 = tf.Variable(tf.zeros([n_output]))

# The activation function of a node defines the output of that node given an input or set of inputs. Only nonlinear
# activation functions allow networks to compute nontrivial problems using only a small number of nodes.

# Layers with Sigmoid function outputs
L1 = tf.sigmoid(tf.matmul(X, W1) + B1)
L2 = tf.sigmoid(tf.matmul(L1, W2) + B2)
L3 = tf.sigmoid(tf.matmul(L2, W3) + B3)

# A cost function is a measure of "how good" a neural network did with respect to it's given training sample and the
# expected output. It also may depend on variables such as weights and biases.

# Cross-entropy cost function
# cost_func = cross_entropy(Y, L3)

# Mean squared error cost function
cost_func = mean_squared_error(Y, L3)

# The update rules are determined by the Optimizer. The performance and update speed may heavily vary from optimizer
# to optimizer. The gradient tells us the update direction, but it is still unclear how big of a step we might take.
# Short steps keep us on track, but it might take a very long time until we reach a (local) minimum. Large steps speed
# up the process, but it might push us off the right direction.

# Adadelta Optimizer
# optimizer = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost_func)

# Gradient Descent Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_func)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)

    # train
    for epoch in range(epochs):
        _, cost = session.run([optimizer, cost_func], feed_dict={X: x_data, Y: y_data})

        if epoch % print_epoch == 0:
            print("epoch: %s, cost: %s" % (epoch, cost))

    # round outputs and calculate accuracy
    prediction = tf.equal(tf.round(L3), Y)
    accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
    result = accuracy.eval({X: x_data_test, Y: y_data_test})
    print("accuracy: %s" % (result * 100))
