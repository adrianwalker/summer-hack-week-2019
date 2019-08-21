# https://www.kaggle.com/uciml/pima-indians-diabetes-database#diabetes.csv

import tensorflow as tf
import random

# Column indexes

PREGNANCIES_COLUMN = 0
GLUCOSE_COLUMN = 1
BLOOD_PRESSURE_COLUMN = 2
SKIN_THICKNESS_COLUMN = 3
INSULIN_COLUMN = 4
BMI_COLUMN = 5
DIABETES_PED_FUNC_COLUMN = 6
AGE_COLUMN = 7
OUTCOME_COLUMN=8

# load file
csv = open("./pima-indians-diabetes.csv", 'r')
lines = csv.readlines()
random.shuffle(lines)
lines = [line.strip().split(',') for line in lines]

def scale(l):
    """
    MinMax scaling
    """
    xMin = min(l)
    xMax = max(l)
    return [(x - xMin) / (xMax - xMin) for x in l]

def column(lines, column, type):
    """
    get a column and convert to a type
    """
    return [type(line[column]) for line in lines]

bmis = scale(column(lines, BMI_COLUMN, float))
ages = scale(column(lines, AGE_COLUMN, float))
outcomes = scale(column(lines, OUTCOME_COLUMN, int))

input_data = list(zip(bmis, ages))
output_data = list(zip(outcomes))

print(input_data)
print(output_data)

# Dataset
head = int(len(input_data) * 0.8) # 80%
tail = int(len(input_data) * 0.2) # 20%
x_data = input_data[0: head]
y_data = output_data[0: head]
x_data_test = input_data[-tail:]
y_data_test = output_data[-tail:]

# Paramters
n_input = len(input_data[0])
n_hidden = 2 * n_input
n_output = 1
learning_rate = 0.1
epochs = 1000
print_epoch = 100

# Placeholders
X = tf.placeholder(tf.float32) # input
Y = tf.placeholder(tf.float32) # output

# Weights
W1 = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([n_hidden, n_output], -1.0, 1.0))

# Bias
B1 = tf.Variable(tf.zeros([n_hidden]))
B2 = tf.Variable(tf.zeros([n_output]))

# The activation function of a node defines the output of that node given an input or set of inputs. Only nonlinear
# activation functions allow networks to compute nontrivial problems using only a small number of nodes.

# Sigmoid functions
F1 = tf.sigmoid(tf.matmul(X, W1) + B1)
F2 = tf.sigmoid(tf.matmul(F1, W2) + B2)

# A cost function is a measure of "how good" a neural network did with respect to it's given training sample and the
# expected output. It also may depend on variables such as weights and biases.

# Cross-entropy cost function
cost_func = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=F2))

# The update rules are determined by the Optimizer. The performance and update speed may heavily vary from optimizer
# to optimizer. The gradient tells us the update direction, but it is still unclear how big of a step we might take.
# Short steps keep us on track, but it might take a very long time until we reach a (local) minimum. Large steps speed
# up the process, but it might push us off the right direction.

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

    # test
    outputs = session.run([F2], feed_dict={X: x_data_test, Y: y_data_test})
    print(outputs)

    # round outputs and calculate accuracy
    prediction = tf.equal(tf.round(F2), Y)
    accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
    result = accuracy.eval({X: x_data_test, Y: y_data_test})
    print(result * 100)
