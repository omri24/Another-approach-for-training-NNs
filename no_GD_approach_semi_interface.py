import numpy as np
import time

# Datasets (the gz files are too big and can't be uploaded to Github- use the export_MNIST module to create them):
train_arr = np.loadtxt("MNIST_train.gz", delimiter=",", dtype=np.float32)
test_arr = np.loadtxt("MNIST_test.gz", delimiter=",", dtype=np.float32)

# Parameters
param_for_background = 0.01
param_for_ink = 0.99
modulu_param = 857  # this parameter depletes the training dataset

# Training:
training_samples = 0
digit_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
database = np.zeros((10, 28 * 28), dtype=np.float32)
binary_database = input("Would you like to train using binary images? (1 + enter -> yes): ")
for i in range(train_arr.shape[0]):
    if i % modulu_param == 0:
        training_samples += 1
        digit, vec = int(train_arr[i][0]), np.reshape(train_arr[i][1:], (1, 28 * 28))
        digit_dict[digit] += 1
        if binary_database == "1":
            mask = vec != 0
        if binary_database != "1":
            mask = vec
        database[digit:digit + 1, :] += 1 * mask
sum_columns = np.sum(database, axis=0, keepdims=True)
for i in range(sum_columns.shape[1]):   # this for loop is to prevent division by 0
    if sum_columns[0, i] == 0:
        sum_columns[0, i] = 0.001
for i in range(database.shape[0]):
    database[i:i + 1, :] = database[i:i + 1, :] / sum_columns
q = "Would you like to active focusing background (recommended) or ink? (1 + enter -> background, 2 + enter -> ink): "
activation_mode = input(q)
if activation_mode not in ["1", "2"]:
    raise ValueError("activation_mode must be 1 or 2")
if activation_mode == "1":
    database = database < param_for_background
if activation_mode == "2":
    database = database > param_for_ink
print("Choosing the following option may improve the results but will increase the time of the calculations")
div_by_row_sum = input("Would you like to divide by the row sum? (1 + enter -> yes): ")
if div_by_row_sum == "1":
    sum_rows = np.sum(database * 1, axis=1, keepdims=True)
    for i in range(sum_rows.shape[0]):  # this for loop is to prevent division by 0
        if sum_rows[i, 0] == 0:
            sum_rows[i, 0] = 0.001
    for i in range(database.shape[1]):
        database[:, i:i + 1] = database[:, i:i + 1] / sum_rows
if div_by_row_sum != "1":
    database = database * 1

# Testing:
algo_outputs = []
real_digits = []
correct = 0
binary_test_samples = input("Would you like to test using binary images? (1 + enter -> yes): ")
timer_start = time.time()
for i in range(test_arr.shape[0]):
    digit, vec = int(test_arr[i][0]), np.reshape(test_arr[i][1:], (1, 28 * 28))
    if binary_test_samples == "1":
        vec = (vec != 0) * 1
    if activation_mode == "1":
        min_dot = 10 ** 100
        digit_memo = -1
    if activation_mode == "2":
        max_dot = -1
        digit_memo = -1
    for j in range(database.shape[0]):
        dot = np.sum(vec * database[j:j+1, :])
        if activation_mode == "1":
            if dot < min_dot:
                min_dot = dot
                digit_memo = j
        if activation_mode == "2":
            if dot > max_dot:
                max_dot = dot
                digit_memo = j
    algo_outputs += [digit_memo]
    real_digits += [digit]
    if digit == digit_memo:
        correct += 1
timer_end = time.time()
calc_time = timer_end - timer_start

# Showing the results
print("Outputs of the algorithm:")
print(algo_outputs)
print("The digits that were actually written:")
print(real_digits)
print("\nTime that took for testing all the 10,000 MNIST samples: " + str(calc_time))
print("Without NN:")
print("Training samples: " + str(training_samples))
print("Samples for each digit:")
print(digit_dict)
print("Correct answers: " + str(correct))
print("Testing samples: " + str(test_arr.shape[0]))
print("The percentage of correct answers: " + str((correct / test_arr.shape[0]) * 100))
