import numpy as np

# Datasets:
train_arr = np.loadtxt("MNIST_train.gz", delimiter=",", dtype=np.float32)
test_arr = np.loadtxt("MNIST_test.gz", delimiter=",", dtype=np.float32)

# Parameters


v_lst = [i * 0.05 for i in range(1,20)]
p_lst = [6000, 3000, 2000, 1500, 1200, 1000, 858, 750, 667, 600]
lst_o_lst = []
for v in v_lst:
    param_for_background = v
    param_for_ink = v
    acc_lst = [v]
    num_samples_lst = []
    for p in p_lst:
        modulu_param = p  # this parameter depletes the training dataset

        # Training:
        training_samples = 0
        digit_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        database = np.zeros((10, 28 * 28), dtype=np.float32)
        binary_database = "6"                                                     # param that can be changed
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
        for i in range(sum_columns.shape[1]):
            if sum_columns[0, i] == 0:
                sum_columns[0, i] = 0.001
        for i in range(database.shape[0]):
            database[i:i + 1, :] = database[i:i + 1, :] / sum_columns
        activation_mode = "1"                                                        # param that can be changed
        if activation_mode not in ["1", "2"]:
            raise ValueError("activation_mode must be 1 or 2")
        if activation_mode == "1":
            database = database < param_for_background
        if activation_mode == "2":
            database = database > param_for_ink
        div_by_row_sum = "6"                                                          # param that can be changed
        if div_by_row_sum == "1":
            sum_rows = np.sum(database * 1, axis=1, keepdims=True)
            for i in range(sum_rows.shape[0]):
                if sum_rows[i, 0] == 0:
                    sum_rows[i, 0] = 9
            for i in range(database.shape[1]):
                database[:, i:i + 1] = database[:, i:i + 1] / sum_rows
        if div_by_row_sum != "1":
            database = database * 1

        # Testing:
        algo_outputs = []
        real_digits = []
        binary_test = "6"                                                        # param that can be changed
        correct = 0
        for i in range(test_arr.shape[0]):
            digit, vec = int(test_arr[i][0]), np.reshape(test_arr[i][1:], (1, 28 * 28))
            if binary_test == "1":
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

        # print(algo_outputs)
        # print(real_digits)
        # print(digit_dict)
        print("\n")
        print("Probability param: " + str(v))
        print("Training samples: " + str(training_samples))
        print("Samples for each digit:")
        print(digit_dict)
        print("Correct answers: " + str(correct))
        print("Testing samples: " + str(test_arr.shape[0]))
        print("The percentage of correct answers: " + str((correct / test_arr.shape[0]) * 100))
        acc_lst += [(correct / test_arr.shape[0]) * 100]
        num_samples_lst += [training_samples]
    lst_o_lst += [acc_lst]

print(acc_lst)
print(num_samples_lst)
to_export = input("Would you like to save the results as CSV? (1 + enter -> yes): ")
if to_export == "1":
    np.savetxt("back_no_div_all_non_bin.csv", np.array(lst_o_lst, dtype=np.float32), delimiter=",")


