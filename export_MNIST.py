import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np


"""
The algorithm will only work with the gz files created by this script.
"""

train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())


to_run_exporter = input("Would you like to export the data of the MNIST dataset? (1 + enter == yes): ")
if to_run_exporter == "1":
    train_arr = np.zeros(shape=(len(train_dataset), 1 + 28 * 28), dtype=np.float32)
    test_arr = np.zeros(shape=(len(test_dataset), 1 + 28 * 28), dtype=np.float32)
    for i in range(len(train_dataset)):
        train_arr[i, 0] = train_dataset[i][1]
        vec = np.reshape(train_dataset[i][0].numpy(), (1, 28 * 28))
        train_arr[i:i + 1, 1:28 * 28 + 1] = vec
    for i in range(len(test_dataset)):
        test_arr[i, 0] = test_dataset[i][1]
        vec = np.reshape(test_dataset[i][0].numpy(), (1, 28 * 28))
        test_arr[ i:i + 1, 1:28 * 28 + 1] = vec
    np.savetxt("MNIST_train.gz", train_arr, delimiter=",")
    np.savetxt("MNIST_test.gz", test_arr, delimiter=",")