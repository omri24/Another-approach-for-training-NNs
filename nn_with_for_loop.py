import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time

# Preparations
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
input_size = 784
num_classes = 10
num_epochs = 300
learning_rate = 0.001

# Datasets
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='./data',
                                          train=False,
                                          transform=transforms.ToTensor())

# FNN structure
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        return out


model = NeuralNet(input_size=input_size, num_classes=num_classes).to(device)
batch_lst = [i * 10 for i in range(1,11)]
acc_lst = []
training_lst = []
times_lst = []

for p in batch_lst:
    l1_np = np.loadtxt("lin1.csv", delimiter=",", dtype=np.float32)
    model.l1.weight = nn.parameter.Parameter(torch.from_numpy(l1_np))
    batch_size = p
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    # Training
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            if i != 0:
                break
            elif i == 0:
                images = images.reshape(-1, 28 * 28).to(device)
                labels = labels.to(device)
                # Forward pass
                if epoch == 0:
                    print("number of training samples: " + str(len(images)))
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # Testing
    timer_start = time.time()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
        acc = 100.0 * n_correct / n_samples
        print("total test samples = " + str(n_samples) + ", correct answers = " + str(n_correct) +
                  ", accuracy = " + str(acc) + "%")
    acc_lst += [acc]
    training_lst += [batch_size]
    timer_end = time.time()
    calc_time = timer_end - timer_start
    print("time to calculate results for all 10,000 testing samples: " + str(calc_time))
    times_lst += [calc_time]

print("\n")
print(acc_lst)
print(training_lst)
print(times_lst)
plt.scatter(training_lst, acc_lst)
plt.xlabel("Training samples")
plt.ylabel("Accuracy %")
plt.grid()
plt.title("Accuracy of a NN trained with a GD algorithm")
plt.show()

to_save_as_csv = input("Would you like to save the results as CSV? (1 + enter -> yes): ")
if to_save_as_csv == "1":
    np.savetxt("accuracy.csv", np.array(acc_lst, dtype=np.float32), delimiter=",")
    np.savetxt("samples_amount.csv", np.array(training_lst, dtype=np.float32), delimiter=",")
    np.savetxt("times.csv", np.array(times_lst, dtype=np.float32), delimiter=",")