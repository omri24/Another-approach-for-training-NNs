# Another-approach-for-training-NNs

## Introduction

Artificial neural networks (NN) that are trained using one of the gradient descent (GD) algorithms have a significant downside- it is almost impossible to explain why such a NN decides one way or another.

In this repository I will depict my attempt to find a replacement for the GD algorithms for handwriting classifying NN. Such an algorithm may create a NN that is not as accurate as a NN that was trained using a GD algorithm. However, an algorithm like that, will allow us to explain the reasoning behind the decisions of the NN.

The problem the NN will face will be a handwriting classification problem. More specifically, digits identification, using the MNIST dataset. However, I will make the “fight” more interesting by significantly shrinking the training dataset. The MNIST dataset contains 60,000 training samples. I will allow the algorithms to learn from datasets no bigger than 100 samples.The testing samples will be chosen randomly. The MNIST testing dataset will not be reduced.

The NN is linear, with no hidden layers.

The non- GD algorithms use gz files that can be created by the export_MNIST script above. The gz files are too heavy to be uploaded.

## Assumptions about the data

1. All data samples are images showing a digit in the range 0- 9.
2. All data samples are images with dimensions 28X28.
3. Each type of digit has some pixels that are very likely to have ink inside, and some that are very unlikely to have ink inside. Here “having ink inside” means that the value of the corresponding cell is positive, and “not having ink inside” means that the value of the cell is zero.

All the assumptions except 3, are trivial for the MNIST dataset. 3 isn’t trivial but it makes sense. 

## method 1 for calculating the NN parameters- ink focusing

This method uses assumption 3. The logic behind this method:
1. Locate the pixels that are positive (“have ink inside”) in a high probability for each digit. This step of the process is done with the training dataset: shift all the samples to binary images (all positive values ->1, zeros aren’t changed), sum the columns (each column represents a pixel), divided each column by its sum and now each cell have inside a probability: given this pixel (database column) is “with ink”, what is the probability for a certain digit (database row).
2. For each digit, set the weights of the NN that connect those “high probability pixels” to the output representing the digit to 1. Any other weight- set to 0.
3. Pass the testing data samples through the NN and classify each sample according to the maximal output of the NN.
Note that high probability is not an accurate thing, and therefore a few probability thresholds will be compared.

## method 2 for calculating the NN parameters- background focusing

Method 2 is similar to method 1 with a slight different:
1. Locate the pixels that are zero (“doesn’t have ink inside”) in a high probability for each digit. This step of the process is done with the training dataset, the process is very similar to the process in method 1.
2. For each digit, set the weights of the NN that connect those “high probability pixels” to the output representing the digit to 1. Any other weight- set to 0.
3. Pass the testing data samples through the NN and classify each sample according to the minimal output of the NN.

## Attempts for enhancement

1. Dividing the values of the weights associated with a certain digit by the sum of all the weights associated with the digit. The logic behind this step is that some digits have in average more pixels “with ink” than the others and therefore a bias might be created in favor/against those digits.
2. Changing part 1 of both methods: skipping the part in which the training samples are changed to binary samples.
3. Changing all testing samples to binary samples before passing through the NN.

## Results- summary

Maximal success rate is 56.84%. It is achieved by methods 1 and 2, for a probability parameter of 0.05 and without using enhancement attempt number 1. Full results in the appendix.

## GD algorithm wins 

![accuracy- seed 0](https://github.com/omri24/Another-approach-for-training-NNs/assets/115406253/01414fa7-3bf8-4ab2-9d02-dedb33000940)

Not much to say, using a GD algorithm, the predictions of the NN are much better.

## Downside of manually choosing the weights 
Changing the training dataset may cause a significant change in the success rate, even if the training sets share the same amount of training samples:

![70 samples bad results](https://github.com/omri24/Another-approach-for-training-NNs/assets/115406253/4507d65b-f23b-4629-8a8b-0eb3f3442cb3)

![70 samples good results](https://github.com/omri24/Another-approach-for-training-NNs/assets/115406253/98f8352c-32a0-40a9-9fbd-ba0d8f640053)

Note that this problem is not apparent in the GD algorithm- success rates don't change dramatically when changing the training sets. 
This is also true for pytorch seeds other than 0:

Seed = 6 -> success rates = [84.74, 87.84, 87.81, 88.47, 88.2, 88.99, 88.67, 89.22, 89.02, 89.4]

Seed = 986 -> success rates = [86.1, 86.78, 88.03, 88.07, 88.75, 89.05, 88.79, 89.09, 88.98, 89.32]

Seed = 7458-> success rates = [85.77, 87.3, 87.8, 87.15, 88.24, 88.48, 88.88, 88.94, 89.21, 89.51]


## conclusions
Training a NN with a GD algorithm achieves much better results than choosing the weights using any of the methods above, with any enhancement. The GD algorithm creates a more stable NN in the sense that the accuracy level of the trained NN is less sensitive to the act of changing a few samples of the training dataset.

## Appendix

![m1](https://github.com/omri24/Another-approach-for-training-NNs/assets/115406253/acac576c-0677-4412-a62b-c0b4a67d17bc)
![m2](https://github.com/omri24/Another-approach-for-training-NNs/assets/115406253/d2a177f0-7852-461d-9db9-22a8e8694a9a)
![m3](https://github.com/omri24/Another-approach-for-training-NNs/assets/115406253/64466c0e-fd85-45b2-a553-4bd7cbe0234b)
![m4](https://github.com/omri24/Another-approach-for-training-NNs/assets/115406253/a2e40631-92e6-4c9e-ab54-d6c0ddd3cde0)
![m5](https://github.com/omri24/Another-approach-for-training-NNs/assets/115406253/affd8e6a-0a47-4da7-9898-81f757621e9d)
![m6](https://github.com/omri24/Another-approach-for-training-NNs/assets/115406253/123e6830-896a-4098-9464-db4e0cf841f3)
![m7](https://github.com/omri24/Another-approach-for-training-NNs/assets/115406253/34c96fe4-84e3-4551-8f02-f5a1e5990df8)


