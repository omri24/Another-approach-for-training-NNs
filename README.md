# Another-approach-for-training-NNs

## Introduction

Artificial neural networks (NN) that are trained using one of the gradient descent (GD) algorithms have a significant downside- it is almost impossible to explain why such a NN decides one way or another.

In this repository I will depict my attempts to find a replacement for the GD algorithms. Such an algorithm may create a NN that is not as accurate as a NN that was trained using a GD algorithm. However, an algorithm like that, will allow us to explain the reasoning behind the decisions of the NN.

The problem the NN will face will be a handwriting classification problem. More specifically, digits identification, using the MNIST dataset. However, I will make the “fight” more interesting by significantly shrinking the training dataset. The MNIST dataset contains 60,000 training samples. I will allow the algorithms to learn from datasets no bigger than 100 samples.The testing samples will be chosen randomly. The MNIST testing dataset will not be reduced.

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
Not much to say, using a GD algorithm, the predictions of the NN are much better.

## Downside of manually choosing the weights 
Changing the training dataset may cause a significant change in the success rate, even if the training sets share the same amount of training samples:
Note that this problem is not apparent in the GD algorithm- success rates don't change dramatically when changing the training sets. 
About the pytorch seed:
if you are think that the conclusion above might be true only for the chosen pytorch seed of zero, I will let you know that this conclusion holds for the following seeds as well:
Seed = 6 -> results = [84.74, 87.84, 87.81, 88.47, 88.2, 88.99, 88.67, 89.22, 89.02, 89.4]
Seed = 986 -> results = [86.1, 86.78, 88.03, 88.07, 88.75, 89.05, 88.79, 89.09, 88.98, 89.32]
Seed = 7458-> results = [85.77, 87.3, 87.8, 87.15, 88.24, 88.48, 88.88, 88.94, 89.21, 89.51]


## conclusions
Training a NN with a GD algorithm achieves much better results than choosing the weights using any of the methods above, with any enhancement. The GD algorithm creates a more stable NN in the sense that the accuracy level of the trained NN is less sensitive to the act of changing a few samples of the training dataset.

## Appendix

