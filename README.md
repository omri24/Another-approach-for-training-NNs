# Another-approach-for-training-NNs

## The most important things- in 6 bullet points:

1. Artificial neural networks (NN) that are trained using one of the gradient descent (GD) algorithms have a significant downside- it is almost impossible to explain why such a NN decides one way or another.

2. This repository depicts an attempt to examine a few methods for choosing the weights without using a GD algorithm. Such methods can be useful if for a certain application, it is necessary to be able to easily explain the logic behind the decision of the NN.

3. The problem that the NN will face is handwritten digits classification (from the MNIST dataset).

4. The NN will be trained using the GD algorithm, and using its replacements, on training datasets of not more than 100 training samples (if we would use all the 60,000 training samples this comparison will probably be less interesting).

5. The results: without using a GD algorithm, the maximal accuracy achieved by the NN is 56.84%. The GD on the other hand creates a NN with accuracy level of more than 85%. GD wins.

6. The full information appears in the PDF file.

