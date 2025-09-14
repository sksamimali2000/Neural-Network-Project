# Simple Neural Network Implementation in Python (XOR Problem)

## Overview
This project demonstrates a basic neural network implemented from scratch using NumPy. It shows both forward and backward propagation without using any deep learning frameworks. The neural network learns to solve the XOR problem.

---

## Data Preparation
```python
import numpy as np

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0,0,0,1]]).T
Sigmoid Activation and Its Derivative
```

```Python
def sig(z):
    return 1 / (1 + np.exp(-z))

def derivativeSig(z):
    return sig(z) * (1 - sig(z))
Training Without Hidden Layer (Initial Forward Pass)


```Python
Copy code
weights = 2 * np.random.random((2, 1)) - 1
bias = 2 * np.random.random(1) - 1

output0 = X
output = sig(np.dot(output0, weights) + bias)
Training With One Hidden Layer (Forward Propagation Example)
```
```Python
wh = 2 * np.random.random((2, 2)) - 1
bh = 2 * np.random.random((1, 2)) - 1
wo = 2 * np.random.random((2, 1)) - 1
bo = 2 * np.random.random((1, 1)) - 1

outputHidden = sig(np.dot(output0, wh) + bh)
output = sig(np.dot(outputHidden, wo) + bo)
```


Training Single Layer Neural Network (Gradient Descent)
```Python
lr = 0.1
for iter in range(10000):
    output0 = X
    output = sig(np.dot(output0, weights) + bias)

    first_term = output - Y
    input_for_last_layer = np.dot(output0, weights) + bias
    second_term = derivativeSig(input_for_last_layer)
    first_two = first_term * second_term

    changes = np.dot(output0.T, first_two)
    weights = weights - lr * changes
    bias_change = np.sum(first_two)
    bias = bias - lr * bias_change
```


Final Output After Training
```Python
output = sig(np.dot(X, weights) + bias)
weights, bias, output
```

Conclusion
This project demonstrates how to implement a simple neural network to solve the XOR problem using only NumPy. It covers:

Sigmoid activation function

Forward propagation

Backpropagation and gradient descent

Learning without any deep learning libraries

The final output shows the network learning to map input combinations to their expected XOR output.
