import sys
import numpy as np
from deepcar import NN

np.random.seed()
x = np.asarray([1, 2])
w = np.asarray([2, 3, 4, 5, 6, 7]).reshape((2, 3))
print(type(x))
print(x)
print(x.shape)

print(np.dot(x, w))

print(np.append(x, [1]))

nn = NN(5, 2, [10, 10])
xs = np.random.normal(size=5).tolist()
print(xs)
ys = nn.forward(xs)
print(ys)