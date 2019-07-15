from typing import List
import numpy as np
from itertools import accumulate
import random
import math

class NN:
    def __init__(self, x_cnt:int, y_cnt:int, hidden_cnts:List[int]):
        self.x_cnt = x_cnt
        self.y_cnt = y_cnt
        self.hidden_cnts = hidden_cnts

        if len(hidden_cnts) < 1:
            raise Exception('Need to have at least 1 hidden layer.')
            
        self.hiddens = []

    def randomize_weights_biases(self, seed=None):
        if seed:
            np.random.seed(seed)
        else:
            np.random.seed()
        random_scale = 1/self.x_cnt**.5
        layer = np.random.normal(scale=random_scale, size=(self.x_cnt+1, self.hidden_cnts[0]))
        self.hiddens.append(layer)
        for i in range(1, len(self.hidden_cnts)):
            layer = np.random.normal(scale=random_scale, size=(self.hidden_cnts[i-1]+1, self.hidden_cnts[i]))
            self.hiddens.append(layer)
        layer = np.random.normal(scale=random_scale, size=(self.hidden_cnts[-1]+1, self.y_cnt))
        self.hiddens.append(layer)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def forward(self, xs:List[float])->List[float]:
        if len(xs) != self.x_cnt:
            raise Exception(f'This NN only accepts input of size {self.x_cnt}')
        output = np.asarray(xs)
        for hidden in self.hiddens:
            output = np.append(output, [1])
            output = NN.sigmoid(np.dot(output, hidden))
        output = NN.softmax(output)
        return output.tolist()

    @staticmethod
    def softmax(x):
        e_power = np.power(math.e, x)
        denominator = e_power.sum()
        return e_power / denominator

    def deep_copy(self):
        '''
        Return a deep copy of self
        '''
        copy = NN(self.x_cnt, self.y_cnt, list(self.hidden_cnts))
        copy.hiddens = [np.copy(h) for h in self.hiddens]
        return copy

    def mutate_randomly(self):
        '''
        Randomly mutate weights and biases, inplace.
        '''
        probs = [0.02] * 4
        mutation_functions = [NN.flip, NN.rand, NN.rand_increase_pct, NN.rand_deduct_pct]
        for hidden in self.hiddens:
            NN._mutate_nparray_with_probs(hidden, probs, mutation_functions)

    @staticmethod
    def _mutate_nparray_with_probs(a, probs:List[float], mutation_functions, seed=None):
        '''
        Similar to mutate_with_probs, but do it on each element of a numpy array. Notice that at each element, the probability is recalculated.

        Method is inplace
        '''
        with np.nditer(a, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = NN.mutate_with_probs(x, probs, mutation_functions, seed)

    @staticmethod
    def mutate_with_probs(x, probs:List[float], mutation_functions, seed=None):
        '''
        Apply a mutation function on x according to the function's probability

        INPUT
            x (float): the number to apply mutation function on
            probs (List[float]): list of propbabilities of mutation_functions. 
                Has to have the same length with list of mutation_functions. If sum of probs < 1, return x with the probability of 1 - sum(probs)
            mutation_functions(List[functions]): list of functions that take a float as an argument and return a float.
        OUTPUT
            float: mutated of input
        '''
        n = len(probs)
        if n != len(mutation_functions):
            raise Exception('len(probs) != len(mutation_functions)')

        random.seed(seed)
        for i, p in enumerate(probs):
            r = random.random()
            if r <= p:
                x = mutation_functions[i](x)
        return x

    flip = lambda x:-x
    rand = lambda x:x - x + random.random() # not redundant, use this to apply on np.array
    rand_increase_pct = lambda x:x*(1 + random.random())
    rand_deduct_pct = lambda x:x*random.random()

def _test_forward_no_exception():
    nn = NN(5, 2, [10, 10])
    nn.randomize_weights_biases(42)
    xs = np.random.normal(size=5).tolist()
    ys = nn.forward(xs)
    print(ys)

def _test_mutate_with_probs():
    input = [1] * 10
    mutate = lambda x: NN.mutate_with_probs(x, [0.2] * 4, [NN.flip, NN.rand, NN.rand_increase_pct, NN.rand_deduct_pct])
    output = map(mutate, input)
    print(list(output))

def _test_mutate_nparray_with_probs():
    a = np.array([float(x) for x in range(20)]).reshape((4, 5))
    NN._mutate_nparray_with_probs(a, [0.2] * 4, [NN.flip, NN.rand, NN.rand_increase_pct, NN.rand_deduct_pct])
    print(a)

if __name__ == '__main__':
    _test_forward_no_exception()
    #_test_mutate_with_probs()
