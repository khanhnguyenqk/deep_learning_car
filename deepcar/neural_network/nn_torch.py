from typing import List
import numpy as np
from itertools import accumulate
import random
import math
import torch
from torch import nn
import torch.nn.functional as F

DEVICE = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print(f'pytorch device: {DEVICE}')

class NNTorch:
    def __init__(self, x_cnt:int, y_cnt:int, hidden_cnts:List[int]):
        self.x_cnt = x_cnt
        self.y_cnt = y_cnt
        self.hidden_cnts = hidden_cnts

        if len(hidden_cnts) < 1:
            raise Exception('Need to have at least 1 hidden layer.')
            
        self.model = self._create_model()

    def _create_model(self):
        cnts = [self.x_cnt] + self.hidden_cnts + [self.y_cnt]

        modules = []
        for i in range(1, len(cnts)):
            modules.append(nn.Linear(cnts[i-1], cnts[i]))
            if i != len(cnts) - 1:
                modules.append(nn.Sigmoid())

        modules.append(nn.Softmax(dim=0))
        model = nn.Sequential(*modules)
        model.to(DEVICE)
        return model

    def randomize_weights_biases(self, seed=None):
        if seed:
            torch.manual_seed(seed)

        std = 1/self.x_cnt**.5
        
        
        for module in self.model:
            if isinstance(module, nn.Linear):
                module.bias.data.fill_(0)
                module.weight.data.normal_(std=std)
        self.model.eval()

    def forward(self, xs:List[float])->List[float]:
        if len(xs) != self.x_cnt:
            raise Exception(f'This NN only accepts input of size {self.x_cnt}')
        input = torch.from_numpy(np.asarray(xs)).to(DEVICE)
        with torch.no_grad():
            output = self.model.forward(input.float())
        output = output.to('cpu')
        return output.tolist()

    def deep_copy(self):
        '''
        Return a deep copy of self
        '''
        copy = NNTorch(self.x_cnt, self.y_cnt, list(self.hidden_cnts))
        copy.model.load_state_dict(self.model.state_dict())
        return copy

    def mutate_randomly(self, intensity=1, min_prob=0.02, max_prob=0.06):
        '''
        Randomly mutate weights and biases, inplace.
        
        INPUT
            intensity (float): in range [0, 1]. Mutation prob = min_prob + intensity * (max_prob - min_prob)
        '''
        probs = [min_prob + intensity * (max_prob - min_prob)] * 4
        mutation_functions = [NNTorch.flip, NNTorch.rand, NNTorch.rand_increase_pct, NNTorch.rand_deduct_pct]

        self.model.to('cpu')
        for module in self.model:
            if isinstance(module, nn.Linear):
                array = module.weight.data.numpy()
                NNTorch._mutate_nparray_with_probs(array, probs, mutation_functions)
                module.weight.data = torch.from_numpy(array)

                
                array = module.bias.data.numpy()
                NNTorch._mutate_nparray_with_probs(array, probs, mutation_functions)
                module.bias.data = torch.from_numpy(array)
        self.model.to(DEVICE)

    @staticmethod
    def _mutate_nparray_with_probs(a, probs:List[float], mutation_functions, seed=None):
        '''
        Similar to mutate_with_probs, but do it on each element of a numpy array. Notice that at each element, the probability is recalculated.

        Method is inplace
        '''
        with np.nditer(a, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = NNTorch.mutate_with_probs(x, probs, mutation_functions, seed)

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
    nn = NNTorch(5, 2, [10, 10])
    nn.randomize_weights_biases(42)
    xs = np.random.normal(size=5).tolist()
    ys = nn.forward(xs)
    print(ys)

def _test_deep_copy():
    nn = NNTorch(5, 2, [10, 10])
    copy = nn.deep_copy()
    nn.randomize_weights_biases(42)
    print(nn.model[0].weight.data)
    print(copy.model[0].weight.data)

if __name__ == '__main__':
    #_test_deep_copy()
    _test_forward_no_exception()
    #_test_mutate_with_probs()
