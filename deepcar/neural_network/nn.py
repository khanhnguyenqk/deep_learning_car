from typing import List
import numpy as np

class NN:
    def __init__(self, x_cnt:int, y_cnt:int, hidden_cnts:List[int]):
        self.x_cnt = x_cnt
        self.y_cnt = y_cnt
        self.hidden_cnts = hidden_cnts

        if len(hidden_cnts) < 1:
            raise Exception('Need to have at least 1 hidden layer.')

        # random weights and biases
        np.random.seed()

        self.hiddens = []
        random_scale = 1/x_cnt**.5
        layer = np.random.normal(scale=random_scale, size=(x_cnt+1, hidden_cnts[0]))
        self.hiddens.append(layer)
        for i in range(1, len(hidden_cnts)):
            layer = np.random.normal(scale=random_scale, size=(hidden_cnts[i-1]+1, hidden_cnts[i]))
            self.hiddens.append(layer)
        layer = np.random.normal(scale=random_scale, size=(hidden_cnts[-1]+1, y_cnt))
        self.hiddens.append(layer)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, xs:List[float])->List[float]:
        if len(xs) != self.x_cnt:
            raise Exception(f'This NN only accepts input of size {self.x_cnt}')
        output = np.asarray(xs)
        for hidden in self.hiddens:
            output = np.append(output, [1])
            output = self.sigmoid(np.dot(output, hidden))
        return output.tolist()