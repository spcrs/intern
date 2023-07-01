from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNModel, self).__init__()

        self.hidden_dim = hidden_dim


        self.rnn = nn.RNN(input_dim, hidden_dim,
                          batch_first=True, nonlinearity="tanh")

        self.fc = nn.Linear(hidden_dim, output_dim)

        # nn.init.zeros_(self.rnn.weight_ih_l0)
        # nn.init.zeros_(self.rnn.weight_hh_l0)
        # nn.init.zeros_(self.rnn.bias_ih_l0)
        # nn.init.zeros_(self.rnn.bias_hh_l0)
        # nn.init.zeros_(self.fc.weight)
        # nn.init.zeros_(self.fc.bias)

    def forward(self, x):

        # h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        out, hn = self.rnn(x)

        out = self.fc(out[:,-1, :])

        return out


