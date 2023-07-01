import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        
        self.lstm = nn.LSTM(input_dim, hidden_dim,  batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        
        out, hn = self.lstm(x)

        out = self.fc(out[:, -1, :])  #hn 

        return out

