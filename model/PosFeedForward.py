import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    def __init__(self, idim, hidden_units, dropout_rate=0.0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(idim, hidden_units * 2)
        self.w_2 = nn.Linear(hidden_units, idim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.w_1(x)
        x = F.glu(x)
        return self.w_2(self.dropout(x))
    
