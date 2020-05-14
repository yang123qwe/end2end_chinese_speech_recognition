import torch
import numpy as np
import torch.nn as nn
import math
        
class MultiHeadedAttention(nn.Module):

    def __init__(self, n_head, n_feat, dropout_rate=0.0):
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask):
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  
        k = k.transpose(1, 2)  
        v = v.transpose(1, 2)  
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k) 
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  
            min_value = float(np.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  
        else:
            self.attn = torch.softmax(scores, dim=-1)  
        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, v)  
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)  
        return self.linear_out(x) 
    
    

    
   




















