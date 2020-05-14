import torch
import torch.nn as nn
from .fbanksampe import fesubsampling
from .PosFeedForward import PositionwiseFeedForward
from .Attention import MultiHeadedAttention

class Layers(nn.Module):
    def __init__(self, attention_heads, d_model, linear_units, residual_dropout_rate):
        super(Layers, self).__init__()

        self.self_attn = MultiHeadedAttention(attention_heads, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, linear_units)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(residual_dropout_rate)
        self.dropout2 = nn.Dropout(residual_dropout_rate)

    def forward(self, x, mask):
        residual = x
        x = residual + self.dropout1(self.self_attn(x, x, x, mask))
        x = self.norm1(x)

        residual = x
        x = residual + self.dropout2(self.feed_forward(x))
        x = self.norm2(x)

        return x, mask


class speech_model(nn.Module):

    def __init__(self, input_size=40, d_model=320, attention_heads=8, linear_units=1280, num_blocks=12, 
                 repeat_times=1, pos_dropout_rate=0.0, slf_attn_dropout_rate=0.0, ffn_dropout_rate=0.0, 
                 residual_dropout_rate=0.1):
        super(speech_model, self).__init__()

        self.embed = fesubsampling(input_size, d_model)

        self.blocks = nn.ModuleList([
            Layers(attention_heads,
                                    d_model, 
                                    linear_units, 
                                    residual_dropout_rate) for _ in range(num_blocks)
        ])
        self.liner = nn.Linear(d_model , 4709)
        self.softmax = nn.LogSoftmax(dim=2)
    def forward(self, inputs):

        enc_mask = torch.sum(inputs, dim=-1).ne(0).unsqueeze(-2)
        enc_output, enc_mask = self.embed(inputs, enc_mask)

        enc_output.masked_fill_(~enc_mask.transpose(1, 2), 0.0)

        for _, block in enumerate(self.blocks):
            enc_output, _ = block(enc_output, enc_mask)
        lin_ = self.liner(enc_output.transpose(0,1))
        logits_ctc_ = self.softmax(lin_)
        return logits_ctc_ 
        



















