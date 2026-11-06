
import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000, base=10000):
        super(PositionalEncoding, self).__init__()
        # 创建一个 (max_seq_length, d_model) 的位置矩阵
        position = torch.arange(0, max_seq_length).unsqueeze(1)  # (max_seq_length, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(base) / d_model))
        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        pe = pe.unsqueeze(0)  # 添加 batch 维度
        self.register_buffer('pe', pe)  # 将 pe 注册为模型缓冲区，防止被训练

    def forward(self, x):
        # 叠加位置编码，假设 x 的形状是 (batch_size, seq_length, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x