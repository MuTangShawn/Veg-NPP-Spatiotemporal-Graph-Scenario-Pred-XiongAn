
import torch
from torch import nn

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(d_model, 1)  # 注意力权重生成器

    def forward(self, x):
        # x: (batch_size, seq_length, d_model)
        weights = self.attention_weights(x).squeeze(-1)  # (batch_size, seq_length)
        weights = torch.softmax(weights, dim=1)  # 归一化权重
        x = torch.sum(x * weights.unsqueeze(-1), dim=1)  # 加权求和 (batch_size, d_model)
        return x
