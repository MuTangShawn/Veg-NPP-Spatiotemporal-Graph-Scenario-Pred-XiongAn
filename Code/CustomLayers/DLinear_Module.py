
import torch
from torch import nn
import torch.nn.functional as F

class DLinear_Model(nn.Module):
    def __init__(
        self,
        in_length,
        out_length,
        kernel_size=5,
        in_features=1,
        out_features=1,
        ):
        super(DLinear_Model, self).__init__()

        self.in_length = in_length
        self.out_length = out_length
        self.kernel_size = kernel_size
        self.in_features = in_features
        self.out_features = out_features

        self.fc_trend = nn.Linear(self.in_length * self.in_features, self.out_length * self.out_features)
        self.fc_seasonal = nn.Linear(self.in_length * self.in_features, self.out_length * self.out_features)

        pass

    def forward(self, x): # x shape: (B, T, D)
        B = x.shape[0]
        x = x.transpose(1, 2) # (B, D, T)

        pad_front = (self.kernel_size - 1) // 2
        pad_end = self.kernel_size - 1 - pad_front
        x_pad = F.pad(x, pad=(pad_front, pad_end), mode="replicate") # (B, D, T_pad)
        x_mov_avg = F.avg_pool1d(x_pad, self.kernel_size, 1, 0)
        x = x - x_mov_avg

        x = x.reshape(B, -1)
        x_mov_avg = x_mov_avg.reshape(B, -1)

        x = self.fc_seasonal(x)
        x_mov_avg = self.fc_trend(x_mov_avg)

        x = x.view(B, self.out_length, self.out_features)
        x_mov_avg = x_mov_avg.view(B, self.out_length, self.out_features)
        x = x + x_mov_avg

        return x


class DLinear_Share(nn.Module):
    def __init__(
        self,
        in_length,
        out_length,
        ):
        super(DLinear_Share, self).__init__()
        self.in_length = in_length
        self.out_length = out_length
        self.model = DLinear_Model(
                in_length=in_length,
                out_length=out_length,
                in_features=1,
                out_features=1,
            )
    def forward(self, x): # x shape: (B, T, D)
        B, _, D = x.shape
        x = x.transpose(1, 2) # (B, D, T)
        x = x.contiguous().view(B * D, self.in_length) # (B * D, T)
        x = x.unsqueeze(-1) # (B * D, T, 1)
        x = self.model(x)
        x = x.squeeze(-1) # (B * D, T)
        x = x.view(B, D, self.out_length) # (B, D, T)
        x = x.transpose(1, 2).contiguous() # (B, T, D)
        return x

class DLinear_Indiv(nn.Module):
    def __init__(
        self,
        in_length,
        out_length,
        d_model,
        ):
        super(DLinear_Indiv, self).__init__()
        self.in_length = in_length
        self.out_length = out_length
        self.d_model = d_model
        self.model_list = nn.ModuleList(
            [DLinear_Model(
                in_length=in_length,
                out_length=out_length,
                in_features=1,
                out_features=1,)
             for _ in range(self.d_model)])
    def forward(self, x): # x shape: (B, T, D)
        out_list = []
        for i in range(self.d_model):
            x_now = x[..., i] # (B, T_in)
            x_now = torch.unsqueeze(x_now, dim=-1) # (B, T_in, 1)
            y_now = self.model_list[i](x_now) # Êä³ö shape: (B, T_out, 1)
            out_list.append(y_now)
        else:
            out_cat = torch.cat(out_list, dim=-1)
        return out_cat