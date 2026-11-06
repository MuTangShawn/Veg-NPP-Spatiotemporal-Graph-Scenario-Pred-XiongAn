
import math
import torch
from torch import nn
import torch.nn.functional as F

class NHiTS_Block(nn.Module):
    def __init__(
        self,
        in_length,
        out_length,
        hidden_features,
        theta_features,
        kernel_size,
        stride,
        padding="same",
        pool_mode="max",
        in_features=1,
        out_features=1,
        num_hidden_layers=3,
        ):
        super(NHiTS_Block, self).__init__()

        self.in_length = in_length
        self.out_length = out_length
        self.kernel_size = kernel_size
        self.stride = stride

        if(padding == "same"):
            self.padding = (self.kernel_size - 1) // 2
        elif(padding == "valid"):
            self.padding = 0
        else:
            self.padding = padding

        self.pool_mode = pool_mode
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.theta_features = theta_features
        self.num_hidden_layers = num_hidden_layers

        self.pooled_length = self.conv_output_length()

        self.fc_in = nn.Linear(self.pooled_length * self.in_features, self.hidden_features)
        self.fc_hiddens = nn.ModuleList([nn.Linear(self.hidden_features, self.hidden_features) for _ in range(self.num_hidden_layers)])

        self.fc_theta_backcast = nn.Linear(self.hidden_features, self.theta_features * self.in_features)
        self.fc_theta_forecast = nn.Linear(self.hidden_features, self.theta_features * self.out_features)

        pass

    def conv_output_length(self):
        dilation = 1
        total_pad = self.padding * 2

        conv_length = (self.in_length + total_pad - dilation * (self.kernel_size - 1) - 1) / self.stride + 1
        return math.floor(conv_length)

    def forward(self, x): # x shape: (B, T, D)
        B = x.shape[0]
        x = x.transpose(1, 2) # (B, D_in, T_in)

        if(self.pool_mode == "average"):
            x = F.avg_pool1d(x, self.kernel_size, self.stride, self.padding) # (B, D_in, T_pool)
        elif(self.pool_mode == "max"):
            x = F.max_pool1d(x, self.kernel_size, self.stride, self.padding) # (B, D_in, T_pool)

        x = x.reshape(B, -1) # (B, D_in * T_pool)

        x = F.relu(self.fc_in(x))

        for hidden_layer in self.fc_hiddens:
            x = F.relu(hidden_layer(x))
        
        x_backcast = self.fc_theta_backcast(x) # (B, D * ¦È)
        x_forecast = self.fc_theta_forecast(x) # (B, D * ¦È)

        x_backcast = x_backcast.view(B, self.in_features, self.theta_features) # (B, D_in, ¦È)
        x_forecast = x_forecast.view(B, self.out_features, self.theta_features) # (B, D_out, ¦È)

        x_backcast = F.interpolate(x_backcast, self.in_length, mode="linear") # (B, D_in, T_in)
        x_forecast = F.interpolate(x_forecast, self.out_length, mode="linear") # (B, D_out, T_out)

        x_backcast = x_backcast.transpose(1, 2) # (B, T_in, D_in)
        x_forecast = x_forecast.transpose(1, 2) # (B, T_out, D_out)

        return x_backcast, x_forecast

class NHiTS_Stack(nn.Module):
    def __init__(
        self,
        in_length,
        out_length,
        hidden_features,
        theta_features,
        padding="same",
        pool_mode="max",
        num_blocks=3,
        in_features=1,
        out_features=1,
        num_hidden_layers=3,
        ):
        super(NHiTS_Stack, self).__init__()

        self.num_blocks = num_blocks
        self.block_list = nn.ModuleList([
            NHiTS_Block(
                in_length=in_length,
                out_length=out_length,
                hidden_features=hidden_features,
                theta_features=theta_features,
                kernel_size=2 ** i,
                stride=2 ** i,
                padding=padding,
                pool_mode=pool_mode,
                in_features=in_features,
                out_features=out_features,
                num_hidden_layers=num_hidden_layers,
            )
            for i in range(self.num_blocks)
        ])
        pass

    def forward(self, x): # x shape: (B, T, D)
        residual = x
        forecast_final = 0

        for block in self.block_list:
            backcast, forecast = block(residual)
            residual = residual - backcast
            forecast_final = forecast_final + forecast

        return forecast_final, residual

class NHiTS_Model(nn.Module):
    def __init__(
        self,
        in_length,
        out_length,
        hidden_features,
        theta_features,
        padding="same",
        pool_mode="max",
        num_stacks=3,
        num_blocks=3,
        in_features=1,
        out_features=1,
        num_hidden_layers=3,
        ):
        super(NHiTS_Model, self).__init__()

        self.num_stacks = num_stacks
        self.stack_list = nn.ModuleList([
            NHiTS_Stack(
                in_length=in_length,
                out_length=out_length,
                hidden_features=hidden_features,
                theta_features=theta_features,
                padding=padding,
                pool_mode=pool_mode,
                num_blocks=num_blocks,
                in_features=in_features,
                out_features=out_features,
                num_hidden_layers=num_hidden_layers,
            )
            for _ in range(self.num_stacks)
        ])
        pass

    def forward(self, x): # x shape: (B, T, D)
        residual = x
        forecast_final = 0

        for stack in self.stack_list:
            forecast_stack, residual = stack(residual)
            forecast_final += forecast_stack

        return forecast_final

class NHiTS_PerCH(nn.Module):
    def __init__(
        self,
        in_length,
        out_length,
        hidden_features,
        theta_features,
        padding="same",
        pool_mode="max",
        num_stacks=3,
        num_blocks=3,
        num_hidden_layers=3,
        share_thetas=True,
        res_scale=1.,
        ):
        super(NHiTS_PerCH, self).__init__()
        self.in_length = in_length
        self.out_length = out_length
        self.model = NHiTS_Model(
                in_length=in_length,
                out_length=out_length,
                hidden_features=hidden_features,
                theta_features=theta_features,
                padding=padding,
                pool_mode=pool_mode,
                num_stacks=num_stacks,
                num_blocks=num_blocks,
                in_features=1,
                out_features=1,
                num_hidden_layers=num_hidden_layers,
            )
    def forward(self, x): # x shape: (B, T, D)
        B, _, D = x.shape
        x = x.transpose(1, 2) # (B, D, T)
        x = x.reshape(B * D, self.in_length) # (B * D, T)
        x = x.unsqueeze(-1) # (B * D, T, 1)
        x = self.model(x)
        x = x.squeeze(-1) # (B * D, T)
        x = x.view(B, D, self.out_length) # (B, D, T)
        x = x.transpose(1, 2) # (B, T, D)
        return x
