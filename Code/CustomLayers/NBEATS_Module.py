
import torch
from torch import nn
import torch.nn.functional as F

class NBEATS_Block(nn.Module):
    def __init__(
        self,
        in_length,
        out_length,
        hidden_features,
        theta_features,
        in_features=1,
        out_features=1,
        num_hidden_layers=3,
        share_thetas=True,
        ):
        super(NBEATS_Block, self).__init__()

        self.in_length = in_length
        self.out_length = out_length
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.theta_features = theta_features
        self.num_hidden_layers = num_hidden_layers
        self.share_thetas = share_thetas

        self.fc_in = nn.Linear(self.in_length * self.in_features, self.hidden_features)
        self.fc_hiddens = nn.ModuleList([nn.Linear(self.hidden_features, self.hidden_features) for _ in range(self.num_hidden_layers)])

        self.fc_theta_backcast = nn.Linear(self.hidden_features, self.theta_features)
        if self.share_thetas:
            self.fc_theta_forecast = self.fc_theta_backcast
        else:
            self.fc_theta_forecast = nn.Linear(self.hidden_features, self.theta_features)

        self.fc_out_backcast = nn.Linear(self.theta_features, self.in_length * self.in_features)
        self.fc_out_forecast = nn.Linear(self.theta_features, self.out_length * self.out_features)

        pass

    def forward(self, x):
        B = x.shape[0]
        x = x.reshape(B, -1)

        x = F.relu(self.fc_in(x))

        for hidden_layer in self.fc_hiddens:
            x = F.relu(hidden_layer(x))
        
        x_backcast = self.fc_theta_backcast(x)
        if self.share_thetas:
            x_forecast = x_backcast
        else:
            x_forecast = self.fc_theta_forecast(x)

        x_backcast = self.fc_out_backcast(x_backcast)
        x_forecast = self.fc_out_forecast(x_forecast)

        x_backcast = x_backcast.reshape(B, self.in_length, self.in_features)
        x_forecast = x_forecast.reshape(B, self.out_length, self.out_features)

        return x_backcast, x_forecast

class NBEATS_Stack(nn.Module):
    def __init__(
        self,
        in_length,
        out_length,
        hidden_features,
        theta_features,
        num_blocks=3,
        in_features=1,
        out_features=1,
        num_hidden_layers=3,
        share_thetas=True,
        res_scale=1.,
        ):
        super(NBEATS_Stack, self).__init__()

        self.num_blocks = num_blocks
        self.res_scale = res_scale
        self.block_list = nn.ModuleList([
            NBEATS_Block(
                in_length=in_length,
                out_length=out_length,
                hidden_features=hidden_features,
                theta_features=theta_features,
                in_features=in_features,
                out_features=out_features,
                num_hidden_layers=num_hidden_layers,
                share_thetas=share_thetas,
            )
            for _ in range(self.num_blocks)
        ])
        pass

    def forward(self, x): # x shape: (B, T, D)
        residual = x
        forecast_final = 0

        for block in self.block_list:
            backcast, forecast = block(residual)
            residual = residual - self.res_scale * backcast
            forecast_final = forecast_final + self.res_scale * forecast

        return forecast_final, residual

class NBEATS_Model(nn.Module):
    def __init__(
        self,
        in_length,
        out_length,
        hidden_features,
        theta_features,
        num_stacks=3,
        num_blocks=3,
        in_features=1,
        out_features=1,
        num_hidden_layers=3,
        share_thetas=True,
        res_scale=1.,
        ):
        super(NBEATS_Model, self).__init__()

        self.num_stacks = num_stacks
        self.stack_list = nn.ModuleList([
            NBEATS_Stack(
                in_length=in_length,
                out_length=out_length,
                hidden_features=hidden_features,
                theta_features=theta_features,
                num_blocks=num_blocks,
                in_features=in_features,
                out_features=out_features,
                num_hidden_layers=num_hidden_layers,
                share_thetas=share_thetas,
                res_scale=res_scale,
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

class NBEATS_PerCH(nn.Module):
    def __init__(
        self,
        in_length,
        out_length,
        hidden_features,
        theta_features,
        num_stacks=3,
        num_blocks=3,
        num_hidden_layers=3,
        share_thetas=True,
        res_scale=1.,
        ):
        super(NBEATS_PerCH, self).__init__()
        self.in_length = in_length
        self.out_length = out_length
        self.model = NBEATS_Model(
                in_length=in_length,
                out_length=out_length,
                hidden_features=hidden_features,
                theta_features=theta_features,
                num_stacks=num_stacks,
                num_blocks=num_blocks,
                in_features=1,
                out_features=1,
                num_hidden_layers=num_hidden_layers,
                share_thetas=share_thetas,
                res_scale=res_scale,
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
