
from matplotlib.pyplot import cool
import numpy as np

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sklearn.linear_model
import sklearn.metrics

import matplotlib.pyplot as plt

import tqdm

import GetModel
import DrawLog
import CONFIG
import PATH

def logcosh(pred, true):
    loss = torch.log(torch.cosh(pred - true))
    return loss.mean()

def r2_score(pred, true):
    true_mean = torch.mean(true)
    ss_tot = torch.sum((true - true_mean) ** 2)
    ss_res = torch.sum((true - pred) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def train_func(loader, model, optimizer, loss_func, metrics_func, device): 
    model.train() # 训练模式
    
    batch_loss_list, batch_metrics_list = [], []
    
    # 每个Mini-batch更新一次参数
    pbar = tqdm.tqdm(loader)
    for data in pbar:
        data = data.to(device)
        outputs = model(data.x, data.edge_index, data.edge_attr, data.future_fac, data.batch)  # 前向传播
        
        batch_size = int(data.batch.max().item()) + 1

        outputs = outputs.view(batch_size, -1, CONFIG.VMD_K)
        target = data.y.view(batch_size, -1, CONFIG.VMD_K)

        loss_tot = loss_func(outputs, target) # 计算损失
        loss_sum_cha = torch.sum(loss_tot, dim=-1)
        loss = torch.mean(loss_sum_cha)

        optimizer.zero_grad()  # 梯度清0
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新
        
        metrics = metrics_func(outputs, target) # 计算指标
        metrics = metrics.mean()

        batch_loss_list.append(loss.item())
        batch_metrics_list.append(metrics.item())
        pbar.set_postfix(loss='{:.4f}'.format(loss.item()), metrics='{:.4f}'.format(metrics.item()))  # 进度条右边显示信息
    
    return np.mean(batch_loss_list), np.mean(batch_metrics_list), np.std(batch_loss_list), np.std(batch_metrics_list)
    
def val_func(loader, model, loss_func, metrics_func, device):
    model.eval() # 测试模式

    batch_loss_list, batch_metrics_list = [], []
    
    pbar = tqdm.tqdm(loader)
    for data in pbar:

        data = data.to(device)

        outputs = model(data.x, data.edge_index, data.edge_attr, data.future_fac, data.batch)  # 前向传播
        
        batch_size = int(data.batch.max().item()) + 1
        
        outputs = outputs.view(batch_size, -1, CONFIG.VMD_K)
        target = data.y.view(batch_size, -1, CONFIG.VMD_K)

        loss_tot = loss_func(outputs, target) # 计算损失
        loss_sum_cha = torch.sum(loss_tot, dim=-1)
        loss = torch.mean(loss_sum_cha)

        metrics = metrics_func(outputs, target) # 计算指标
        metrics = metrics.mean()

        batch_loss_list.append(loss.item())
        batch_metrics_list.append(metrics.item())
        pbar.set_postfix(loss='{:.4f}'.format(loss.item()), metrics='{:.4f}'.format(metrics.item()))  # 进度条右边显示信息
    
    return np.mean(batch_loss_list), np.mean(batch_metrics_list), np.std(batch_loss_list), np.std(batch_metrics_list)

def test_func(loader, model, loss_func, metrics_func, device):
    model.eval() # 测试模式

    batch_loss_list, batch_metrics_list = [], []
    
    pbar = tqdm.tqdm(loader)
    for data in pbar:

        data = data.to(device)
        outputs = model(data.x, data.edge_index, data.edge_attr, data.future_fac, data.batch)  # 前向传播
        
        batch_size = int(data.batch.max().item()) + 1
        
        outputs = outputs.view(batch_size, -1, CONFIG.VMD_K)
        target = data.y.view(batch_size, -1, CONFIG.VMD_K)

        loss_tot = loss_func(outputs, target) # 计算损失
        loss_sum_cha = torch.sum(loss_tot, dim=-1)
        loss = torch.mean(loss_sum_cha)

        metrics = metrics_func(outputs, target) # 计算指标
        metrics = metrics.mean()

        batch_loss_list.append(loss.item())
        batch_metrics_list.append(metrics.item())
        pbar.set_postfix(loss='{:.4f}'.format(loss.item()), metrics='{:.4f}'.format(metrics.item()))  # 进度条右边显示信息
    
    return np.mean(batch_loss_list), np.mean(batch_metrics_list), np.std(batch_loss_list), np.std(batch_metrics_list)

    

def train(train_loader, val_loader, test_loader):
    model = GetModel.get_model()
    print(model)

    # 统计模型总参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 使用GPU or CPU
    model = model.to(device) # 加载模型
    optimizer = optim.Adam(model.parameters(), lr=CONFIG.INITIAL_LR, weight_decay=1e-4) # 优化器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=16, cooldown=8, min_lr=1e-9)
    
    #loss_func = torch.nn.L1Loss(reduction="mean")
    #loss_func = torch.nn.SmoothL1Loss(reduction="none")
    loss_func = logcosh

    metrics_func = r2_score
    #metrics_func = torch.nn.CosineSimilarity(dim=1)
    
    train_loss_avg_list, train_metrics_avg_list = [], []
    train_loss_std_list, train_metrics_std_list = [], []
    val_loss_avg_list, val_metrics_avg_list = [], []
    val_loss_std_list, val_metrics_std_list = [], []
    for epoch in range(CONFIG.EPOCHS):
        print('\nEpoch #{:03d}'.format(epoch + 1) + "/{:03d}".format(CONFIG.EPOCHS))
        
        train_loss_avg, train_metrics_avg, train_loss_std, train_metrics_std = train_func(train_loader, model, optimizer, loss_func=loss_func, metrics_func=metrics_func, device=device)
        val_loss_avg, val_metrics_avg, val_loss_std, val_metrics_std = val_func(val_loader, model, loss_func=loss_func, metrics_func=metrics_func, device=device)
        scheduler.step(val_loss_avg)

        lr_now = optimizer.param_groups[0]['lr']

        print('Train_Loss={:.4f}, Train_Metrics={:.4f}, Val_Loss={:.4f}, Val_Metrics={:.4f}, Learning_Rate={:.2e}'.format(train_loss_avg, train_metrics_avg, val_loss_avg, val_metrics_avg, lr_now))
        
        train_loss_avg_list.append(train_loss_avg)
        train_metrics_avg_list.append(train_metrics_avg)
        
        train_loss_std_list.append(train_loss_std)
        train_metrics_std_list.append(train_metrics_std)
        

        val_loss_avg_list.append(val_loss_avg)
        val_metrics_avg_list.append(val_metrics_avg)
        
        val_loss_std_list.append(val_loss_std)
        val_metrics_std_list.append(val_metrics_std)
        pass
    
    GetModel.save_model(model)
    DrawLog.write_log(train_loss_avg_list, train_metrics_avg_list, val_loss_avg_list, val_metrics_avg_list)
    
    test_loss_avg, test_metrics_avg, test_loss_std, test_metrics_std = test_func(test_loader, model, loss_func=loss_func, metrics_func=metrics_func, device=device)
    print('Test_Loss={:.4f}, Test_Metrics={:.4f}'.format(test_loss_avg, test_metrics_avg))
    
    pass
    