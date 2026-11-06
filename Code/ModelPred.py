
import os
import time

import numpy as np
import pandas as pd

import sklearn.preprocessing

import tqdm

import torch

from torch_geometric.data import Data

import GetModel
from DataProcess import stl_residual_vmd

import CONFIG
import PATH

def load_data():
    node_data = pd.read_csv(PATH.NODE_CSV_PATH, header=0, index_col=None)
    edge_data = pd.read_csv(PATH.EDGE_CSV_PATH, header=0, index_col=None)

    tmp_orig_data = pd.read_csv(PATH.TMP_ORIG_CSV_PATH, header=0, index_col=None)
    tmp_cmip_data = pd.read_csv(PATH.TMP_CMIP_CSV_PATH, header=0, index_col=None)

    prec_orig_data = pd.read_csv(PATH.PREC_ORIG_CSV_PATH, header=0, index_col=None)
    prec_cmip_data = pd.read_csv(PATH.PREC_CMIP_CSV_PATH, header=0, index_col=None)

    pet_orig_data = pd.read_csv(PATH.PET_ORIG_CSV_PATH, header=0, index_col=None)
    pet_cmip_data = pd.read_csv(PATH.PET_CMIP_CSV_PATH, header=0, index_col=None)

    lulc_data = pd.read_csv(PATH.LULC_CSV_PATH, header=0, index_col=None)

    node_data = np.array(node_data.values, dtype=np.float32) # rows: nodes; cols: time steps
    node_decomp_data = [stl_residual_vmd(ts) for ts in tqdm.tqdm(node_data)]
    node_decomp_data = np.array(node_decomp_data, dtype=np.float32) # (Nodes, Dims, Times)

    node_decomp_max = np.max(node_decomp_data, axis=-1, keepdims=True) # (Nodes, Dims, 1)
    node_decomp_min = np.min(node_decomp_data, axis=-1, keepdims=True) # (Nodes, Dims, 1)
        
    node_decomp_norm = np.where(np.abs(node_decomp_max) > np.abs(node_decomp_min), node_decomp_data / np.abs(node_decomp_max), node_decomp_data / np.abs(node_decomp_min))
    node_decomp_norm = np.transpose(node_decomp_norm, axes=(2, 0, 1)) # (Times, Nodes, Dims)

    node_mon_avg = []
    for month in tqdm.trange(1, 12 + 1):
        month_ext = node_decomp_norm[month - 1:len(node_decomp_norm) - CONFIG.TEST_LENGTH:12]
        month_avg = np.mean(month_ext, axis=0) # (Nodes, Dims)
        node_mon_avg.append(month_avg)
    else:
        node_mon_avg = np.array(node_mon_avg, dtype=np.float32) # (Times=12, Nodes, Dims)
        node_mon_avg = np.tile(node_mon_avg, ((CONFIG.PRED_END_YEAR - CONFIG.DATA_START_YEAR + 1) * 12 // len(node_mon_avg), 1, 1)) # (PredTimes, Nodes, Dims)
        pass

    edge_data = edge_data.values
    edge_index = edge_data[:, :2] # rows: edges; cols: 2
    edge_index = np.array(edge_index, dtype=np.uint16).T # rows: 2; cols: edges

    edge_attr = edge_data[:, 2:] # rows: edges; cols: attrs
    edge_attr = np.array(edge_attr, dtype=np.float32)


    tmp_orig_data = np.array(tmp_orig_data.values, dtype=np.float32) # rows: nodes; cols: time steps
    tmp_orig_data = tmp_orig_data.T # rows: time steps; cols: nodes
    tmp_cmip_data = np.array(tmp_cmip_data.values, dtype=np.float32) # rows: nodes; cols: time steps
    tmp_cmip_data = tmp_cmip_data.T # rows: time steps; cols: nodes
    tmp_data = np.concatenate([tmp_orig_data[:(CONFIG.SCEN_START_YEAR - CONFIG.DATA_START_YEAR) * 12],
                               tmp_cmip_data[(CONFIG.SCEN_START_YEAR - CONFIG.CMIP_START_YEAR) * 12:(CONFIG.PRED_END_YEAR - CONFIG.CMIP_START_YEAR + 1) * 12]], axis=0)
    
    prec_orig_data = np.array(prec_orig_data.values, dtype=np.float32) # rows: nodes; cols: time steps
    prec_orig_data = prec_orig_data.T # rows: time steps; cols: nodes
    prec_cmip_data = np.array(prec_cmip_data.values, dtype=np.float32) # rows: nodes; cols: time steps
    prec_cmip_data = prec_cmip_data.T # rows: time steps; cols: nodes
    prec_data = np.concatenate([prec_orig_data[:(CONFIG.SCEN_START_YEAR - CONFIG.DATA_START_YEAR) * 12],
                                prec_cmip_data[(CONFIG.SCEN_START_YEAR - CONFIG.CMIP_START_YEAR) * 12:(CONFIG.PRED_END_YEAR - CONFIG.CMIP_START_YEAR + 1) * 12]], axis=0)

    pet_orig_data = np.array(pet_orig_data.values, dtype=np.float32) # rows: nodes; cols: time steps
    pet_orig_data = pet_orig_data.T # rows: time steps; cols: nodes
    pet_cmip_data = np.array(pet_cmip_data.values, dtype=np.float32) # rows: nodes; cols: time steps
    pet_cmip_data = pet_cmip_data.T # rows: time steps; cols: nodes
    pet_data = np.concatenate([pet_orig_data[:(CONFIG.SCEN_START_YEAR - CONFIG.DATA_START_YEAR) * 12],
                               pet_cmip_data[(CONFIG.SCEN_START_YEAR - CONFIG.CMIP_START_YEAR) * 12:(CONFIG.PRED_END_YEAR - CONFIG.CMIP_START_YEAR + 1) * 12]], axis=0)

    lulc_data = np.array(lulc_data.values, dtype=np.float32) # rows: nodes; cols: time steps
    lulc_data = lulc_data.T # rows: time = 1987~2050 years; cols: nodes
    lulc_data = np.repeat(lulc_data, 12, axis=0) # rows: time steps (1987-2050); cols: nodes
    lulc_data = lulc_data[:(CONFIG.PRED_END_YEAR - CONFIG.DATA_START_YEAR + 1) * 12] # rows: time steps; cols: nodes
    lulc_data = np.expand_dims(lulc_data, axis=-1) # (times, nodes, 1)

    edge_scaler = sklearn.preprocessing.MinMaxScaler()
    edge_attr_scale = edge_scaler.fit_transform(edge_attr)
    edge_attr_scale = edge_attr_scale.astype(np.float32)

    '''
    tmp_scaler = sklearn.preprocessing.MinMaxScaler()
    tmp_scaler = tmp_scaler.fit(tmp_orig_data[:len(node_data) - CONFIG.TEST_LENGTH])
    tmp_data_scale = tmp_scaler.transform(tmp_data)
    tmp_data_scale = tmp_data_scale.astype(np.float32)
    tmp_data_scale = np.expand_dims(tmp_data_scale, axis=-1)
    '''

    tmp_mon_avg, tmp_mon_std = [], []
    for month in tqdm.trange(1, 12 + 1):
        month_ext = tmp_data[month - 1:len(node_decomp_norm) - CONFIG.TEST_LENGTH:12]
        month_avg = np.mean(month_ext, axis=0) # (Nodes,)
        tmp_mon_avg.append(month_avg)
        month_std = np.std(month_ext, axis=0) # (Nodes,)
        tmp_mon_std.append(month_std)
    else:
        tmp_mon_avg = np.array(tmp_mon_avg, dtype=np.float32) # (12, Nodes)
        tmp_mon_std = np.array(tmp_mon_std, dtype=np.float32) # (12, Nodes)
        pass

    tmp_data_scale = []
    for i in tqdm.trange(len(tmp_data)):
        month_m1 = i % 12
        tmp_scale_now = (tmp_data[i] - tmp_mon_avg[month_m1]) / tmp_mon_std[month_m1] # Standardization
        tmp_data_scale.append(tmp_scale_now)
    else:
        tmp_data_scale = np.array(tmp_data_scale, dtype=np.float32) # rows: time steps; cols: nodes
        tmp_data_scale = np.expand_dims(tmp_data_scale, axis=-1) # (times, nodes, 1)
        #tmp_data_scale = tmp_data_scale / (1 + np.abs(tmp_data_scale)) # Softsign
        pass

    '''
    prec_scaler = sklearn.preprocessing.MinMaxScaler()
    prec_scaler = prec_scaler.fit(prec_orig_data[:len(node_data) - CONFIG.TEST_LENGTH])
    prec_data_scale = prec_scaler.transform(prec_data)
    prec_data_scale = prec_data_scale.astype(np.float32)
    prec_data_scale = np.expand_dims(prec_data_scale, axis=-1)
    '''
    prec_mon_avg, prec_mon_std = [], []
    for month in tqdm.trange(1, 12 + 1):
        month_ext = prec_data[month - 1:len(node_decomp_norm) - CONFIG.TEST_LENGTH:12]
        month_avg = np.mean(month_ext, axis=0) # (Nodes,)
        prec_mon_avg.append(month_avg)
        month_std = np.std(month_ext, axis=0) # (Nodes,)
        prec_mon_std.append(month_std)
    else:
        prec_mon_avg = np.array(prec_mon_avg, dtype=np.float32) # (12, Nodes)
        prec_mon_std = np.array(prec_mon_std, dtype=np.float32) # (12, Nodes)
        pass

    prec_data_scale = []
    for i in tqdm.trange(len(prec_data)):
        month_m1 = i % 12
        prec_scale_now = (prec_data[i] - prec_mon_avg[month_m1]) / prec_mon_std[month_m1] # Standardization
        prec_data_scale.append(prec_scale_now)
    else:
        prec_data_scale = np.array(prec_data_scale, dtype=np.float32) # rows: time steps; cols: nodes
        prec_data_scale = np.expand_dims(prec_data_scale, axis=-1) # (times, nodes, 1)
        #prec_data_scale = prec_data_scale / (1 + np.abs(prec_data_scale)) # Softsign
        pass

    '''
    pet_scaler = sklearn.preprocessing.MinMaxScaler()
    pet_scaler = pet_scaler.fit(pet_data[:len(node_data) - CONFIG.TEST_LENGTH])
    pet_data_scale = pet_scaler.transform(pet_data)
    pet_data_scale = pet_data_scale.astype(np.float32)
    pet_data_scale = np.expand_dims(pet_data_scale, axis=-1)
    '''
    pet_mon_avg, pet_mon_std = [], []
    for month in tqdm.trange(1, 12 + 1):
        month_ext = pet_data[month - 1:len(node_decomp_norm) - CONFIG.TEST_LENGTH:12]
        month_avg = np.mean(month_ext, axis=0) # (Nodes,)
        pet_mon_avg.append(month_avg)
        month_std = np.std(month_ext, axis=0) # (Nodes,)
        pet_mon_std.append(month_std)
    else:
        pet_mon_avg = np.array(pet_mon_avg, dtype=np.float32) # (12, Nodes)
        pet_mon_std = np.array(pet_mon_std, dtype=np.float32) # (12, Nodes)
        pass

    pet_data_scale = []
    for i in tqdm.trange(len(pet_data)):
        month_m1 = i % 12
        pet_scale_now = (pet_data[i] - pet_mon_avg[month_m1]) / pet_mon_std[month_m1] # Standardization
        pet_data_scale.append(pet_scale_now)
    else:
        pet_data_scale = np.array(pet_data_scale, dtype=np.float32) # rows: time steps; cols: nodes
        pet_data_scale = np.expand_dims(pet_data_scale, axis=-1) # (times, nodes, 1)
        #pet_data_scale = pet_data_scale / (1 + np.abs(pet_data_scale)) # Softsign
        pass

    fac_data = np.concatenate([tmp_data_scale, prec_data_scale, pet_data_scale, lulc_data, node_mon_avg], axis=-1)

    return node_decomp_norm, edge_index, edge_attr_scale, fac_data, (node_decomp_max, node_decomp_min)

def pred_func(node_data, edge_index, edge_attr, fac_data, model, device):
    model.eval() # 测试模式
    
    y_pred_seq = []
    
    for i in tqdm.trange((CONFIG.PRED_END_YEAR - CONFIG.DATA_START_YEAR + 1) * 12 - CONFIG.TIME_STEP):
        if(i <= len(node_data) - CONFIG.TIME_STEP):
            x_seq = node_data[i:i + CONFIG.TIME_STEP]
        elif(i < len(node_data)):
            y_pred_temp = y_pred_seq[-(i + CONFIG.TIME_STEP - len(node_data)):]
            y_pred_temp = np.array(y_pred_temp, dtype=np.float32)
            x_seq = np.append(node_data[i:], y_pred_temp, axis=0)
        else:
            x_seq = y_pred_seq[-CONFIG.TIME_STEP:]
            x_seq = np.array(x_seq, dtype=np.float32)
            pass
        x_seq = np.transpose(x_seq, axes=(1, 0, 2)) # (Nodes, Times, Dims)
        future_fac = fac_data[i + CONFIG.TIME_STEP:i + CONFIG.TIME_STEP + CONFIG.OUT_LENGTH]
        future_fac = np.squeeze(future_fac, axis=0)
        x_data = Data(
            x=torch.tensor(x_seq, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            future_fac=torch.tensor(future_fac, dtype=torch.float),
        )
        x_data = x_data.to(device)
        
        with torch.no_grad(): y_pred = model(x_data.x, x_data.edge_index, x_data.edge_attr, x_data.future_fac, None)
        y_pred = y_pred.to("cpu") # rows: nodes; cols: vmd_k
        y_pred = y_pred.numpy()
        y_pred_seq.append(y_pred)

        
    return np.array(y_pred_seq, np.float32) # (Times, Nodes, Dims)

def model_predict():

    date_dir = "Pred_" + time.strftime("%Y-%m-%d", time.localtime())
    pred_dir = os.path.join(PATH.PRED_ROOT_DIR, date_dir)
    if(not os.path.exists(pred_dir)): os.makedirs(pred_dir)

    node_data, edge_index, edge_attr, fac_data, (scale_max, scale_min) = load_data()

    model = GetModel.get_model()
    print(model)
    
    # 统计模型总参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 使用GPU or CPU
    model = model.to(device) # 加载模型

    y_pred_seq = pred_func(node_data, edge_index, edge_attr, fac_data, model, device) # (Times, Nodes, Dims)
    y_pred_seq = np.transpose(y_pred_seq, axes=(1, 2, 0)) # (Nodes, Dims, Times)
    y_pred_unscale = np.where(np.abs(scale_max) > np.abs(scale_min), y_pred_seq * np.abs(scale_max), y_pred_seq * np.abs(scale_min)) # (Nodes, Dims, Times)
    y_pred_vmdinv = np.sum(y_pred_unscale, axis=1, dtype=np.float32) # (Nodes, Times)
    y_pred_vmdinv = np.where(y_pred_vmdinv < 0, 0, y_pred_vmdinv)

    csv_path = os.path.join(pred_dir, "NPP_COUNTY_PRED.csv")
    pd.DataFrame(y_pred_vmdinv).to_csv(csv_path)
    pass