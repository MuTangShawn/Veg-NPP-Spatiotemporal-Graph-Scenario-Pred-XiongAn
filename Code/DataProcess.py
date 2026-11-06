
import numpy as np
import pandas as pd

import sklearn.preprocessing

from statsmodels.tsa.seasonal import STL
from vmdpy import VMD

import matplotlib.pyplot as plt
import tqdm

import CONFIG
import PATH

def stl_decomposition(ts_vec: np.ndarray, start_date="1987-01-01", period=12, freq='MS'):
    """
    对输入的一维时间序列向量进行 STL 分解。
    
    参数:
        ts_vec (np.ndarray): 原始时间序列，一维 numpy 向量。
        start_date (str): 起始日期，默认 "1987-01-01"。
        period (int): 季节性周期长度，比如 12 对应月度序列的年周期。
        freq (str): 时间序列频率，默认 'M' 表示月度数据（可设置为 'D', 'Q', 'A' 等）。
    
    返回:
        (trend, seasonal, resid): 返回趋势、季节性和残差三个分量，都是 pd.Series。
    """
    if not isinstance(ts_vec, np.ndarray):
        raise TypeError("ts_vec must be a numpy.ndarray")
    if ts_vec.ndim != 1:
        raise ValueError("ts_vec must be a one-dimensional array")

    # 创建时间索引
    date_index = pd.date_range(start=start_date, periods=len(ts_vec), freq=freq)
    time_series = pd.Series(ts_vec, index=date_index)

    # STL 分解
    stl_res = STL(time_series, period=period, seasonal=15, trend=101, robust=True).fit()
    '''
    # ✅ 可视化
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(time_series, label='Original', color='black')
    axes[0].set_title(f"Pixel ({i}, {j}) - Original Time Series")

    axes[1].plot(stl_res.trend, label='Trend', color='blue')
    axes[1].set_title("Trend")

    axes[2].plot(stl_res.seasonal, label='Seasonal', color='green')
    axes[2].set_title("Seasonal")

    axes[3].plot(stl_res.resid, label='Residual', color='red')
    axes[3].set_title("Residual")

    plt.tight_layout()
    plt.show()
    '''
    return stl_res.trend, stl_res.seasonal, stl_res.resid

def time_series_vmd(ts_vec: np.ndarray, alpha=10000, tau=0., K=CONFIG.VMD_K, DC=0, init=1, tol=1e-7):
    """
    对输入的一维时间序列向量进行 VMD 分解。
    
    参数:
        ts_vec (np.ndarray): 原始时间序列，一维 numpy 向量。
        alpha (float): 平滑参数。表示对模态带宽的惩罚项（惩罚模态分量的频率带宽），值越大 → 模态越窄、平滑。一般取 2000~10000。
        tau (float): 时间延迟约束。保留时间延迟信息的权重，通常设为 0。
        K (int): 要分解的模态数，即要分成多少个子信号，一般为 3~8 ，可调优选择最佳 K。
        DC (int, 0 or 1): 是否强制将第一个模态作为DC分量(直流分量), 设置为 1 时，第一个分量为常数项；否则设 0。
        init (int): 初始化中心频率的方法，1=均匀分布，2=随机，通常设为 1（默认）。
        tol (float): 收敛容差，优化目标达到该精度即终止迭代	一般设为 1e-7，更小则更精确但更慢。
    
    返回:
        u (np.ndarray): 时域下的分解结果：每个模态（mode）的时间序列，形状为 (K, N)，其中 K 为模态数量，N 为时间长度。
        u_hat (np.ndarray): 频域下的模态表示（傅里叶变换结果），形状为 (K, N)，用于频谱分析。
        omega (np.ndarray): 每个模态的中心频率（从训练中自适应学习），形状为 (K,)。
    """
    if not isinstance(ts_vec, np.ndarray):
        raise TypeError("ts_vec must be a numpy.ndarray")
    if ts_vec.ndim != 1:
        raise ValueError("ts_vec must be a one-dimensional array")
    
    # VMD 分解
    u, u_hat, omega = VMD(ts_vec, alpha, tau, K, DC, init, tol)
    '''
    # ✅ 可视化
    fig, axes = plt.subplots(K + 1, 1, figsize=(12, 8), sharex=True)

    for k in range(K + 1):
        if (k == 0):
            label = "Original"
            axes[k].plot(ts_vec, label=label, c="C" + str(k))
            axes[k].set_title(label)
        else:
            label = "Mode " + str(k)
            axes[k].plot(u[k - 1], label=label, c="C" + str(k))
            axes[k].set_title(label)
    else:
        label = "Restore"
        axes[0].plot(np.sum(u, axis=0), label=label, c="C" + str(K))
        #axes[0].set_title(label)
    

    plt.tight_layout()
    plt.show()
    '''
    return u, u_hat, omega

def stl_residual_vmd(ts_vec: np.ndarray, K=CONFIG.VMD_K):
    trend, seasonal, resid = stl_decomposition(ts_vec)
    trend = trend.to_numpy()
    seasonal = seasonal.to_numpy()
    resid = resid.to_numpy()
    u, u_hat, omega = time_series_vmd(resid, K=K)

    res_arr = [trend, seasonal, ]
    for k in range(K):
        res_arr.append(u[k])
    else:
        res_arr = np.array(res_arr, dtype=np.float32)
    '''
    # ✅ 可视化
    fig, axes = plt.subplots(K + 3 + 1, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(ts_vec, label='Original', color='C0')
    axes[0].set_title(f"Pixel ({i}, {j}) - Original Time Series")

    axes[1].plot(trend, label='Trend', color='C1')
    axes[1].set_title("Trend Term")

    axes[2].plot(seasonal, label='Seasonal', color='C2')
    axes[2].set_title("Seasonal Term")

    axes[3].plot(resid, label='Residual', color='C3')
    axes[3].set_title("Residual Term")

    for k in range(K):
        label = "Mode " + str(k + 1)
        title = "Residual Term VMD Mode " + str(k + 1)
        axes[k + 4].plot(u[k], label=label, c="C" + str(k + 4))
        axes[k + 4].set_title(title)
    else:
        axes[0].plot(np.sum(res_arr, axis=0), label="Restore", c="black", alpha=0.5)
        axes[3].plot(np.sum(u, axis=0), label="Restore", c="black", alpha=0.5)
    

    plt.tight_layout()
    plt.show()
    '''
    return res_arr


def org_data_process():
    node_data = pd.read_csv(PATH.NODE_CSV_PATH, header=0, index_col=None)
    edge_data = pd.read_csv(PATH.EDGE_CSV_PATH, header=0, index_col=None)

    tmp_data = pd.read_csv(PATH.TMP_ORIG_CSV_PATH, header=0, index_col=None)
    prec_data = pd.read_csv(PATH.PREC_ORIG_CSV_PATH, header=0, index_col=None)
    pet_data = pd.read_csv(PATH.PET_ORIG_CSV_PATH, header=0, index_col=None)
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
        node_mon_avg = np.tile(node_mon_avg, (len(node_decomp_norm) // len(node_mon_avg), 1, 1)) # (Times, Nodes, Dims)
        pass

    edge_data = edge_data.values
    edge_index = edge_data[:, :2] # rows: edges; cols: 2
    edge_index = np.array(edge_index, dtype=np.uint16).T # rows: 2; cols: edges

    edge_attr = edge_data[:, 2:] # rows: edges; cols: attrs
    edge_attr = np.array(edge_attr, dtype=np.float32)

    tmp_data = np.array(tmp_data.values, dtype=np.float32) # rows: nodes; cols: time steps
    tmp_data = tmp_data.T # rows: time steps; cols: nodes
    
    prec_data = np.array(prec_data.values, dtype=np.float32) # rows: nodes; cols: time steps
    prec_data = prec_data.T # rows: time steps; cols: nodes

    pet_data = np.array(pet_data.values, dtype=np.float32) # rows: nodes; cols: time steps
    pet_data = pet_data.T # rows: time steps; cols: nodes

    lulc_data = np.array(lulc_data.values, dtype=np.float32) # rows: nodes; cols: time steps
    lulc_data = lulc_data.T # rows: time = 1987~2050 years; cols: nodes
    lulc_data = np.repeat(lulc_data, 12, axis=0) # rows: time steps (1987-2050); cols: nodes
    lulc_data = lulc_data[:len(node_decomp_norm)] # rows: time steps; cols: nodes
    lulc_data = np.expand_dims(lulc_data, axis=-1) # (times, nodes, 1)

    edge_scaler = sklearn.preprocessing.MinMaxScaler()
    edge_attr_scale = edge_scaler.fit_transform(edge_attr)
    edge_attr_scale = edge_attr_scale.astype(np.float32)

    '''
    tmp_scaler = sklearn.preprocessing.MinMaxScaler()
    tmp_scaler = tmp_scaler.fit(tmp_data[:-CONFIG.TEST_LENGTH])
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
    prec_scaler = prec_scaler.fit(prec_data[:-CONFIG.TEST_LENGTH])
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
    pet_scaler = pet_scaler.fit(pet_data[:-CONFIG.TEST_LENGTH])
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

    graph_data_list = []
    for i in tqdm.trange(node_decomp_norm.shape[0] - CONFIG.TIME_STEP):
        data_x = node_decomp_norm[i:i + CONFIG.TIME_STEP]
        data_x = np.transpose(data_x, axes=(1, 0, 2)) # (Nodes, Times, Dims)
        data_y = node_decomp_norm[i + CONFIG.TIME_STEP:i + CONFIG.TIME_STEP + CONFIG.OUT_LENGTH]
        data_y = np.transpose(data_y, axes=(1, 0, 2)) # (Nodes, Times=1, Dims)
        data_y = np.squeeze(data_y, axis=1) # (Nodes, Dims)
        future_fac = fac_data[i + CONFIG.TIME_STEP:i + CONFIG.TIME_STEP + CONFIG.OUT_LENGTH] # (1, Nodes, Dims)
        future_fac = np.squeeze(future_fac, axis=0) # (Nodes, Dims)

        graph_data = {
            'node_features': data_x,
            'edge_index': edge_index,
            'edge_attr': edge_attr_scale,
            'future_fac': future_fac,
            'ground_truth': data_y,
        }
        graph_data_list.append(graph_data)

    train_data_list, val_data_list, test_data_list = [], [], []
    for i in tqdm.trange(len(graph_data_list)):
        if(len(graph_data_list) - i <= CONFIG.TEST_LENGTH):
            test_data_list.append(graph_data_list[i])
        elif(i % 7 == 2 or i & 7 == 6):
            val_data_list.append(graph_data_list[i])
        else:
            train_data_list.append(graph_data_list[i])
    print("Train Dataset:", len(train_data_list), "Val Dataset:", len(val_data_list), "Test Dataset:", len(test_data_list))

    return train_data_list, val_data_list, test_data_list