# -*- coding: utf-8 -*-
# """
# Created on Sat Jun 18 15:37:00 2022
# @author: Dong HUANG
# Contact: huang_dong@tju.edu.cn
# """

# %% Import
import functools
import os
import pandas as pd
import scipy.signal as signal
import numpy as np
from scipy.signal import butter, filtfilt


def butter_notchstop(notch,Q,fs):
    b, a = signal.iirnotch(notch, Q,fs)
    return b, a


def preprocess_norm(eeg_data):
    scale_mean = np.mean(eeg_data, axis=-1, keepdims=True)
    scale_std = np.std(eeg_data, axis=-1, keepdims=True)
    eeg_data = (eeg_data - scale_mean) / (scale_std + 1e-8)

    return eeg_data


def preprocess_norm_layer(eeg_data):
    scale_mean = np.mean(eeg_data, axis=1, keepdims=True)
    scale_std = np.std(eeg_data, axis=1, keepdims=True)
    eeg_data = (eeg_data - scale_mean) / (scale_std + 1e-5)

    return eeg_data


@functools.lru_cache()
def cache_read(filename: str):
    ext = os.path.splitext(filename)[-1]
    if ext == '.csv':
        file_content = pd.read_csv(filename)
    else:
        raise NotImplementedError

    return file_content


def preprocess_ref(data: np.ndarray, ref_type: str) -> np.ndarray:
    # 原始参考逻辑（例：none, average 等）
    if ref_type == 'none':
        return data
    elif ref_type == 'average':
        # 平均参考：每个通道减去所有通道的平均值
        return data - np.mean(data, axis=0, keepdims=True)
    # 新增 SEED 双极参考分支
    elif ref_type == 'bipolar_SEED':
        import pandas as pd
        # 读取预先准备的通道对 CSV 文件
        refs = pd.read_csv('mid_files/reference_SEED_1.csv')  # 文件包含"anode, cathode"两列
        new_data = []
        for _, row in refs.iterrows():
            anode_idx = int(row['anode'])   # 阳极通道序号
            cathode_idx = int(row['cathode'])  # 阴极通道序号
            # 取对应通道信号做差
            new_data.append(data[anode_idx, :] - data[cathode_idx, :])
        eeg = np.stack(new_data, axis=0)
        return eeg
    else:
        raise ValueError(f"Unknown reference type: {ref_type}")



def preprocess_filt(data, low_cut=0.5, high_cut=45, fs=250, order=6):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    proced = signal.filtfilt(b, a, data)
    return proced


def preprocess_bsfilt(data, low_cut=48, high_cut=52, fs=250):
    # win = firwin(11, [low_cut, high_cut], pass_zero='bandpass', fs=fs)
    # proced = lfilter(win, 1, data)
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(6, [low, high], btype='bandstop')
    proced = signal.filtfilt(b, a, data)
    return proced


def preprocess_notch(data, notch=50, Q=35, fs=250):
    # notch, Q, fs = 50, 35, 200
    b, a = butter_notchstop(notch, Q, fs)
    filted_eeg_rawdata = filtfilt(b, a, data)
    return filted_eeg_rawdata


def preprocess_hpfilt(data, low_cut=0.1, fs=250):
    b, a = butter(11, low_cut, btype='hp', fs=fs)
    filted_data = filtfilt(b, a, data)
    return filted_data


def preprocess_01norm(data):
    scale_min = np.min(data, axis=-1, keepdims=True)
    scale_max = np.max(data, axis=-1, keepdims=True)
    eeg_data = (data - scale_min) / (scale_max - scale_min + 1e-5)

    return eeg_data


def preprocess_resample(data: np.ndarray, fs: int = 250, refs: int = 125):
    up_factor = refs
    down_factor = fs
    proced = signal.resample_poly(data, up=up_factor, down=down_factor, axis=-1)
    return proced


class PreProcessSequential:
    def __init__(self, config):
        self.fs = config.get('srate', 250)
        self.refs = config.get('re_srate', 100)
        self.bplow = config.get('bp_low', 1)
        self.bphigh = config.get('bp_high', 45)
        self.bslow = config.get('bs_low', 49)
        self.bshigh = config.get('bs_high', 51)
        self.notch = config.get('notch', 50)

    def __call__(self, data: np.ndarray):
        return self._sequential(data)

    def _sequential(self, x):
        x = preprocess_resample(x, fs=self.fs, refs=self.refs)
        # x = preprocess_bsfilt(x, low_cut=self.bslow, high_cut=self.bshigh, fs=self.fs)
        x = preprocess_filt(x, low_cut=self.bplow, high_cut=self.bphigh, fs=100)
        x = preprocess_norm(x)
        return x


def preprocess_seed_trial(raw_eeg, config):
    """对SEED数据集的单个trial进行预处理

    Args:
        raw_eeg: 原始EEG数据，形状为 [channels, time_points]
        config: 配置参数

    Returns:
        segments_array: 预处理后的数据片段数组
    """
    # 1) 获取配置参数
    ref_type = config.get('rerefence_type', 'average')
    fs = config.get('srate', 200)
    re_fs = config.get('re_srate', 100)

    # 2) 应用重参考
    eeg = preprocess_ref(raw_eeg, ref_type)

    # 3) 下采样
    if fs != re_fs:
        eeg = preprocess_resample(eeg, fs=fs, refs=re_fs)

    # 4) 带通滤波 1-45Hz
    eeg = preprocess_filt(eeg, low_cut=1, high_cut=45, fs=re_fs, order=6)

    # 5) Z分数归一化
    eeg = preprocess_norm(eeg)

    # 6) 提取最后60秒(如果需要)
    last_sec = 60
    start_idx = max(0, eeg.shape[1] - last_sec * re_fs)
    eeg = eeg[:, start_idx:]

    # 7) 滑动窗口分段
    win_len_sec = config.get('windowLength', 14)
    step_size = config.get('windowStep', 4)
    win_len = int(win_len_sec * re_fs)
    step_size = int(step_size * re_fs)

    # 计算可以提取的片段数量
    total_points = eeg.shape[1]
    segments = []

    for start in range(0, total_points - win_len, step_size):
        segment = eeg[:, start:start + win_len]
        segments.append(segment)

    if not segments:
        return None

    segments_array = np.stack(segments, axis=0)
    return segments_array