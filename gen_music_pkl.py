import os
import random
import argparse
import json
import torch
import torch.utils.data
import sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle as pkl
import pandas as pd
from scipy import interpolate
import joblib as jl
import librosa
from pymo.parsers import BVHParser
from pymo.writers import BVHWriter
from pymo.preprocessing import MocapParameterizer, RootTransformer
import tqdm
import glob
import pandas as pd


def process_audio(audio_file_name, save_path):
    """
    一个wav
    kthjazz_gCH_sFM_cAll_d02_mCH_ch01_beatlestreetwashboardbandfortyandtight_003.wav
    生成2个完全相同的文件
    kthjazz_gCH_sFM_cAll_d02_mCH_ch01_beatlestreetwashboardbandfortyandtight_003_00.audio29_30fps.pkl
    kthjazz_gCH_sFM_cAll_d02_mCH_ch01_beatlestreetwashboardbandfortyandtight_003_00_mirrored.audio29_30fps.pkl
    主要数据
    mfcc:          20个，对应librosa的mfcc
    chroma:         6个，对应librosa的chroma_cens
    spectralflux:   1个，频谱流量，对应librosa的onset_strength
    Beatactivation: 1个，？
    Beat:           1个，是拍还是小节
    """
    audio_name = os.path.basename(audio_file_name)
    audio_name = audio_name.split('.')[0]

    # 原数据集文件，目标是为了找个起始帧，好与动作数据集对齐
    raw_pkl_file = './data/motorica_dance/%s_00.audio29_30fps.pkl' % audio_name
    with open(raw_pkl_file, 'rb') as f:
        raw_panda_data = pkl.load(f).astype('float32')
    raw_frames = raw_panda_data.shape[0]
    raw_beats_data = raw_panda_data['Beat_0'].values
    raw_beats_index = np.where(raw_beats_data > 0.99)
    raw_beats_index = raw_beats_index[0]
    raw_first_beat = raw_beats_index[0]

    # librosa处理
    FPS = 30
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    data, _ = librosa.load(audio_file_name, sr=SR)
    envelope = librosa.onset.onset_strength(y=data, sr=SR)
    mfcc = librosa.feature.mfcc(y=data, sr=SR, n_mfcc=20).T
    chroma = librosa.feature.chroma_cens(y=data, sr=SR, hop_length=HOP_LENGTH, n_chroma=6).T
    tempo, beat_idxs = librosa.beat.beat_track(onset_envelope=envelope, sr=SR, hop_length=HOP_LENGTH,
                                               start_bpm=120.0, tightness=100)
    beat_onehot = np.zeros_like(envelope, dtype=np.float32)
    beat_onehot[beat_idxs] = 1.0

    # 与原数据集对齐
    frames = envelope.shape[0]  # 帧数
    first_beat = beat_idxs[0]
    start_index = first_beat - raw_first_beat
    end_index = start_index + raw_frames
    assert(frames >= end_index), audio_name
    if frames < end_index:
        end_index = frames
    mfcc = mfcc[start_index:end_index,...]
    chroma = chroma[start_index:end_index,...]
    envelope = envelope[start_index:end_index, ...]
    beat_onehot = beat_onehot[start_index:end_index, ...]
    assert(beat_onehot[raw_first_beat] == 1)

    # 组织数据
    final_frames = envelope.shape[0]
    channels = audio_feature = np.concatenate([
        mfcc, chroma, envelope[:, None], envelope[:, None], beat_onehot[:, None]
    ], axis=-1)  # 合并后的numpy数组，shape=(frame, 29)
    time_list = [i/FPS for i in range(final_frames)]  # 以秒计算的各帧时间，shape=(frame,)
    time_index = pd.to_timedelta(time_list, unit='s')  # 转成panda的数据
    column_names = np.loadtxt('./data/motorica_dance/audio29_features.txt', dtype=str).tolist() # 字符串的list，shape=(29,)
    panda_data = pd.DataFrame(data=channels, index=time_index, columns=column_names)

    # 保存
    save_name_1 = os.path.join(save_path, audio_name + '_00.audio29_30fps.pkl')
    save_name_2 = os.path.join(save_path, audio_name + '_00_mirrored.audio29_30fps.pkl')
    with open(save_name_1, 'wb') as pkl_f1:
        pkl.dump(panda_data, pkl_f1)
    with open(save_name_2, 'wb') as pkl_f2:
        pkl.dump(panda_data, pkl_f2)

    """
    if False:
        with open(save_name_1, 'rb') as ff:
            reload_panda_data = pkl.load(ff).astype('float32')
        diff = reload_panda_data - panda_data
    """
    return


if __name__ == "__main__":
    save_path = './data/music_pkl/'
    music_files = glob.glob('./data/wav/*.wav')
    music_files.sort()
    for file_name in tqdm.tqdm(music_files):
        print("Process %s" % file_name)
        process_audio(file_name, save_path)
        break
