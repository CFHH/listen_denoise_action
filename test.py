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


def dataframe_nansinf2zeros(df):
    df.fillna(0, inplace=True)  # 用0填充NA/NaN
    df.replace([np.inf, -np.inf], 0, inplace=True) # 用0填充np.inf和-np.inf
    return df

def feats_to_bvh(pred_clips):
    # import pdb;pdb.set_trace()
    data_pipeline = jl.load('./data/motorica_dance/data_pipe.expmap_30fps.sav')
    n_feats = data_pipeline["cnt"].n_features
    data_pipeline["root"].separate_root = False

    print('inverse_transform...')
    bvh_data = data_pipeline.inverse_transform(pred_clips[:, :, :n_feats])
    return bvh_data


def test_audio_file():
    file1 = './data/motorica_dance/kthjazz_gCH_sFM_cAll_d02_mCH_ch01_beatlestreetwashboardbandfortyandtight_003_00.audio29_30fps.pkl'
    with open(file1, 'rb') as f1:
        data1 = pkl.load(f1).astype('float32')
        feats1 = dataframe_nansinf2zeros(data1)

    wav_file = './data/wav/kthjazz_gCH_sFM_cAll_d02_mCH_ch01_beatlestreetwashboardbandfortyandtight_003.wav'
    FPS = 30
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    data, _ = librosa.load(wav_file, sr=SR)
    envelope = librosa.onset.onset_strength(y=data, sr=SR)
    mfcc = librosa.feature.mfcc(y=data, sr=SR, n_mfcc=20).T
    chroma = librosa.feature.chroma_cens(y=data, sr=SR, hop_length=HOP_LENGTH, n_chroma=6).T
    tempo, beat_idxs = librosa.beat.beat_track(onset_envelope=envelope, sr=SR, hop_length=HOP_LENGTH,
                                               start_bpm=120.0, tightness=100)
    beat_onehot = np.zeros_like(envelope, dtype=np.float32)
    beat_onehot[beat_idxs] = 1.0
    return


def test_motion_file():
    file2 = './data/motorica_dance/kthjazz_gCH_sFM_cAll_d02_mCH_ch01_beatlestreetwashboardbandfortyandtight_003_00.expmap_30fps.pkl'
    with open(file2, 'rb') as f2:
        data2 = pkl.load(f2).astype('float32')
        feats2 = dataframe_nansinf2zeros(data2).values
    return


def test_sav_file():
    filename = './data/motorica_dance/data_pipe.expmap_30fps.sav'
    data_pipeline = jl.load(filename)
    n_feats = data_pipeline["cnt"].n_features
    data_pipeline["root"].separate_root = False
    return


def test_bvh():
    # 加载pkl文件
    pkl_file = './data/motorica_dance/kthjazz_gCH_sFM_cAll_d02_mCH_ch01_beatlestreetwashboardbandfortyandtight_003_00.expmap_30fps.pkl'
    with open(pkl_file, 'rb') as f2:
        pkl_data = pkl.load(f2).astype('float32')
        feats = dataframe_nansinf2zeros(pkl_data).values
        rddata1 = pkl_data['reference_dXposition']

    # 加载bvh文件
    filename = './data/bvh/kthjazz_gCH_sFM_cAll_d02_mCH_ch01_beatlestreetwashboardbandfortyandtight_003.bvh'
    bvh_parser = BVHParser()
    bvh_data = bvh_parser.parse(filename)
    bvh_data.values = bvh_data.values[::4]
    bvh_data.framerate = bvh_data.framerate * 4
    print(type(bvh_data))
    nframes = bvh_data.values.shape[0]
    bvh_datas = [bvh_data]

    # 取alpha、beta、gamma参数
    parameterizer = MocapParameterizer('expmap')
    expmap_datas = parameterizer.fit_transform(bvh_datas)
    expmap_data = expmap_datas[0]
    my_pkl_data = expmap_data.values  # 没有['reference_dXposition']

    # 取reference参数
    root_transformer = RootTransformer('pos_xyz_rot_deltas') # pos_xyz_rot_deltas
    trans_datas = root_transformer.fit_transform(bvh_datas)
    trans_data = trans_datas[0]
    my_pkl_data['reference_dXposition'] = trans_data.values['reference_dXposition']
    my_pkl_data['reference_dZposition'] = trans_data.values['reference_dZposition']
    my_pkl_data['reference_dYrotation'] = trans_data.values['reference_dYrotation']

    # 实际使用的骨骼
    motions_cols = np.loadtxt('./data/motorica_dance/pose_features.expmap.txt', dtype=str).tolist()
    my_pkl_data_final = my_pkl_data[motions_cols]

    # 比较差异
    pkl_diff = my_pkl_data_final - pkl_data

    # 转回bvh
    pred_clips = pkl_data.values[np.newaxis,...]
    pred_bvh_datas = feats_to_bvh(pred_clips)
    pred_bvh_data = pred_bvh_datas[0]
    bvh_diff = pred_bvh_data.values[bvh_data.values.columns].values - bvh_data.values[bvh_data.values.columns].values
    pred_bvh_data.values['Hips_Xposition'] += bvh_data.values['Hips_Xposition'][0] - pred_bvh_data.values['Hips_Xposition'][0]
    pred_bvh_data.values['Hips_Zposition'] += bvh_data.values['Hips_Zposition'][0] - pred_bvh_data.values['Hips_Zposition'][0]

    # 写bvh
    writer = BVHWriter()
    fname = f"./test.bvh"
    with open(fname, 'w') as f:
        writer.write(pred_bvh_data, f)

    return


if __name__ == "__main__":
    test_bvh()