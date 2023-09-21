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


def write_bvh(bvh_data, fname):
    writer = BVHWriter()
    with open(fname, 'w') as f:
        writer.write(bvh_data, f)


def process_audio():
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


def process_motion():
    # 一些文件


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
    # 这算成30fps
    bvh_data.values = bvh_data.values[::4]
    bvh_data.framerate = bvh_data.framerate * 4
    bvh_data.values['Hips_Xposition'] -= bvh_data.values['Hips_Xposition'][0]
    bvh_data.values['Hips_Zposition'] -= bvh_data.values['Hips_Zposition'][0]
    # 保存一下
    write_bvh(bvh_data, './raw.bvh')

    print(type(bvh_data))
    nframes = bvh_data.values.shape[0]
    bvh_datas = [bvh_data]

    # 取alpha、beta、gamma参数
    parameterizer = MocapParameterizer('expmap')
    expmap_datas = parameterizer.fit_transform(bvh_datas)
    expmap_data = expmap_datas[0]
    my_pkl_data = expmap_data.values  # 没有['reference_dXposition']
    # 测试
    test_bvh_datas_1 = parameterizer.inverse_transform(expmap_datas)
    diff_1 = test_bvh_datas_1[0].values - bvh_datas[0].values  # 是零

    # 取reference参数
    root_transformer = RootTransformer('pos_rot_deltas', separate_root=False) # pos_xyz_rot_deltas, pos_rot_deltas
    trans_datas = root_transformer.fit_transform(bvh_datas)
    trans_data = trans_datas[0]
    my_pkl_data['reference_dXposition'] = trans_data.values['reference_dXposition']
    my_pkl_data['reference_dZposition'] = trans_data.values['reference_dZposition']
    my_pkl_data['reference_dYrotation'] = trans_data.values['reference_dYrotation']
    # 测试
    test_bvh_datas = root_transformer.inverse_transform(trans_datas)
    if root_transformer.separate_root:
        test_Hips_Xposition = bvh_data.values['Hips_Xposition'] - bvh_data.values['Hips_Xposition'][0]
        test_Hips_Zposition = bvh_data.values['Hips_Zposition'] - bvh_data.values['Hips_Zposition'][0]
        test_Hips_Yrotation = bvh_data.values['Hips_Yrotation'] - bvh_data.values['Hips_Yrotation'][0]
        test1 = test_bvh_datas[0].values['reference_Xposition'] - test_Hips_Xposition
        test2 = test_bvh_datas[0].values['reference_Zposition'] - test_Hips_Zposition
        test3 = test_bvh_datas[0].values['reference_Yrotation'] - test_Hips_Yrotation
        max1 = np.max(np.abs(test1.values))
        max2 = np.max(np.abs(test2.values))
        max3 = np.max(np.abs(test3.values))
    else:
        test_bvh_data = test_bvh_datas[0]
        test_bvh_data.values['Hips_Xposition'] -= test_bvh_data.values['Hips_Xposition'][0]
        test_bvh_data.values['Hips_Zposition'] -= test_bvh_data.values['Hips_Zposition'][0]
        bvh_diff = test_bvh_data.values - bvh_data.values

    # 实际使用的骨骼
    motions_cols = np.loadtxt('./data/motorica_dance/pose_features.expmap.txt', dtype=str).tolist()
    my_pkl_data_final = my_pkl_data[motions_cols]

    # 比较差异
    pkl_diff = my_pkl_data_final - pkl_data

    # pkl_data转回bvh
    pkl_clips = pkl_data.values[np.newaxis,...]
    pkl_bvh_datas = feats_to_bvh(pkl_clips)
    pkl_bvh_data = pkl_bvh_datas[0]
    pkl_bvh_diff = pkl_bvh_data.values[bvh_data.values.columns].values - bvh_data.values[bvh_data.values.columns].values
    pkl_bvh_data.values['Hips_Xposition'] += bvh_data.values['Hips_Xposition'][0] - pkl_bvh_data.values['Hips_Xposition'][0]
    pkl_bvh_data.values['Hips_Zposition'] += bvh_data.values['Hips_Zposition'][0] - pkl_bvh_data.values['Hips_Zposition'][0]
    write_bvh(pkl_bvh_data, './pkl.bvh')

    #pkl_data转回bvh
    my_clips = my_pkl_data_final.values[np.newaxis, ...]
    my_bvh_datas = feats_to_bvh(my_clips)
    my_bvh_data = my_bvh_datas[0]
    my_bvh_diff = my_bvh_data.values[bvh_data.values.columns].values - bvh_data.values[bvh_data.values.columns].values
    my_bvh_data.values['Hips_Xposition'] += bvh_data.values['Hips_Xposition'][0] - my_bvh_data.values['Hips_Xposition'][0]
    my_bvh_data.values['Hips_Zposition'] += bvh_data.values['Hips_Zposition'][0] - my_bvh_data.values['Hips_Zposition'][0]
    write_bvh(my_bvh_data, './mine.bvh')

    return


if __name__ == "__main__":
    process_motion()
    #process_audio()