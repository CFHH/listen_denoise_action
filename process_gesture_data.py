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


def process_motion():
    # 一些文件
    bvh_filename = './data/bvh/kthjazz_gCH_sFM_cAll_d02_mCH_ch01_beatlestreetwashboardbandfortyandtight_003.bvh'
    bone_feature_filename = './data/motorica_dance/pose_features.expmap.txt'

    # 加载bvh文件
    bvh_parser = BVHParser()
    bvh_data = bvh_parser.parse(bvh_filename)
    # 这算成30fps
    bvh_data.values = bvh_data.values[::4]
    bvh_data.framerate = bvh_data.framerate * 4
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

    # 取reference参数
    root_transformer = RootTransformer('pos_xyz_rot_deltas') # pos_xyz_rot_deltas
    trans_datas = root_transformer.fit_transform(bvh_datas)
    trans_data = trans_datas[0]
    my_pkl_data['reference_dXposition'] = trans_data.values['reference_dXposition']
    my_pkl_data['reference_dZposition'] = trans_data.values['reference_dZposition']
    my_pkl_data['reference_dYrotation'] = trans_data.values['reference_dYrotation']

    # 实际使用的骨骼
    motions_cols = np.loadtxt(bone_feature_filename, dtype=str).tolist()
    my_pkl_data_final = my_pkl_data[motions_cols]

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