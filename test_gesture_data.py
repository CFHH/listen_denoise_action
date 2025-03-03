import os
import shutil
import tqdm
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle as pkl
from scipy import interpolate
import librosa
import copy
import joblib as jl
import pandas as pd
from pymo.data import MocapData
from pymo.parsers import BVHParser
from pymo.writers import BVHWriter
from pymo.preprocessing import MocapParameterizer, RootTransformer


def dataframe_nansinf2zeros(df):
    df.fillna(0, inplace=True)  # 用0填充NA/NaN
    df.replace([np.inf, -np.inf], 0, inplace=True) # 用0填充np.inf和-np.inf
    return df


def dance_feats_to_bvh(pred_clips):
    # import pdb;pdb.set_trace()
    data_pipeline = jl.load('./data/motorica_dance/data_pipe.expmap_30fps.sav')
    n_feats = data_pipeline["cnt"].n_features
    data_pipeline["root"].separate_root = False
    print('inverse_transform...')
    bvh_data = data_pipeline.inverse_transform(pred_clips[:, :, :n_feats])
    return bvh_data


def gesture_feats_to_bvh(pred_clips, parameterizer, motions_cols, mocap_data_sample):
    ignored_joints = ['RightToeBase', 'RightForeFoot', 'LeftToeBase', 'LeftForeFoot', 'pCube4']
    hands = ['LeftHand', 'RightHand']
    fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    indexs = [1, 2, 3]
    for hand in hands:
        for finger in fingers:
            for index in indexs:
                name = f'{hand}{finger}{index}'
                ignored_joints.append(name)

    ignored_columns = []
    # 以下是parameterizer需要的
    for joint in ignored_joints:
        ignored_columns.append(joint + '_alpha')
        ignored_columns.append(joint + '_beta')
        ignored_columns.append(joint + '_gamma')

    full_columns = motions_cols + ignored_columns

    clip = pred_clips[0]
    frames = clip.shape[0]

    zeros = np.zeros((frames, len(ignored_columns)))
    channels = np.concatenate([clip, zeros], axis=-1)

    fps = 30
    time_list = [i / fps for i in range(frames)]  # 以秒计算的各帧时间，shape=(frame,)
    time_index = pd.to_timedelta(time_list, unit='s')  # 转成panda的数据
    panda_data = pd.DataFrame(data=channels, index=time_index, columns=full_columns)  # 就是pkl

    new_data = MocapData()
    new_data.skeleton = copy.deepcopy(mocap_data_sample.skeleton)
    new_data.channel_names = copy.deepcopy(mocap_data_sample.channel_names)
    new_data.root_name = copy.deepcopy(mocap_data_sample.root_name)
    new_data.values = panda_data
    new_data.framerate = 1/fps
    #new_data.take_name = ''

    new_datas = [new_data]
    my_bvh_datas = parameterizer.inverse_transform(new_datas)
    return my_bvh_datas


def write_bvh(bvh_data, fname):
    writer = BVHWriter()
    with open(fname, 'w') as f:
        writer.write(bvh_data, f)


def process_motion():
    # 一些文件
    bvh_filename = './data/speech_gesture/TestSeq010.bvh'
    bone_feature_filename = './data/speech_gesture/pose_features.expmap.txt'

    # 加载bvh文件
    bvh_parser = BVHParser()
    bvh_data = bvh_parser.parse(bvh_filename)
    # 这算成30fps
    bvh_data.values = bvh_data.values[::2]
    bvh_data.framerate = bvh_data.framerate * 2
    # 初始站位归零
    bvh_data.values['Hips_Xposition'] = bvh_data.values['Hips_Xposition'] - bvh_data.values['Hips_Xposition'][0]
    bvh_data.values['Hips_Zposition'] = bvh_data.values['Hips_Zposition'] - bvh_data.values['Hips_Zposition'][0]
    # 保存一下
    write_bvh(bvh_data, './raw.bvh')
    bvh_datas = [bvh_data]

    # 取alpha、beta、gamma参数
    parameterizer = MocapParameterizer('expmap')
    expmap_datas = parameterizer.fit_transform(bvh_datas)
    expmap_data = expmap_datas[0]
    my_pkl_data = expmap_data.values  # 没有['reference_dXposition']
    # 测试
    test_bvh_datas_1 = parameterizer.inverse_transform(expmap_datas)
    bvh_diff_1 = test_bvh_datas_1[0].values - bvh_data.values

    # 实际使用的骨骼
    motions_cols = np.loadtxt(bone_feature_filename, dtype=str).tolist()
    my_pkl_data_final = my_pkl_data[motions_cols]

    #pkl_data转回bvh
    my_clips = my_pkl_data_final.values[np.newaxis, ...]
    is_dance_data = False
    if is_dance_data:
        my_bvh_datas = dance_feats_to_bvh(my_clips)
    else:
        my_bvh_datas = gesture_feats_to_bvh(my_clips, parameterizer, motions_cols, expmap_data)
    my_bvh_data = my_bvh_datas[0]
    #my_bvh_diff = my_bvh_data.values[bvh_data.values.columns].values - bvh_data.values[bvh_data.values.columns].values
    my_bvh_data.values['Hips_Xposition'] += bvh_data.values['Hips_Xposition'][0] - my_bvh_data.values['Hips_Xposition'][0]
    my_bvh_data.values['Hips_Zposition'] += bvh_data.values['Hips_Zposition'][0] - my_bvh_data.values['Hips_Zposition'][0]
    write_bvh(my_bvh_data, './mine.bvh')

    return


if __name__ == "__main__":
    process_motion()