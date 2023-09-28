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
import copy
from pymo.data import MocapData
from pymo.parsers import BVHParser
from pymo.writers import BVHWriter
from pymo.preprocessing import JointSelector, RootTransformer, MocapParameterizer, ConstantsRemover, FeatureCounter, Numpyfier
from pymo.Quaternions import Quaternions
from pymo.pipeline import get_pipeline, transform, transform2pkl, inverse_transform


def dance_feats_to_bvh(pred_clips):
    # import pdb;pdb.set_trace()
    data_pipeline = jl.load('./data/motorica_dance/data_pipe.expmap_30fps.sav')
    n_feats = data_pipeline["cnt"].n_features
    data_pipeline["root"].separate_root = False
    print('inverse_transform...')
    bvh_data = data_pipeline.inverse_transform(pred_clips[:, :, :n_feats])
    return bvh_data


def gesture_feats_to_bvh(pred_clips, parameterizer, motions_cols, root_transformer, mocap_data_sample):
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
    # 以下3个是root_transformer需要的
    ignored_columns.append('Hips_Xposition')
    ignored_columns.append('Hips_Zposition')
    ignored_columns.append('reference_dYposition')
    ignored_columns.append('Hips_Yrotation')

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
    #root_transformer = RootTransformer('abdolute_translation_deltas', separate_root=False)
    temp_datas = parameterizer.inverse_transform(new_datas)
    my_bvh_datas = root_transformer.inverse_transform(temp_datas)
    # 这些没有啊
    if root_transformer.separate_root:
        my_bvh_datas[0].values['Hips_Xposition'] = my_bvh_datas[0].values['reference_Xposition']
        my_bvh_datas[0].values['Hips_Zposition'] = my_bvh_datas[0].values['reference_Zosition']
        my_bvh_datas[0].values['Hips_Yrotation'] = my_bvh_datas[0].values['reference_Yrotation']
    else:
        pass
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
    # xz
    bvh_data.values['Hips_Xposition'] = bvh_data.values['Hips_Xposition'] - bvh_data.values['Hips_Xposition'][0]
    bvh_data.values['Hips_Zposition'] = bvh_data.values['Hips_Zposition'] - bvh_data.values['Hips_Zposition'][0]
    # 保存一下
    write_bvh(bvh_data, './raw.bvh')
    bvh_datas = [bvh_data]

    # 取alpha、beta、gamma参数
    parameterizer = MocapParameterizer('expmap')
    expmap_datas = parameterizer.fit_transform(bvh_datas)
    expmap_data = expmap_datas[0]
    my_pkl_data = expmap_data.values  # (frame, 174)，有Hips_XYZposition
    # 测试
    test_bvh_datas_1 = parameterizer.inverse_transform(expmap_datas)
    bvh_diff_1 = test_bvh_datas_1[0].values - bvh_data.values

    # 取reference参数
    method = 'abdolute_translation_deltas' # pos_xyz_rot_deltas pos_rot_deltas
    root_transformer = RootTransformer(method, separate_root=False)
    trans_datas = root_transformer.fit_transform(bvh_datas)
    trans_data = trans_datas[0]
    if method == 'abdolute_translation_deltas':
        my_pkl_data['Hips_dXposition'] = trans_data.values['Hips_dXposition']
        my_pkl_data['Hips_dZposition'] = trans_data.values['Hips_dZposition']
    else:
        my_pkl_data['reference_dXposition'] = trans_data.values['reference_dXposition']
        my_pkl_data['reference_dZposition'] = trans_data.values['reference_dZposition']
        my_pkl_data['reference_dYrotation'] = trans_data.values['reference_dYrotation']
    #测试
    test_bvh_datas = root_transformer.inverse_transform(trans_datas)
    if method == 'abdolute_translation_deltas':
        test_bvh_data = test_bvh_datas[0]
        test_bvh_data.values['Hips_Xposition'] -= test_bvh_data.values['Hips_Xposition'][0]
        test_bvh_data.values['Hips_Zposition'] -= test_bvh_data.values['Hips_Zposition'][0]
        bvh_diff = test_bvh_data.values - bvh_data.values
    elif root_transformer.separate_root:
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
    motions_cols = np.loadtxt(bone_feature_filename, dtype=str).tolist()
    my_pkl_data_final = my_pkl_data[motions_cols]

    #pkl_data转回bvh
    my_clips = my_pkl_data_final.values[np.newaxis, ...]
    is_dance_data = False
    if is_dance_data:
        my_bvh_datas = dance_feats_to_bvh(my_clips)
    else:
        my_bvh_datas = gesture_feats_to_bvh(my_clips, parameterizer, motions_cols, root_transformer, expmap_data)
    my_bvh_data = my_bvh_datas[0]
    #my_bvh_diff = my_bvh_data.values[bvh_data.values.columns].values - bvh_data.values[bvh_data.values.columns].values
    my_bvh_data.values['Hips_Xposition'] += bvh_data.values['Hips_Xposition'][0] - my_bvh_data.values['Hips_Xposition'][0]
    my_bvh_data.values['Hips_Zposition'] += bvh_data.values['Hips_Zposition'][0] - my_bvh_data.values['Hips_Zposition'][0]
    write_bvh(my_bvh_data, './mine.bvh')

    return


def test_quat():
    rot_order = 'ZXY'
    erler = np.array([0, 0, -180], dtype=float)
    rotation = np.pi / 180.0 * erler
    quat = Quaternions.from_euler(rotation, order=rot_order.lower(), world=False)

    side_dir = quat * np.array([[-1.0, 0.0, 0.0]])
    forward = np.cross(np.array([[0.0, 1.0, 0.0]]), side_dir)

    return


def mirror_bvh():
    all_joints= ['RightFoot', 'RightLeg', 'RightUpLeg',
                 'LeftFoot', 'LeftLeg', 'LeftUpLeg',
                 'RightHand', 'RightForeArm', 'RightArm', 'RightShoulder',
                 'LeftHand', 'LeftForeArm', 'LeftArm', 'LeftShoulder',
                 'Head', 'Neck1', 'Neck', 'Spine3', 'Spine2', 'Spine1', 'Spine', 'Hips']
    left_right_joints = ['Foot', 'Leg', 'UpLeg', 'Hand', 'ForeArm', 'Arm', 'Shoulder']

    bvh_filename = './data/speech_gesture/TestSeq010.bvh'
    # 加载bvh文件
    bvh_parser = BVHParser()
    bvh_data = bvh_parser.parse(bvh_filename)
    # 这算成30fps
    bvh_data.values = bvh_data.values[::2]
    bvh_data.framerate = bvh_data.framerate * 2
    # xz
    bvh_data.values['Hips_Xposition'] = bvh_data.values['Hips_Xposition'] - bvh_data.values['Hips_Xposition'][0]
    bvh_data.values['Hips_Zposition'] = bvh_data.values['Hips_Zposition'] - bvh_data.values['Hips_Zposition'][0]

    # 平移：x=-x，z=-z，y不变
    bvh_data.values['Hips_Xposition'] *= -1

    bvh_data.values['Hips_Zposition'] *= -1
    # 旋转：1、左右互换；2、所有节点，x旋转不变，y旋转变相反数，z旋转变相反数。名字举例Hip_Zrotation、RightFoot_Xrotation
    for name in left_right_joints:
        for r in ['_Xrotation', '_Yrotation', '_Zrotation']:
            left_key = 'Left' + name + r
            right_key = 'Right' + name + r
            temp = bvh_data.values[left_key]
            bvh_data.values[left_key] = bvh_data.values[right_key]
            bvh_data.values[right_key] = temp
    for name in all_joints:
        for r in ['_Yrotation', '_Zrotation']:
            key = name + r
            bvh_data.values[key] *= -1


    # 保存一下
    write_bvh(bvh_data, './mirror.bvh')
    return


def test_pipeline1():
    # 一些文件
    bvh_filename = './data/speech_gesture/TestSeq010.bvh'

    # 加载bvh文件
    bvh_parser = BVHParser()
    bvh_data = bvh_parser.parse(bvh_filename)
    # 这算成30fps
    bvh_data.values = bvh_data.values[::2]
    bvh_data.framerate = 1 / 30
    # xz
    bvh_data.values['Hips_Xposition'] = bvh_data.values['Hips_Xposition'] - bvh_data.values['Hips_Xposition'][0]
    bvh_data.values['Hips_Zposition'] = bvh_data.values['Hips_Zposition'] - bvh_data.values['Hips_Zposition'][0]
    # 保存一下
    write_bvh(bvh_data, './raw.bvh')
    bvh_datas = [bvh_data]

    # pipeline
    pipe = get_pipeline(False)
    clips = transform(pipe, bvh_datas)
    my_bvh_datas = inverse_transform(pipe, clips)

    my_bvh_data = my_bvh_datas[0]
    my_bvh_diff = my_bvh_data.values[bvh_data.values.columns].values - bvh_data.values[bvh_data.values.columns].values
    my_bvh_data.values['Hips_Xposition'] += bvh_data.values['Hips_Xposition'][0] - my_bvh_data.values['Hips_Xposition'][0]
    my_bvh_data.values['Hips_Zposition'] += bvh_data.values['Hips_Zposition'][0] - my_bvh_data.values['Hips_Zposition'][0]
    write_bvh(my_bvh_data, './my_pipe.bvh')

    return


def warmup_pipeline(dataset_root):
    skeleton_bvh = os.path.join(dataset_root, 'skeleton.bvh')
    bvh_parser = BVHParser()
    bvh_data = bvh_parser.parse(skeleton_bvh)
    bvh_data.framerate = 1 / 30
    bvh_datas = [bvh_data]
    pipeline = get_pipeline(False)
    transform(pipeline, bvh_datas)
    return pipeline


def test_pipeline2():
    # 一些文件
    bvh_filename = './data/speech_gesture/TestSeq010.bvh'
    bone_feature_filename = './data/my_gesture_data/pose_features.expmap.txt'
    motions_cols = np.loadtxt(bone_feature_filename, dtype=str).tolist()

    # 加载bvh文件
    bvh_parser = BVHParser()
    bvh_data = bvh_parser.parse(bvh_filename)
    # 这算成30fps
    bvh_data.values = bvh_data.values[::2]
    bvh_data.framerate = 1 / 30
    # xz
    bvh_data.values['Hips_Xposition'] = bvh_data.values['Hips_Xposition'] - bvh_data.values['Hips_Xposition'][0]
    bvh_data.values['Hips_Zposition'] = bvh_data.values['Hips_Zposition'] - bvh_data.values['Hips_Zposition'][0]
    # 保存一下
    write_bvh(bvh_data, './raw.bvh')

    # 模拟到pkl文件
    bvh_datas = [bvh_data]
    pipe = get_pipeline(False)
    mocap_datas = transform2pkl(pipe, bvh_datas)
    pkl_data = mocap_datas[0].values[motions_cols]
    # 模拟加载pkl到训练生成数据
    clips = pkl_data.values[np.newaxis, ...]
    # 模拟把训练生成数据转成bvh
    pipe2 = warmup_pipeline('./data/my_gesture_data')
    my_bvh_datas = inverse_transform(pipe2, clips)

    my_bvh_data = my_bvh_datas[0]
    my_bvh_diff = my_bvh_data.values[bvh_data.values.columns].values - bvh_data.values[bvh_data.values.columns].values
    my_bvh_data.values['Hips_Xposition'] += bvh_data.values['Hips_Xposition'][0] - my_bvh_data.values['Hips_Xposition'][0]
    my_bvh_data.values['Hips_Zposition'] += bvh_data.values['Hips_Zposition'][0] - my_bvh_data.values['Hips_Zposition'][0]
    write_bvh(my_bvh_data, './my_pipe.bvh')

    return


if __name__ == "__main__":
    test_pipeline2()
    #process_motion()