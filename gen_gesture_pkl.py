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
from utils.logging_mixin import gesture_feats_to_bvh


def write_bvh(bvh_data, fname):
    writer = BVHWriter()
    with open(fname, 'w') as f:
        writer.write(bvh_data, f)


def process_motion(bvh_filename, motions_cols, save_path, all_files):
    motion_name = os.path.basename(bvh_filename)
    motion_name = motion_name.split('.')[0]

    # 跳过已经有的
    temp_name = motion_name + '_00'
    all_files.append(temp_name)
    save_name_1 = os.path.join(save_path, motion_name + '_00.expmap_30fps.pkl')
    if os.path.isfile(save_name_1):
        return

    # 加载bvh文件
    bvh_parser = BVHParser()
    bvh_data = bvh_parser.parse(bvh_filename)
    # 这算成30fps
    bvh_data.values = bvh_data.values[::2]
    bvh_data.framerate = bvh_data.framerate * 2
    # 初始站位归零
    bvh_data.values['Hips_Xposition'] -= bvh_data.values['Hips_Xposition'][0]
    bvh_data.values['Hips_Zposition'] -= bvh_data.values['Hips_Zposition'][0]
    # 保存一下
    bvh_datas = [bvh_data]

    # 算alpha、beta、gamma参数
    parameterizer = MocapParameterizer('expmap')
    expmap_datas = parameterizer.fit_transform(bvh_datas)
    expmap_data = expmap_datas[0]
    full_pkl_data = expmap_data.values

    # TODO dxposition、dzposition、dyrotation

    # 实际训练的的骨骼
    panda_data = full_pkl_data[motions_cols]

    # 写pkl
    with open(save_name_1, 'wb') as pkl_f1:
        pkl.dump(panda_data, pkl_f1)

    dotest = True
    if dotest:
        # 加载pkl
        with open(save_name_1, 'rb') as ff:
            reload_panda_data = pkl.load(ff).astype('float32')
        diff = reload_panda_data - panda_data

        # pkl_data转回bvh
        my_clips = reload_panda_data.values[np.newaxis, ...]
        dataset_root = './data/my_gesture_data/'
        my_bvh_datas = gesture_feats_to_bvh(my_clips, dataset_root)
        my_bvh_data = my_bvh_datas[0]
        my_bvh_diff = my_bvh_data.values[bvh_data.values.columns].values - bvh_data.values[bvh_data.values.columns].values
        my_bvh_data.values['Hips_Xposition'] += bvh_data.values['Hips_Xposition'][0] - my_bvh_data.values['Hips_Xposition'][0]
        my_bvh_data.values['Hips_Zposition'] += bvh_data.values['Hips_Zposition'][0] - my_bvh_data.values['Hips_Zposition'][0]
        write_bvh(my_bvh_data, f'./{motion_name}.bvh')
    return



def process_new_dataset():
    save_path = './data/my_gesture_data/'

    bone_feature_filename = './data/my_gesture_data/pose_features.expmap.txt'
    motions_cols = np.loadtxt(bone_feature_filename, dtype=str).tolist()

    # 处理test
    all_files = []
    motion_files = glob.glob('./data/my_gesture_data/GENEA/test/motion/*.bvh')
    motion_files.sort()
    for bvh_filename in tqdm.tqdm(motion_files):
        print("Process %s" % bvh_filename)
        process_motion(bvh_filename, motions_cols, save_path, all_files)
    # 处理train


if __name__ == "__main__":
    process_new_dataset()