import os
import shutil
import tqdm
import glob
import numpy as np
import random
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
from utils.logging_mixin import custom_feats_to_bvh, roottransformer_method, roottransformer_separate_root
from gen_music_pkl import process_audio
from pymo.pipeline import get_pipeline, transform, transform2pkl, inverse_transform
import codecs


def write_bvh(bvh_data, fname):
    writer = BVHWriter()
    with open(fname, 'w') as f:
        writer.write(bvh_data, f)


def mirror(bvh_data, skeleton_type):
    if skeleton_type == 'default':
        pass
    elif skeleton_type == 'GENEA':
        all_joints = ['RightFoot', 'RightLeg', 'RightUpLeg',
                      'LeftFoot', 'LeftLeg', 'LeftUpLeg',
                      'RightHand', 'RightForeArm', 'RightArm', 'RightShoulder',
                      'LeftHand', 'LeftForeArm', 'LeftArm', 'LeftShoulder',
                      'Head', 'Neck1', 'Neck', 'Spine3', 'Spine2', 'Spine1', 'Spine', 'Hips']
        left_right_joints = ['Foot', 'Leg', 'UpLeg', 'Hand', 'ForeArm', 'Arm', 'Shoulder']
        root_joint = 'Hips'
        str_left = 'Left'
        str_right = 'Right'
    elif skeleton_type == 'smpl':
        all_joints = ['pelvis',
                      'l_hip', 'l_knee', 'l_ankle', 'l_foot',
                      'r_hip', 'r_knee', 'r_ankle', 'r_foot',
                      'spine1', 'spine2', 'spine3', 'neck', 'head',
                      'l_collar', 'l_shoulder', 'l_elbow', 'l_wrist', 'l_hand',
                      'r_collar', 'r_shoulder', 'r_elbow', 'r_wrist', 'r_hand']
        left_right_joints = ['hip', 'knee', 'ankle', 'foot', 'collar', 'shoulder', 'elbow', 'wrist', 'hand']
        root_joint = 'pelvis'
        str_left = 'l_'
        str_right = 'r_'

    # 平移：x=-x，z=-z，y不变
    bvh_data.values[f'{root_joint}_Xposition'] *= -1
    #bvh_data.values[f'{root_joint}_Zposition'] *= -1  # TODO vq_action的需要再验证一次

    # 旋转：1、左右互换；2、所有节点，x旋转不变，y旋转变相反数，z旋转变相反数
    for name in left_right_joints:
        for r in ['_Xrotation', '_Yrotation', '_Zrotation']:
            left_key = str_left + name + r
            right_key = str_right + name + r
            temp = bvh_data.values[left_key]
            bvh_data.values[left_key] = bvh_data.values[right_key]
            bvh_data.values[right_key] = temp
    for name in all_joints:
        for r in ['_Yrotation', '_Zrotation']:
            key = name + r
            bvh_data.values[key] *= -1

    return bvh_data


def process_motion(bvh_filename, fps, motions_cols, dataset_root, save_path, all_files, process_mirror=True, genra=None):
    motion_name = os.path.basename(bvh_filename)
    motion_name = motion_name.split('.')[0]

    # 生成最后的文件名
    temp_name = motion_name
    if genra is not None:
        temp_name = temp_name + genra + '_00'  # 模拟符合命名规范
    if all_files is not None:
        all_files.append(temp_name)
    save_name_1 = os.path.join(save_path, temp_name + '.expmap_30fps.pkl')
    if process_mirror:
        temp_name = temp_name + '_mirrored'
        if all_files is not None:
            all_files.append(temp_name)
        save_name_2 = os.path.join(save_path, temp_name + '.expmap_30fps.pkl')

    # 跳过已经有的
    if process_mirror:
        if os.path.isfile(save_name_1) and os.path.isfile(save_name_2):
            return
    else:
        if os.path.isfile(save_name_1):
            return

    ####################################################################################################################
    skeleton_type = 'smpl'
    root_joint = 'pelvis'

    # 加载bvh文件
    bvh_parser = BVHParser()
    bvh_data = bvh_parser.parse(bvh_filename)
    # 折算成30fps
    assert fps == 30
    if fps != 30:
        #bvh_data.values = bvh_data.values[::2]  # TODO 具体情况具体改吧，肯定不可能去插值的
        pass
    bvh_data.framerate = 1 / 30
    # 初始站位归零
    bvh_data.values[f'{root_joint}_Xposition'] -= bvh_data.values[f'{root_joint}_Xposition'][0]
    bvh_data.values[f'{root_joint}_Zposition'] -= bvh_data.values[f'{root_joint}_Zposition'][0]

    # 转成pkl数据
    bvh_datas = [bvh_data]
    pipe = get_pipeline(skeleton_type)
    mocap_datas = transform2pkl(pipe, bvh_datas)
    panda_data = mocap_datas[0].values[motions_cols]

    # 写pkl
    with open(save_name_1, 'wb') as pkl_f1:
        pkl.dump(panda_data, pkl_f1)

    dotest = False
    if dotest:
        write_bvh(bvh_data, f'./{motion_name}_raw.bvh')
        # pkl_data转回bvh
        with open(save_name_1, 'rb') as ff:
            reload_panda_data = pkl.load(ff).astype('float32')
        my_clips = reload_panda_data.values[np.newaxis, ...]
        my_bvh_datas = custom_feats_to_bvh(my_clips, dataset_root, skeleton_type)
        my_bvh_data = my_bvh_datas[0]
        my_bvh_data.values[f'{root_joint}_Xposition'] -= my_bvh_data.values[f'{root_joint}_Xposition'][0]
        my_bvh_data.values[f'{root_joint}_Zposition'] -= my_bvh_data.values[f'{root_joint}_Zposition'][0]
        write_bvh(my_bvh_data, f'./{motion_name}_pkl.bvh')

    ####################################################################################################################

    # 镜像
    if process_mirror:
        mirror_bvh_data = mirror(bvh_data, skeleton_type)
        mirror_bvh_datas = [mirror_bvh_data]

        mirror_mocap_datas = transform2pkl(pipe, mirror_bvh_datas)
        mirror_panda_data = mirror_mocap_datas[0].values[motions_cols]

        with open(save_name_2, 'wb') as pkl_f2:
            pkl.dump(mirror_panda_data, pkl_f2)

        if dotest:
            write_bvh(mirror_bvh_data, f'./{motion_name}_mirror_raw.bvh')
            # pkl_data转回bvh
            with open(save_name_2, 'rb') as ff:
                reload_panda_data = pkl.load(ff).astype('float32')
            my_clips = reload_panda_data.values[np.newaxis, ...]
            my_bvh_datas = custom_feats_to_bvh(my_clips, dataset_root, skeleton_type)
            my_bvh_data = my_bvh_datas[0]
            my_bvh_data.values[f'{root_joint}_Xposition'] -= my_bvh_data.values[f'{root_joint}_Xposition'][0]
            my_bvh_data.values[f'{root_joint}_Zposition'] -= my_bvh_data.values[f'{root_joint}_Zposition'][0]
            write_bvh(my_bvh_data, f'./{motion_name}_mirror_pkl.bvh')

    ####################################################################################################################

    return


def process_dataset():
    """
    用一份新骨骼需要做的事情：
    1、skeleton.bvh
    2、pose_features.expmap.txt
    3、pipeline.py, get_pipeline()
    4、*.yaml
    5、logging_mixin.py, custom_feats_to_bvh
    6、镜像算法

    需要重新定义一份骨骼
    骨骼旋转顺序到处要改
    根骨骼名字pelvis

    命名规范
    预处理后的动作和音频数据的文件名，第一个_后是风格，最后一个_后是00；如果是镜像，再接一个_mirrored
    用于eval的音频文件，必须以_风格结尾
    """
    # 原始数据集
    bvh_path = 'I:/vq_action/data/1_aligned/skjx/bvh_bpm0_fps30/'
    wav_path = 'I:/vq_action/data/1_aligned/skjx/music-aligned/'
    fps = 30
    # 训练数据集
    dataset_root = './data/smpl_dance/'
    save_path = dataset_root
    eval_path = './data/eval_for_smpl_dance/'

    # 拆分数据集(skjx共255首，225首用来train，20首用来test，10首用来eval)
    motion_files = glob.glob(os.path.join(bvh_path, '*.bvh'))
    motion_cnt = len(motion_files)
    eval_cnt = int(motion_cnt * 0.04)
    test_cnt = int(motion_cnt * 0.08)
    train_cnt = motion_cnt - test_cnt - eval_cnt
    test_eval_files = random.sample(motion_files, eval_cnt + test_cnt)
    for i in test_eval_files:
        motion_files.remove(i)
    train_files = motion_files
    train_files.sort()
    test_files = test_eval_files[0:test_cnt]
    test_files.sort()
    eval_files = test_eval_files[test_cnt:]
    eval_files.sort()

    #
    encoder = codecs.getincrementalencoder('utf-8')()
    bone_feature_filename = os.path.join(dataset_root, 'pose_features.expmap.txt')
    motion_columns = np.loadtxt(bone_feature_filename, dtype=str).tolist()

    def _bvh_filename_2_wav_filename(bvh_filename):
        base_name = os.path.basename(bvh_filename)
        base_name = base_name.split('.')[0]
        wav_filename = os.path.join(wav_path, f'{base_name}.wav')
        return wav_filename, base_name

    genra = '_gOK'
    # train
    all_files = []
    for bvh_filename in tqdm.tqdm(train_files):
        print("Process TRAIN: %s" % bvh_filename)
        wav_filename, base_name = _bvh_filename_2_wav_filename(bvh_filename)
        if '_' in base_name:
            continue  # 没法去模拟符合命名规范了
        process_motion(bvh_filename, fps, motion_columns, dataset_root, save_path, all_files, process_mirror=True, genra=genra)
        process_audio(wav_filename, save_path, None, align_to_raw_data=False, process_mirror=True, genra=genra)
        #break
    save_list_name = os.path.join(save_path, 'dance_train_files.txt')
    with open(save_list_name, 'wb') as f:
        for line in all_files:
            line = line + '\n'
            data = encoder.encode(line)
            f.write(data)
    shutil.copy(save_list_name, os.path.join(save_path, 'dance_train_files_kth.txt'))

    # test
    all_files = []
    for bvh_filename in tqdm.tqdm(test_files):
        print("Process TEST: %s" % bvh_filename)
        wav_filename, base_name = _bvh_filename_2_wav_filename(bvh_filename)
        if '_' in base_name:
            continue
        process_motion(bvh_filename, fps, motion_columns, dataset_root, save_path, all_files, process_mirror=True, genra=genra)
        process_audio(wav_filename, save_path, None, align_to_raw_data=False, process_mirror=True, genra=genra)
        #break
    save_list_name = os.path.join(save_path, 'dance_test_files.txt')
    with open(save_list_name, 'wb') as f:
        for line in all_files:
            line = line + '\n'
            data = encoder.encode(line)
            f.write(data)
    shutil.copy(save_list_name, os.path.join(save_path, 'dance_test_files_kth.txt'))

    # eval
    all_files = []
    for bvh_filename in tqdm.tqdm(eval_files):
        print("Process EVAL: %s" % bvh_filename)
        wav_filename, base_name = _bvh_filename_2_wav_filename(bvh_filename)
        if '_' in base_name:
            continue
        copy_file_name = os.path.join(eval_path, f'{base_name}{genra}.wav')
        if not os.path.isfile(copy_file_name):
            shutil.copy(wav_filename, copy_file_name)
        process_audio(copy_file_name, eval_path, all_files, align_to_raw_data=False, process_mirror=False, genra='')
        #break
    save_list_name = os.path.join(eval_path, 'gen_files.txt')
    with open(save_list_name, 'wb') as f:
        for line in all_files:
            line = line + '\n'
            data = encoder.encode(line)
            f.write(data)

    # TODO 手工操作，把gen_files.txt里的相关文件从dataset_root里删除，不删其实也没事
    #      多次运行的最终状态是。。。

    return


def change_file_encoding():
    filename = './data/smpl_dance/dance_train_files.txt'
    filename = './data/smpl_dance/dance_test_files.txt'
    filename = './data/eval_for_smpl_dance/gen_files.txt'
    with open(filename) as f:
        all_files = f.readlines()

    encoder = codecs.getincrementalencoder('utf-8')()
    save_list_name = './test.txt'
    with open(save_list_name, 'wb') as f:
        for line in all_files:
            data = encoder.encode(line)
            f.write(data)

    with open(save_list_name, encoding='utf-8') as f:
        files = f.readlines()

    return


def process_other_wav_for_eval():
    import time
    eval_path = './data/eval_for_smpl_dance2/'
    all_files = []
    music_files = glob.glob(os.path.join(eval_path, '*.wav'))
    music_files.sort()
    for file_name in tqdm.tqdm(music_files):
        print("Processing %s ......" % file_name)
        start_time = time.time()
        result = process_audio(file_name, eval_path, all_files, align_to_raw_data=False, process_mirror=False, genra='')
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Process {file_name} (duration {result['duration']} seconds) cost {elapsed_time} seconds")

    encoder = codecs.getincrementalencoder('utf-8')()
    save_list_name = os.path.join(eval_path, 'gen_files.txt')
    with open(save_list_name, 'wb') as f:
        for line in all_files:
            line = line + '\n'
            data = encoder.encode(line)
            f.write(data)
    return


if __name__ == "__main__":
    #change_file_encoding()
    #process_dataset()
    process_other_wav_for_eval()