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
from utils.logging_mixin import custom_feats_to_bvh, roottransformer_method, roottransformer_separate_root
from gen_music_pkl import process_audio
from pymo.pipeline import get_pipeline, transform, transform2pkl, inverse_transform


def write_bvh(bvh_data, fname):
    writer = BVHWriter()
    with open(fname, 'w') as f:
        writer.write(bvh_data, f)


def mirror(bvh_data):
    all_joints = ['RightFoot', 'RightLeg', 'RightUpLeg',
                  'LeftFoot', 'LeftLeg', 'LeftUpLeg',
                  'RightHand', 'RightForeArm', 'RightArm', 'RightShoulder',
                  'LeftHand', 'LeftForeArm', 'LeftArm', 'LeftShoulder',
                  'Head', 'Neck1', 'Neck', 'Spine3', 'Spine2', 'Spine1', 'Spine', 'Hips']
    left_right_joints = ['Foot', 'Leg', 'UpLeg', 'Hand', 'ForeArm', 'Arm', 'Shoulder']

    # 平移：x=-x，z=-z，y不变
    bvh_data.values['Hips_Xposition'] *= -1
    bvh_data.values['Hips_Zposition'] *= -1

    # 旋转：1、左右互换；2、所有节点，x旋转不变，y旋转变相反数，z旋转变相反数
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

    return bvh_data


def process_motion(bvh_filename, motions_cols, save_path, all_files, process_mirror=True):
    motion_name = os.path.basename(bvh_filename)
    motion_name = motion_name.split('.')[0]

    # 跳过已经有的
    temp_name = motion_name + '_gSP_00'
    all_files.append(temp_name)
    save_name_1 = os.path.join(save_path, temp_name + '.expmap_30fps.pkl')
    if process_mirror:
        temp_name = motion_name + '_gSP_00_mirrored'
        all_files.append(temp_name)
        save_name_2 = os.path.join(save_path, temp_name + '.expmap_30fps.pkl')

    if process_mirror:
        if os.path.isfile(save_name_1) and os.path.isfile(save_name_2):
            return motion_name
    else:
        if os.path.isfile(save_name_1):
            return motion_name

    ####################################################################################################################

    # 加载bvh文件
    bvh_parser = BVHParser()
    bvh_data = bvh_parser.parse(bvh_filename)
    # 折算成30fps
    bvh_data.values = bvh_data.values[::2]
    bvh_data.framerate = 1 / 30
    # 初始站位归零
    bvh_data.values['Hips_Xposition'] -= bvh_data.values['Hips_Xposition'][0]
    bvh_data.values['Hips_Zposition'] -= bvh_data.values['Hips_Zposition'][0]
    # 保存一下
    bvh_datas = [bvh_data]

    use_v1 = False
    if use_v1:
        # 算alpha、beta、gamma参数
        parameterizer = MocapParameterizer('expmap')
        expmap_datas = parameterizer.fit_transform(bvh_datas)
        expmap_data = expmap_datas[0]
        full_pkl_data = expmap_data.values

        # TODO dxposition、dzposition、dyrotation
        root_transformer = RootTransformer(roottransformer_method, separate_root=roottransformer_separate_root)
        trans_datas = root_transformer.fit_transform(bvh_datas)
        trans_data = trans_datas[0]
        if roottransformer_method == 'abdolute_translation_deltas':
            full_pkl_data['Hips_dXposition'] = trans_data.values['Hips_dXposition']
            full_pkl_data['Hips_dZposition'] = trans_data.values['Hips_dZposition']

        # 实际训练的的骨骼
        panda_data = full_pkl_data[motions_cols]
    else:
        pipe = get_pipeline('GENEA')
        mocap_datas = transform2pkl(pipe, bvh_datas)
        panda_data = mocap_datas[0].values[motions_cols]

    # 写pkl
    with open(save_name_1, 'wb') as pkl_f1:
        pkl.dump(panda_data, pkl_f1)

    ####################################################################################################################

    # 镜像
    if process_mirror:
        mirror_bvh_data = mirror(bvh_data)
        mirror_bvh_datas = [mirror_bvh_data]
        if use_v1:
            pass
        else:
            mirror_mocap_datas = transform2pkl(pipe, mirror_bvh_datas)
            mirror_panda_data = mirror_mocap_datas[0].values[motions_cols]

        with open(save_name_2, 'wb') as pkl_f2:
            pkl.dump(mirror_panda_data, pkl_f2)

    ####################################################################################################################

    dotest = False
    if dotest:
        if process_mirror:
            write_bvh(mirror_bvh_data, f'./{motion_name}_mirror_raw.bvh') # 这个是对的
            # 加载pkl
            with open(save_name_2, 'rb') as ff:
                reload_panda_data = pkl.load(ff).astype('float32')
            #diff = reload_panda_data - panda_data

            # pkl_data转回bvh
            my_clips = reload_panda_data.values[np.newaxis, ...]
            dataset_root = './data/my_gesture_data/'
            my_bvh_datas = custom_feats_to_bvh(my_clips, dataset_root, 'GENEA')
            my_bvh_data = my_bvh_datas[0]
            #my_bvh_diff = my_bvh_data.values[bvh_data.values.columns].values - bvh_data.values[bvh_data.values.columns].values
            my_bvh_data.values['Hips_Xposition'] += bvh_data.values['Hips_Xposition'][0] - my_bvh_data.values['Hips_Xposition'][0]
            my_bvh_data.values['Hips_Zposition'] += bvh_data.values['Hips_Zposition'][0] - my_bvh_data.values['Hips_Zposition'][0]
            write_bvh(my_bvh_data, f'./{motion_name}_mirror_pkl.bvh') # 这个不一样，所以inverse还是有问题
        else:
            write_bvh(bvh_data, f'./{motion_name}_raw.bvh')
            # 加载pkl
            with open(save_name_1, 'rb') as ff:
                reload_panda_data = pkl.load(ff).astype('float32')
            # diff = reload_panda_data - panda_data

            # pkl_data转回bvh
            my_clips = reload_panda_data.values[np.newaxis, ...]
            dataset_root = './data/my_gesture_data/'
            my_bvh_datas = custom_feats_to_bvh(my_clips, dataset_root, 'GENEA')
            my_bvh_data = my_bvh_datas[0]
            # my_bvh_diff = my_bvh_data.values[bvh_data.values.columns].values - bvh_data.values[bvh_data.values.columns].values
            my_bvh_data.values['Hips_Xposition'] += bvh_data.values['Hips_Xposition'][0] - \
                                                    my_bvh_data.values['Hips_Xposition'][0]
            my_bvh_data.values['Hips_Zposition'] += bvh_data.values['Hips_Zposition'][0] - \
                                                    my_bvh_data.values['Hips_Zposition'][0]
            write_bvh(my_bvh_data, f'./{motion_name}_pkl.bvh')  # 这个不一样，所以inverse还是有问题

    return motion_name


def process_paired_dataset():
    save_path = './data/my_gesture_data/'

    bone_feature_filename = './data/my_gesture_data/pose_features.expmap.txt'
    motions_cols = np.loadtxt(bone_feature_filename, dtype=str).tolist()

    """
    all_files = []
    bvh_filename = './data/my_gesture_data/GENEA/test/motion/TestSeq010.bvh'
    motion_name = process_motion(bvh_filename, motions_cols, save_path, all_files, process_mirror=False)
    """

    sub_dataset_names = ['test', 'train']
    for sub_dataset_name in sub_dataset_names:
        all_files = []
        motion_files = glob.glob(f'./data/my_gesture_data/GENEA/{sub_dataset_name}/motion/*.bvh')
        motion_files.sort()
        for bvh_filename in tqdm.tqdm(motion_files):
            print("Process %s" % bvh_filename)
            motion_name = process_motion(bvh_filename, motions_cols, save_path, all_files, process_mirror=False)
            audio_file_name = f'./data/my_gesture_data/GENEA/{sub_dataset_name}/audio/{motion_name}.wav'
            process_audio(audio_file_name, save_path, None, align_to_raw_data=False, process_mirror=True, genra='_gSP')
            #break

        save_list_name = os.path.join(save_path, f'dance_{sub_dataset_name}_files.txt')
        with open(save_list_name, 'w') as f:
            for line in all_files:
                f.write(line + '\n')
        shutil.copy(save_list_name, os.path.join(save_path, f'dance_{sub_dataset_name}_files_kth.txt'))

    return


def process_new_dataset():
    save_path = './data/my_speech_for_eval/'
    audio_file_name = './data/my_speech_for_eval/rap小孔小孩demo2copy.wav'
    process_audio(audio_file_name, save_path, None, align_to_raw_data=False, process_mirror=False, genra='_gSP')



if __name__ == "__main__":
    #process_paired_dataset()
    process_new_dataset()