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
from pytorch_lightning import seed_everything
from helper.smpl_bvh_loader import load_bvh_motion
import soundfile


dance_names = [
    '1999',
    'AlcoholFree',
    'bbam',
    'BBoomBBoom',
    'BONBON',
    'BOSS',
    'BackToTheFuture',
    'BangBangBang',
    'Bday',
    'BloomingDay',
    'ComeSeeMe',
    'EasyLove',
    'ElectricKiss',
    'Energetic',
    'FANCY',
    'FIESTA',
    'Girls',
    'GoodTime',
    'HighHeel',
    'HotPink',
    'HystericBullet',
    'IAmTheBest',
    'LoveitLoveit',
    'MASAYUMECHASING',
    'Magic',
    'PICKME',
    'PiNKCAT',
    'SINGULARITY',
    'Seve',
    'ShakeIt',
    'StrayKids',
    'TeenTop',
    'Thumbsup',
    'USA',
    'Updown',
    'babe',
    'bbam',
    'dingga',
    'fantasticbaby',
    'hip',
    'im_not_cool',
    'odd_eye',
    'perfume',
    'phuthon',
    'play',
    #'sheidoubielinse',
    'violeta',
    'wadada',
    'werock',
    '创造101',
    '卡路里',
    '致青春',
    '大艺术家',
    '梦想之门',
    '舞可替代',
    '虎视眈眈',
    '響喜乱舞',
    '偶像万万岁',
    '和我交往吗',
    '第三类接触',
    '舞い落ちる花びら',
]

music_dirs = [
    './data/frankenstein/raw_audio/prompt_0',
    './data/frankenstein/raw_audio/prompt_1',
    './data/frankenstein/raw_audio/prompt_2',
    './data/frankenstein/raw_audio/prompt_3',
    './data/frankenstein/raw_audio/prompt_4',
]


def load_music(music_file, fps=60):
    print(f'loading music: {music_file} ...')
    HOP_LENGTH = 512
    LOAD_FPS = fps
    SR = LOAD_FPS * HOP_LENGTH
    data, _ = librosa.load(music_file, sr=SR)
    duration = librosa.get_duration(y=data, sr=SR)
    frames = (data.shape[0] / SR) * LOAD_FPS
    return data, SR


def frankenstein_music():
    bvh_save_path = './data/frankenstein/bvh'   # bvh拷贝到这里
    wav_save_path = './data/frankenstein/wav'   # 拼接后的音乐放这里
    fps = 30

    # 音乐列表
    music_files_list = []
    music_file_len_list = []
    for music_dir in music_dirs:
        files = glob.glob(os.path.join(music_dir, '*.mp3'))
        files.sort()  # 顺序固定
        music_files_list.append(files)
        music_file_len_list.append(len(files))
    seed_everything(999)
    #random.shuffle(music_list)  # 不打乱
    cur_music_idx = []  # 下次该读哪个音乐
    for i in range(len(music_files_list)):
        cur_music_idx.append(0)

    # 每段音乐都是140bpm，长度在8小节，略有误差，60 / 140 * 4 * 8 = 13.714秒，30fps是411.429帧
    expected_music_duration = 60 / 140 * 4 * 8
    expected_music_frames = expected_music_duration * fps  # 不是整数

    # 动作列表
    bvh_dir = 'I:/vq_action/data/2_unified/skjx_bvh_bpm140_fps30'
    dance_cnt = len(dance_names)
    for i in range(dance_cnt):
        dance_name = dance_names[i]
        print(f'processing ({i}/{dance_cnt}) {dance_name} ...')
        bvh_filename = os.path.join(bvh_dir, f'{dance_name}.bvh')
        if not os.path.isfile(bvh_filename):
            print(f'{dance_name} does not exist')

        # 先拷贝
        copy_filename = os.path.join(bvh_save_path, f'{dance_name}.bvh')
        shutil.copy(bvh_filename, copy_filename)

        # 加载bvh
        root_position, rotation, frametime, name, parent, offsets = load_bvh_motion(copy_filename, True)
        motion_frames = root_position.shape[0]
        music_cnt = int(round(motion_frames / expected_music_frames))

        # 合并音乐
        save_wav_file_name = os.path.join(wav_save_path, f'{dance_name}.wav')
        music_style = i % len(music_files_list)
        music_list = music_files_list[music_style]
        begin_idx = cur_music_idx[music_style]
        print(f'choosing music style {music_style}, from index = {begin_idx}, need {music_cnt}')
        if begin_idx + music_cnt > len(music_list):
            print(f'{dance_name} not enough music!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            break

        music_data_list = []
        chosen_cnt = 0

        music_idx = begin_idx
        while chosen_cnt < music_cnt:
            data, SR = load_music(music_list[music_idx])
            music_idx += 1
            expected_len = int(expected_music_duration * SR)
            if len(data) < expected_len:
                print(f'{music_list[music_idx]} is too short')
                continue
            chosen_cnt += 1
            slice_data = data[0:expected_len]
            music_data_list.append(slice_data)

        merged_audio_data = np.concatenate(music_data_list, axis=0)
        soundfile.write(save_wav_file_name, merged_audio_data, SR)
        cur_music_idx[music_style] = music_idx

    return


if __name__ == "__main__":
    # 1、把bvh转成140bpm（因为音乐是140bpm的，已经验证过）
    #    这个在vq_action中完成
    # 2、根据舞蹈长度，把音乐拼接成对应长度的片段（据说每段音乐都是8小节，这个不确定）
    frankenstein_music()
    # 3、回到老路上
