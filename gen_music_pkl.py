import os
import shutil
import tqdm
import glob
import numpy as np
import pickle as pkl
import pandas as pd
import librosa
import madmom
from madmom.features import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
from functools import partial


class PreProcessor(madmom.processors.SequentialProcessor):
    """
    RNNDownBeatProcessor 的输入是文件名，不通过librosa，这里改动一下
    RNNDownBeatProcessor::__init__()中有pre_processor, nn, act，共3部分
    这里是pre_processor，直接取RNNDownBeatProcessor::__init__()里的pre_processor
    """
    origin = RNNDownBeatProcessor()
    def __init__(self, **kwargs):
        super(PreProcessor, self).__init__([self.origin.processors[0]])

class NNProcessor(madmom.processors.SequentialProcessor):
    """
    抄的RNNDownBeatProcessor::__init__()里的nn
    """
    def __init__(self, **kwargs):
        nn = madmom.ml.nn.NeuralNetworkEnsemble.load(madmom.models.DOWNBEATS_BLSTM, **kwargs)
        super(NNProcessor, self).__init__([nn])

class ActProcessor(madmom.processors.SequentialProcessor):
    """
    抄的RNNDownBeatProcessor::__init__()里的act
    """
    def __init__(self, **kwargs):
        act = partial(np.delete, obj=0, axis=1)
        super(ActProcessor, self).__init__([act])


def get_beat_activation(audio_file_name):
    HOP_LENGTH = 441
    MADMOM_FPS = 30
    MADMOM_SR = HOP_LENGTH * MADMOM_FPS

    data, _ = librosa.load(audio_file_name, sr=MADMOM_SR)
    duration = librosa.get_duration(y=data, sr=MADMOM_SR)  # y.shape[0] / MADMOM_SR
    frames = int(duration * MADMOM_FPS + 1)

    pre_proc = PreProcessor()
    feat = pre_proc(data)  # (帧数, 314)
    nn_prox = NNProcessor()
    act_proc = ActProcessor()
    beat_activation = act_proc(nn_prox(feat))  # (帧数, 2)

    track_proc = DBNDownBeatTrackingProcessor(beats_per_bar=4, min_bpm=80, max_bpm=215,
                                              num_tempi=60, transition_lambda=100,
                                              observation_lambda=16, threshold=0.05,
                                              correct=True, fps=MADMOM_FPS)
    track_res = track_proc(beat_activation)  # (帧数, 2)
    beats = track_res[:, 0] # track_res第0列是用时间表示的beat（单位是秒），注意不是帧
    beats_index = np.round(beats / (1 / MADMOM_FPS))
    beats_index = beats_index.astype(int)

    return beat_activation, beats_index

def get_spectral_flux(audio_file_name):
    HOP_LENGTH = 441
    MADMOM_FPS = 30
    MADMOM_SR = HOP_LENGTH * MADMOM_FPS
    sodf = madmom.features.onsets.SpectralOnsetProcessor(sample_rate=MADMOM_SR) # 默认参数就是 spectral_flux
    spectral_flux = sodf(audio_file_name)
    return spectral_flux

def get_chroma(audio_file_name):
    dcp = madmom.audio.chroma.DeepChromaProcessor()
    chroma = dcp(audio_file_name)
    return chroma


def process_audio(audio_file_name, save_path, all_files, align_to_raw_data=False, process_mirror=True, genra=''):
    """
    :param audio_file_name:
    :param save_path:
    :param all_files: 收集文件名，可以为None
    :param align_to_raw_data: 对已经有动作pkl的文件，指定为True，生成过程会试图与动作pkl的帧对齐
    :param process_mirror: 生成_mirrored文件
    :param genra: 有些文件自身不带类型，通过这个参数补充，比如'_gSP'
    :return: None

    一个wav
    kthjazz_gCH_sFM_cAll_d02_mCH_ch01_beatlestreetwashboardbandfortyandtight_003.wav
    生成2个完全相同的文件
    kthjazz_gCH_sFM_cAll_d02_mCH_ch01_beatlestreetwashboardbandfortyandtight_003_00.audio29_30fps.pkl
    kthjazz_gCH_sFM_cAll_d02_mCH_ch01_beatlestreetwashboardbandfortyandtight_003_00_mirrored.audio29_30fps.pkl
    主要数据
    mfcc:          20个，对应librosa的mfcc
    chroma:         6个，对应librosa的chroma_cens
    spectralflux:   1个，频谱流量，对应librosa的onset_strength
    Beatactivation: 1个，中间数据
    Beat:           1个，是拍还是小节？经过测试，只能是拍
    """
    audio_name = os.path.basename(audio_file_name)
    audio_name = audio_name.split('.')[0]

    # 跳过已经有的
    temp_name = audio_name + genra + '_00'
    if all_files is not None:
        all_files.append(temp_name)
    save_name_1 = os.path.join(save_path, temp_name + '.audio29_30fps.pkl')
    if process_mirror:
        temp_name = audio_name + genra + '_00_mirrored'
        if all_files is not None:
            all_files.append(temp_name)
        save_name_2 = os.path.join(save_path, temp_name + '.audio29_30fps.pkl')
    if process_mirror:
        if os.path.isfile(save_name_1) and os.path.isfile(save_name_2):
            return
    else:
        if os.path.isfile(save_name_1):
            return

    # 原数据集文件，目标是为了找个起始帧，好与动作数据集对齐
    if align_to_raw_data:
        raw_pkl_file = './data/motorica_dance/%s_00.audio29_30fps.pkl' % audio_name
        with open(raw_pkl_file, 'rb') as f:
            raw_panda_data = pkl.load(f).astype('float32')
        raw_frames = raw_panda_data.shape[0]
        raw_beats_data = raw_panda_data['Beat_0'].values
        raw_beat_idxs = np.where(raw_beats_data > 0.99)
        raw_beat_idxs = raw_beat_idxs[0]
        raw_first_beat = raw_beat_idxs[0]

    # madmon处理
    #chroma = get_chroma(audio_file_name)

    # 用librosa/madmon处理，获得mfcc、chroma、spectral_flux、beat_activation、beat_onehot
    FPS = 30
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    data, _ = librosa.load(audio_file_name, sr=SR)
    envelope = librosa.onset.onset_strength(y=data, sr=SR)
    mfcc = librosa.feature.mfcc(y=data, sr=SR, n_mfcc=20).T
    chroma = librosa.feature.chroma_cens(y=data, sr=SR, hop_length=HOP_LENGTH, n_chroma=6).T
    #tempo, beat_idxs = librosa.beat.beat_track(onset_envelope=envelope, sr=SR, hop_length=HOP_LENGTH, start_bpm=120.0, tightness=100)

    spectral_flux = get_spectral_flux(audio_file_name)
    beat_activation, beat_idxs = get_beat_activation(audio_file_name)
    beat_activation = beat_activation[:,0] # (frames,2)取首列

    beat_onehot = np.zeros_like(envelope, dtype=np.float32)
    beat_onehot[beat_idxs] = 1.0

    # 与原数据集对齐
    frames = min(mfcc.shape[0], chroma.shape[0], spectral_flux.shape[0], beat_activation.shape[0], beat_onehot.shape[0])
    first_beat = beat_idxs[0]

    def _get_start_index():
        nonlocal raw_frames, raw_first_beat, frames, first_beat, beat_idxs, audio_name
        frames_diff = frames - raw_frames
        first_beat_diff = first_beat - raw_first_beat
        frames_per_beat = (beat_idxs[-1] - beat_idxs[0]) / (len(beat_idxs) - 1)

        if first_beat >= raw_first_beat:
            # 训练数据是原始数据的一个裁剪，理应如此
            start_index = first_beat - raw_first_beat
        else:
            # raw_first_beat > first_beat
            raw_first_beat_copy = raw_first_beat
            while raw_first_beat_copy > first_beat:
                raw_first_beat_copy -= frames_per_beat
            if raw_first_beat_copy >= 0:
                start_index = first_beat - raw_first_beat_copy
            else:
                start_index = first_beat - raw_first_beat_copy

        max_start_index = frames_diff  # start_index不该超过这个值
        while start_index > max_start_index:
            start_index -= frames_per_beat
        if start_index < 0:
            if start_index > -3:
                start_index = 0
            else:
                start_index += frames_per_beat

        print(f"raw_frames={raw_frames}, raw_first_beat={raw_first_beat}, frames={frames}, first_beat={first_beat}, frames_per_beat={frames_per_beat} ==> start_index={start_index}")
        return int(round(start_index))

    if align_to_raw_data:
        start_index = _get_start_index()
        assert start_index >= 0, f"{audio_name}, raw_first_beat={raw_first_beat}, my_first_beat={first_beat}"
        end_index = start_index + raw_frames
        #assert(frames >= end_index), f"{audio_name}, raw_frames={raw_frames}, my_frames={frames}-{start_index}"
    else:
        start_index = 0
        end_index = frames

    if frames < end_index:
        end_index = frames

    mfcc = mfcc[start_index:end_index,...]
    chroma = chroma[start_index:end_index,...]
    spectral_flux = spectral_flux[start_index:end_index, ...]
    beat_activation = beat_activation[start_index:end_index, ...]
    beat_onehot = beat_onehot[start_index:end_index, ...]
    #assert(beat_onehot[raw_first_beat] == 1)

    # 组织数据
    final_frames = mfcc.shape[0]
    channels = audio_feature = np.concatenate([
        mfcc, chroma, spectral_flux[:, None], beat_activation[:, None], beat_onehot[:, None]
    ], axis=-1)  # 合并后的numpy数组，shape=(frame, 29)
    time_list = [i/FPS for i in range(final_frames)]  # 以秒计算的各帧时间，shape=(frame,)
    time_index = pd.to_timedelta(time_list, unit='s')  # 转成panda的数据
    column_names = np.loadtxt('./data/motorica_dance/audio29_features.txt', dtype=str).tolist() # 字符串的list，shape=(29,)
    panda_data = pd.DataFrame(data=channels, index=time_index, columns=column_names)

    # 保存
    with open(save_name_1, 'wb') as pkl_f1:
        pkl.dump(panda_data, pkl_f1)
    if process_mirror:
        with open(save_name_2, 'wb') as pkl_f2:
            pkl.dump(panda_data, pkl_f2)

    # 测试
    do_test = False
    if do_test:
        with open(save_name_1, 'rb') as ff:
            reload_panda_data = pkl.load(ff).astype('float32')
        diff = reload_panda_data - panda_data

    return


def process_raw_dataset():
    save_path = './data/my_train_data/'

    #test
    #process_audio('./data/wav/kthjazz_gCH_sFM_cAll_d02_mCH_ch01_charlestonchaserswabashblues_004.wav', save_path, align_to_raw_data=True)

    all_files = []
    train_files = []
    test_files = []
    music_files = glob.glob('./data/wav/*.wav')
    music_files.sort()
    for file_name in tqdm.tqdm(music_files):
        print("Process %s" % file_name)
        process_audio(file_name, save_path, all_files, align_to_raw_data=True)
        #break

    raw_train_files = np.loadtxt('./data/motorica_dance/dance_train_files.txt', dtype=str).tolist()
    raw_test_files = np.loadtxt('./data/motorica_dance/dance_test_files.txt', dtype=str).tolist()
    for raw_file in raw_train_files:
        if raw_file in all_files:
            train_files.append(raw_file)
    for raw_file in raw_test_files:
        if raw_file in all_files:
            test_files.append(raw_file)

    save_train_name = os.path.join(save_path, 'dance_train_files.txt')
    with open(save_train_name, 'w') as f:
        for line in train_files:
            f.write(line + '\n')
    shutil.copy(save_train_name, os.path.join(save_path, 'dance_train_files_kth.txt'))

    save_test_name = os.path.join(save_path, 'dance_test_files.txt')
    with open(save_test_name, 'w') as f:
        for line in test_files:
            f.write(line + '\n')
    shutil.copy(save_test_name, os.path.join(save_path, 'dance_test_files_kth.txt'))

    raw_motion_files = glob.glob('./data/motorica_dance/*.expmap_30fps.pkl')
    for motion_file_name in raw_motion_files:
        base_name = os.path.basename(motion_file_name)
        audio_name = base_name.split('.')[0]
        if audio_name in all_files:
            dest_file_name = os.path.join(save_path, base_name)
            shutil.copy(motion_file_name, dest_file_name)

    other_files = ['data_pipe.expmap_30fps.sav',
                   'audio18_features.txt',
                   'audio29_features.txt',
                   'ch0_spec_beatact_features.txt',
                   'dance_styles.txt',
                   'dance_styles_kth.txt',
                   'gen_files.txt',
                   'pose_features.expmap.txt'
                   ]
    for other_file in other_files:
        src_file_name = os.path.join('./data/motorica_dance', other_file)
        dest_file_name = os.path.join(save_path, other_file)
        shutil.copy(src_file_name, dest_file_name)

    print("DONE !")


def process_new_dataset():
    save_path = './data/my_wav/'
    all_files = []
    music_files = glob.glob('./data/my_wav/*.wav')
    music_files.sort()
    for file_name in tqdm.tqdm(music_files):
        print("Process %s" % file_name)
        process_audio(file_name, save_path, all_files, align_to_raw_data=False)


if __name__ == "__main__":
    #process_raw_dataset()
    process_new_dataset()
