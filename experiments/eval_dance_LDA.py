import os
import numpy as np
from pytorch_lightning import seed_everything
from models.LightningModel import LitLDA
import scipy.io.wavfile as wav
import librosa
from synthesize import do_synthesize, get_cond
import tqdm
import glob
from pymo.parsers import BVHParser
from utils.logging_mixin import LoggingMixin


def cutwav(wav_dir, wavfile, starttime, endtime, suffix, dest_dir):
    filename = os.path.join(wav_dir, wavfile[0:-3] + '.wav')
    print(f'Cutting AUDIO {filename} from: {starttime} to {endtime}')
    basename = os.path.splitext(os.path.basename(filename))[0]
    out_wav_file = os.path.join(dest_dir, f'{basename}_{suffix}.wav')
    fs, X = wav.read(filename)
    start_idx = int(np.round(starttime * fs))
    end_idx = int(np.round(endtime * fs))
    if end_idx < X.shape[0]:
        wav.write(out_wav_file, fs, X[start_idx:end_idx])
    else:
        print("EOF REACHED")
    return out_wav_file


def exec_cmd(cmd):
    # python 执行命令获取输出
    print(cmd)
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text


def files_to_list(filename):
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()
    files = [f.rstrip() for f in files]
    return files


def get_music_duration(file_name):
    FPS = 60
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    data, _ = librosa.load(file_name, sr=SR)
    seconds = data.shape[0] / SR
    return seconds


def eval(clip_seconds=20, use_gpu=True, render_video=True):
    model_type = 'smpl_dance'
    if model_type == 'raw':
        # 原作者提供的，已经删了
        checkpoint = '../pretrained_models/dance_LDA.ckpt'
        data_dir = '../data/motorica_dance'
        wav_dir = '../data/wav'
        basenames = np.loadtxt('../data/motorica_dance/gen_files.txt', dtype=str).tolist()
        dest_dir = '../results/generated/dance_LDA_raw'
    elif model_type == 'gesture':
        # 自己训练的手势
        checkpoint = '../pretrained_models/my_gesture_data/checkpoints/epoch=9-step=189440.ckpt'
        data_dir = '../data/my_speech_for_eval'
        wav_dir = '../data/my_speech_for_eval'
        basenames = np.loadtxt('../data/my_speech_for_eval/my_gen_files.txt', dtype=str).tolist()
        dest_dir = '../results/generated/gesture_LDA'
    elif model_type == 'smpl_dance':
        checkpoint = '../pretrained_models/smpl_dance/checkpoints/epoch=9-step=1196300.ckpt'
        data_dir = '../data/eval_for_smpl_dance'
        wav_dir = '../data/eval_for_smpl_dance'
        #basenames = np.loadtxt('../data/eval_for_smpl_dance/gen_files.txt', dtype=str).tolist()
        basenames = files_to_list('../data/eval_for_smpl_dance/gen_files.txt')
        dest_dir = '../results/generated/smpl_dance'
    else:
        # 自己训练的舞蹈
        checkpoint = '../pretrained_models/my_train_data/checkpoints/epoch=9-step=403080.ckpt'
        data_dir = '../data/my_wav_for_eval'
        wav_dir = '../data/my_wav_for_eval'
        basenames = np.loadtxt('../data/my_wav_for_eval/my_gen_files.txt', dtype=str).tolist()
        dest_dir = '../results/generated/dance_LDA'
    os.makedirs(dest_dir, exist_ok=True)

    start = 0
    seed = 150
    fps = 30
    trim_s = 0  # 这个就不能不是零
    length_s = clip_seconds  # TODO 每一段生成生成多少秒
    trim = trim_s * fps
    #length = length_s * fps
    fixed_seed = False
    if use_gpu:
        gpu = 'cuda:0'
    else:
        gpu = 'cpu'

    for wavfile in basenames:
        print( f"process {wavfile} ......")
        filename = os.path.join(wav_dir, wavfile[0:-3] + '.wav')
        duration = get_music_duration(filename)

        if model_type == 'gesture':
            trim_head = 8
            start = fps * trim_head
            if length_s <= 0:
                gen_cnt = 1
                length_s = int(duration - trim_head)
            else:
                gen_cnt = int((duration - trim_head) / length_s)
        else:
            start = 0
            if length_s <= 0:
                gen_cnt = 1
                length_s = int(duration)
            else:
                gen_cnt = int(duration / length_s)
        length = length_s * fps

        style_token = wavfile.split('_')[1]
        for postfix in range(gen_cnt):  # 生成几段，每段长length_s秒
            input_file = f'{wavfile}.audio29_{fps}fps.pkl'
            output_file = f'{wavfile[0:-3]}_{postfix}_{style_token}'

            checkpoints = [checkpoint]
            data_dirs = [data_dir]
            input_files = [input_file]
            style_tokens = [style_token]
            startframe = start
            endframe = length
            guidance_factors = []
            """
            trim = trim
            seed = seed
            postfix = postfix
            dest_dir = dest_dir
            gpu = gpu
            render_video = render_video
            """
            outfile = output_file

            # 以下抄自synthesize.py
            out_file_name = os.path.basename(input_files[0]).split('.')[0]
            seed_everything(seed)
            models = []
            l_conds = []
            g_conds = []
            for i in range(len(checkpoints)):
                model = LitLDA.load_from_checkpoint(checkpoints[i], dataset_root=data_dirs[i])
                models.append(model)
                if style_tokens is not None:
                    l_cond, style = get_cond(model, data_dirs[i], input_files[i], style_tokens[i], startframe, endframe)
                else:
                    l_cond, style = get_cond(model, data_dirs[i], input_files[i], "", startframe, endframe)
                l_conds.append(l_cond)
                g_conds.append(style)

            do_synthesize(models, l_conds, g_conds, out_file_name, postfix, trim, dest_dir, guidance_factors, gpu, render_video, outfile)

            temp_wav_file = cutwav(wav_dir, wavfile, (start + trim) / fps, (start + length - trim) / fps, postfix, dest_dir)
            if render_video:
                temp_video_file = os.path.join(dest_dir, output_file + '.mp4')
                final_video_file = os.path.join(dest_dir, output_file + '_audio.mp4')
                cmd_line = f'ffmpeg -y -i {temp_video_file} -i {temp_wav_file} {final_video_file}'
                exec_cmd(cmd_line)
                # os.remove(temp_video_file)  # 先不删除了

            # 下一个postfix
            if not fixed_seed:
                seed += 1
            start = start + length


def bvh2mp4():
    data_path = '../results/generated/from_ubuntu'
    wav_files = glob.glob(os.path.join(data_path, '*.wav'))
    wav_files.sort()

    for wav_file in tqdm.tqdm(wav_files):
        base_name = os.path.basename(wav_file)
        base_name = base_name.split('.')[0]
        style = '_gOK'
        name_prefix = f'{base_name}{style}'
        bvh_file = os.path.join(data_path, f'{name_prefix}.bvh')

        bvh_parser = BVHParser()
        bvh_data = bvh_parser.parse(bvh_file)
        bvh_datas = [bvh_data]

        obj = LoggingMixin()
        pos_datas = obj.bvh_to_pos(bvh_datas)
        obj.render_video(pos_datas, log_dir=data_path, name_prefix=name_prefix)

        video_file_without_music = os.path.join(data_path, name_prefix + '.mp4')
        final_video_file = os.path.join(data_path, name_prefix + '_audio.mp4')
        cmd_line = f'ffmpeg -y -i {video_file_without_music} -i {wav_file} {final_video_file}'
        exec_cmd(cmd_line)
    return


if __name__ == "__main__":
    """
    生成长度与显存需求：
    30s  3963M
    45s  7153M
    60s 10197M
    75s 12517M
    90s 21653M
    """
    #eval(clip_seconds=60, use_gpu=True, render_video=False)  # 这个在gpu机器上执行，一次性生成2分钟需要30G显存
    bvh2mp4()
