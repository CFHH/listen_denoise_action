import os
import numpy as np
import torch
import pickle as pkl
from pytorch_lightning import seed_everything
from models.LightningModel import LitLDA
from gen_music_pkl import process_audio
from synthesize import nans2zeros, sample_mixmodels
from utils.motion_dataset import styles2onehot
from helper.smpl_bvh_loader import load_bvh_motion
from helper.bvh2ue import send_motion
import json
import math


g_model = None
g_http_path = './http_data'
g_eval_path = './http_data/eval'
g_upload_path = './http_data/upload'
g_audio_feats_columns = None
g_all_styles = None
use_gpu = True
if use_gpu:
    gpu = 'cuda:0'
else:
    gpu = 'cpu'
g_gen_seconds = 30


def cache_model():
    global g_model, g_eval_path, g_upload_path, g_audio_feats_columns, g_all_styles, gpu, g_gen_seconds

    os.makedirs(g_eval_path, exist_ok=True)
    os.makedirs(g_upload_path, exist_ok=True)

    checkpoint = './pretrained_models/frankenstein_v2/checkpoints/epoch=8-step=1307070.ckpt'
    print(f'LOADING MODEL {checkpoint} ......')
    g_model = LitLDA.load_from_checkpoint(checkpoint, dataset_root=g_eval_path)
    device = torch.device(gpu)
    g_model.to(device)
    g_model.eval()

    input_feats_file = os.path.join(g_eval_path, g_model.hparams.Data["input_feats_file"])
    g_audio_feats_columns = np.loadtxt(input_feats_file, dtype=str)

    styles_file = os.path.join(g_eval_path, g_model.hparams.Data["styles_file"])
    print(f'LOADING Style File {styles_file} ......')
    g_all_styles = np.loadtxt(styles_file, dtype=str)

    return g_model


def generate_dance_for_music(file_name, style_token='gFF', start_seconds=0):
    """
    :param file_name: 是上传目录里的文件
    :param style_token: gOK表示流行，gFF表示AI生成的音乐
    :param start_seconds:
    :return:
    """
    valid_styles = ['gOK', 'gFF']
    print(f'generate_dance_for_music(), file_name={file_name}, style={style_token}, start_seconds={start_seconds}')
    global g_model, g_eval_path, g_upload_path, g_audio_feats_columns, g_all_styles, gpu, g_gen_seconds
    if g_model is None:
        cache_model()

    error = None
    json_str = ''

    result_data = {'error':'0'}
    full_filename = os.path.join(g_upload_path, file_name)
    if not os.path.isfile(full_filename):
        error = f'{file_name} not uploaded'
        print(error)
        return error, json_str

    base_name = os.path.basename(full_filename)
    parts = base_name.split('.')
    audio_name = parts[0]
    ext = parts[1].lower()
    if ext not in['wav', 'mp3']:
        error = f'ext(.{ext}) not supported'
        print(error)
        return error, json_str

    #base_name.replace('_', '')  # 这是为了最初的代码

    if style_token not in valid_styles:
        error = f'style(.{style_token}) not supported'
        print(error)
        return error, json_str

    print('processing audio data ...')
    r = process_audio(full_filename, g_eval_path, all_files=None, align_to_raw_data=False, process_mirror=False, genra='', exists_ok=False)
    duration = r['duration']
    pkl_data = r['data']

    if duration < start_seconds:
        error = f'duration({duration}) < start_seconds({start_seconds})'
        print(error)
        return error, json_str

    fps = 30

    start_frame = start_seconds * fps
    clip_seconds = min(duration - start_seconds, g_gen_seconds)
    nframes = min(int(clip_seconds * fps), pkl_data.shape[0] - start_frame)
    end_frame = start_frame + nframes

    ctrl = pkl_data[start_frame : end_frame]
    ctrl = ctrl[g_audio_feats_columns]
    ctrl = nans2zeros(torch.from_numpy(ctrl.values).float().unsqueeze(0))
    nbatch = ctrl.size(0)
    nframes = ctrl.size(1)
    styles_onehot = torch.from_numpy(styles2onehot(g_all_styles, style_token)).float()
    style_cond = styles_onehot.repeat(nbatch, nframes, 1) # l_cond
    audio_cond = g_model.standardizeInput(ctrl)           # g_cond

    # do_synthesize(models, l_conds, g_conds, out_file_name, postfix, trim, dest_dir, guidance_factors, gpu, render_video, outfile)
    print('generating dance ...')
    device = torch.device(gpu)
    batch = audio_cond.to(device), style_cond.to(device), None
    models = [g_model]
    batches = [batch]
    guidance_factors = []
    bvh_filename = audio_name  # style_token
    #seed_everything(150)  # 随机种子
    with torch.no_grad():
        clips = sample_mixmodels(models, batches, guidance_factors)
        g_model.log_results(clips, bvh_filename, "", logdir=g_eval_path, render_video=False)

    print('translating for UE ...')
    bvh_filename = os.path.join(g_eval_path, f'{bvh_filename}.bvh')
    ue_data = bvh2uedata(bvh_filename)
    json_data = {
        'audio': file_name,
        'fps': 30,
        'motion': ue_data,
    }
    #json_str = json.dumps(json_data)
    print('done.')
    return error, json_data


def generate_dance_for_music_full(file_name, style_token='gFF'):
    valid_styles = ['gOK', 'gFF']
    print(f'generate_dance_for_music_full(), file_name={file_name}, style={style_token}')
    global g_model, g_eval_path, g_upload_path, g_audio_feats_columns, g_all_styles, gpu, g_gen_seconds
    if g_model is None:
        cache_model()

    error = None
    json_str = ''

    result_data = {'error':'0'}
    full_filename = os.path.join(g_upload_path, file_name)
    if not os.path.isfile(full_filename):
        error = f'{file_name} not uploaded'
        print(error)
        return error, json_str

    base_name = os.path.basename(full_filename)
    parts = base_name.split('.')
    audio_name = parts[0]
    ext = parts[1].lower()
    if ext not in['wav', 'mp3']:
        error = f'ext(.{ext}) not supported'
        print(error)
        return error, json_str

    if style_token not in valid_styles:
        error = f'style(.{style_token}) not supported'
        print(error)
        return error, json_str

    print('processing audio data ...')
    r = process_audio(full_filename, g_eval_path, all_files=None, align_to_raw_data=False, process_mirror=False, genra='', exists_ok=False)
    duration = r['duration']
    pkl_data = r['data']

    fps = 30
    #seed_everything(150)  # 随机种子

    gen_cnt = int(math.ceil(duration / g_gen_seconds))
    start_frame = 0
    ue_data = []
    for i in range(gen_cnt):
        if i == gen_cnt - 1:
            clip_seconds = duration - g_gen_seconds * (gen_cnt - 1)
        else:
            clip_seconds = g_gen_seconds
        nframes = min(int(clip_seconds * fps), pkl_data.shape[0] - start_frame)

        ctrl = pkl_data[start_frame : start_frame + nframes]
        ctrl = ctrl[g_audio_feats_columns]
        ctrl = nans2zeros(torch.from_numpy(ctrl.values).float().unsqueeze(0))
        nbatch = ctrl.size(0)
        nframes = ctrl.size(1)
        styles_onehot = torch.from_numpy(styles2onehot(g_all_styles, style_token)).float()
        style_cond = styles_onehot.repeat(nbatch, nframes, 1)  # l_cond
        audio_cond = g_model.standardizeInput(ctrl)  # g_cond

        # do_synthesize(models, l_conds, g_conds, out_file_name, postfix, trim, dest_dir, guidance_factors, gpu, render_video, outfile)
        print(f'generating dance ({i+1}/{gen_cnt}) ......')
        device = torch.device(gpu)
        batch = audio_cond.to(device), style_cond.to(device), None
        models = [g_model]
        batches = [batch]
        guidance_factors = []
        bvh_filename = audio_name  # style_token
        with torch.no_grad():
            clips = sample_mixmodels(models, batches, guidance_factors)
            g_model.log_results(clips, bvh_filename, "", logdir=g_eval_path, render_video=False)

        print('translating for UE ...')
        bvh_filename = os.path.join(g_eval_path, f'{bvh_filename}.bvh')
        clip_ue_data = bvh2uedata(bvh_filename)
        ue_data = ue_data + clip_ue_data

        # next
        start_frame += nframes

    json_data = {
        'audio': file_name,
        'fps': 30,
        'motion': ue_data,
    }
    #json_str = json.dumps(json_data)
    print('done.')
    return error, json_data


def bvh2uedata(bvh_filename):
    root_position, rotation, frametime, name, parent, offsets = load_bvh_motion(bvh_filename, True)
    root_position *= 100
    root_position -= [-0.0363, 91.213097, 4.3399]
    msg_arr = []
    for idx in range(len(rotation)):
        msg = send_motion(rotation[idx], root_position[idx])
        msg_arr.append(msg)
    #json_str = json.dumps(msg_arr)
    #return json_str
    return msg_arr


def test():
    # 给UE测试用
    import glob, tqdm
    data_path = './results/generated/gen_frankenstein_v2/bvh'
    bvh_files = glob.glob(os.path.join(data_path, '*.bvh'))
    bvh_files.sort()
    for bvh_file in tqdm.tqdm(bvh_files):
        motion_name = os.path.basename(bvh_file)
        motion_name = motion_name.split('.')[0]
        motion_name = motion_name[0:-4]
        json_data = bvh2uedata(bvh_file)
        #json_str = json.dumps(json_data)
        save_file = os.path.join(data_path, f'{motion_name}.json')
        with open(save_file, "w") as json_file:
            json.dump(json_data, json_file, indent=4)
    return


if __name__ == "__main__":
    test()
    cache_model()
    r1 = generate_dance_for_music('嘻哈风格wav.wav')
    r2 = generate_dance_for_music('嘻哈风格mp3.mp3')
    a = 1
