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
from helper.bvh2ue import bvh2ueactor, send_motion
import json


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
g_gen_seconds = 10


def cache_model():
    global g_model, g_eval_path, g_upload_path, g_audio_feats_columns, g_all_styles, gpu, g_gen_seconds

    os.makedirs(g_eval_path, exist_ok=True)
    os.makedirs(g_upload_path, exist_ok=True)

    checkpoint = './pretrained_models/smpl_dance_chroma6_layers15/checkpoints/epoch=10-step=1315930.ckpt'
    g_model = LitLDA.load_from_checkpoint(checkpoint, dataset_root=g_eval_path)
    device = torch.device(gpu)
    g_model.to(device)
    g_model.eval()

    input_feats_file = os.path.join(g_eval_path, g_model.hparams.Data["input_feats_file"])
    g_audio_feats_columns = np.loadtxt(input_feats_file, dtype=str)

    styles_file = os.path.join(g_eval_path, g_model.hparams.Data["styles_file"])
    g_all_styles = np.loadtxt(styles_file, dtype=str)

    return g_model


def generate_dance_for_music(file_name, style_token='gOK'):
    """
    :param file_name: 是上传目录里的文件
    :param style_token:
    :return:
    """
    print(f'generate_dance_for_music(), file_name=f{file_name}')
    global g_model, g_eval_path, g_upload_path, g_audio_feats_columns, g_all_styles, gpu, g_gen_seconds
    if g_model is None:
        cache_model()

    error = 'ok'
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

    print('processing audio data ...')
    r = process_audio(full_filename, g_eval_path, all_files=None, align_to_raw_data=False, process_mirror=False, genra='', exists_ok=False)
    duration = r['duration']
    pkl_data = r['data']

    fps = 30
    clip_seconds = min(duration, g_gen_seconds)
    nframes = min(int(clip_seconds * fps), pkl_data.shape[0])

    ctrl = pkl_data[0:nframes]
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
    json_str = bvh2jsonstr(bvh_filename)
    print('done.')
    return error, json_str


def bvh2jsonstr(bvh_filename):
    root_position, rotation, frametime, name, parent, offsets = load_bvh_motion(bvh_filename, True)
    root_position -= offsets[0]
    root_position *= 100
    msg_arr = []
    for idx in range(len(rotation)):
        msg = send_motion(rotation[idx], root_position[idx])
        msg_arr.append(msg)
    json_str = json.dumps(msg_arr)
    return json_str


if __name__ == "__main__":
    cache_model()
    r1 = generate_dance_for_music('嘻哈风格wav.wav')
    r2 = generate_dance_for_music('嘻哈风格mp3.mp3')
    a = 1

