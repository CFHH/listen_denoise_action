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
from helper.bvh2ue import send_motion, send_motion_genea, SMPL_UE_ROOT_OFFSET
import json
import math


g_http_path = './http_data'
use_gpu = True
if use_gpu:
    gpu = 'cuda:0'
else:
    gpu = 'cpu'


g_models = []

g_dance_model_config = {
    'name' : 'dance',
    'checkpoint' : './pretrained_models/frankenstein_v2/checkpoints/epoch=8-step=1307070.ckpt',
    'model' : None,
    'upload_path' : './http_data/upload',
    'eval_path' : './http_data/eval_dance',
    'pkl_path' : './http_data/pkl',
    'bvh_path' : './http_data/bvh',
    'audio_feats_columns' : None,
    'all_styles' : None,
}
g_models.append(g_dance_model_config)

g_gesture_model_config = {
    'name' : 'gesture',
    'checkpoint' : './pretrained_models/gesture_v2_nomirror/checkpoints/epoch=9-step=189440.ckpt',
    'model' : None,
    'upload_path' : './http_data/upload',
    'eval_path' : './http_data/eval_gesture',
    'pkl_path' : './http_data/pkl',
    'bvh_path' : './http_data/bvh',
    'audio_feats_columns' : None,
    'all_styles' : None,
}
g_models.append(g_gesture_model_config)


def cache_all_models():
    global g_models
    for model in g_models:
        cache_model(model)
    return


def cache_model(model_config):
    global gpu
    model_name = model_config['name']
    print(f'CACHING Model {model_name} ......')

    os.makedirs(model_config['upload_path'], exist_ok=True)
    os.makedirs(model_config['eval_path'], exist_ok=True)

    checkpoint = model_config['checkpoint']
    print(f'LOADING checkpoint from {checkpoint} ......')
    model = LitLDA.load_from_checkpoint(checkpoint, dataset_root=model_config['eval_path'])
    model_config['model'] = model
    device = torch.device(gpu)
    model.to(device)  # 是否需要每次调用模型时再放入gpu
    model.eval()

    input_feats_file = os.path.join(model_config['eval_path'], model.hparams.Data["input_feats_file"])
    model_config['audio_feats_columns'] = np.loadtxt(input_feats_file, dtype=str)

    styles_file = os.path.join(model_config['eval_path'], model.hparams.Data["styles_file"])
    print(f'LOADING Style File {styles_file} ......')
    model_config['all_styles'] = np.loadtxt(styles_file, dtype=str)
    return


def get_model_config(style_token='gFF'):
    global g_models
    for model_config in g_models:
        styles = model_config['all_styles']
        if style_token in styles:
            return model_config
    return None


def generate_action_for_audio(audio_filename, style_token='gFF', start_seconds=0, gen_seconds=30):
    """
    :param audio_filename: 是上传目录里的文件
    :param style_token: gOK表示流行，gFF表示AI生成的音乐
    :param start_seconds: 从第几秒开始生成
    :return:
    """
    global gpu
    print(f'generate_action_for_audio(), audio={audio_filename}, style={style_token}, start={start_seconds}, seconds={gen_seconds}')
    error = None
    json_str = ''
    fps = 30

    model_config = get_model_config(style_token)
    if model_config is None:
        error = f'style(.{style_token}) not supported'
        print(error)
        return error, json_str
    print('Using model %s' % model_config['name'])

    model = model_config['model']
    if model is None:
        cache_model(model_config)
    else:
        """
        device = torch.device(gpu)
        model.to(device)
        model.eval()
        """
        pass

    full_filename = os.path.join(model_config['upload_path'], audio_filename)
    if not os.path.isfile(full_filename):
        error = f'{audio_filename} not uploaded'
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

    print('processing audio data ...')
    r = process_audio(full_filename, model_config['pkl_path'], all_files=None, align_to_raw_data=False, process_mirror=False, genra='', exists_ok=False)
    duration = r['duration']
    pkl_data = r['data']

    if duration < start_seconds + 5:
        error = f'duration({duration}) < start_seconds({start_seconds}) + 5'
        print(error)
        return error, json_str

    start_frame = start_seconds * fps
    clip_seconds = min(duration - start_seconds, gen_seconds)
    nframes = min(int(clip_seconds * fps), pkl_data.shape[0] - start_frame)
    end_frame = start_frame + nframes

    ctrl = pkl_data[start_frame: end_frame]
    ctrl = ctrl[model_config['audio_feats_columns']]
    ctrl = nans2zeros(torch.from_numpy(ctrl.values).float().unsqueeze(0))
    nbatch = ctrl.size(0)
    nframes = ctrl.size(1)
    styles_onehot = torch.from_numpy(styles2onehot(model_config['all_styles'], style_token)).float()
    style_cond = styles_onehot.repeat(nbatch, nframes, 1)  # l_cond
    audio_cond = model.standardizeInput(ctrl)  # g_cond

    # do_synthesize(models, l_conds, g_conds, out_file_name, postfix, trim, dest_dir, guidance_factors, gpu, render_video, outfile)
    print('generating action ...')
    device = torch.device(gpu)
    batch = audio_cond.to(device), style_cond.to(device), None
    models = [model]
    batches = [batch]
    guidance_factors = []
    bvh_filename = audio_name
    # seed_everything(150)  # 随机种子
    with torch.no_grad():
        clips = sample_mixmodels(models, batches, guidance_factors)
        model.log_results(clips, bvh_filename, "", logdir=model_config['bvh_path'], render_video=False)

    def style_to_skeleton(style):
        if style == 'gSP':
            return 'genea'
        else:
            return 'smpl'

    print('translating for UE ...')
    bvh_filename = os.path.join(model_config['bvh_path'], f'{bvh_filename}.bvh')
    ue_data = bvh2uedata(bvh_filename, skeleton=style_to_skeleton(style_token))
    json_data = {
        'audio': audio_filename,
        'fps':fps,
        'motion': ue_data,
    }
    # json_str = json.dumps(json_data)
    print('done.')
    return error, json_data


def bvh2uedata(bvh_filename, skeleton='smpl'):
    """
    :param bvh_filename:
    :param skeleton: smpl, genea
    :return:
    """
    root_position, rotation, frametime, name, parent, offsets = load_bvh_motion(bvh_filename, True)
    if skeleton == 'smpl':
        root_position -= SMPL_UE_ROOT_OFFSET
        root_position *= 100
        msg_arr = []
        for idx in range(len(rotation)):
            msg = send_motion(rotation[idx], root_position[idx])
            msg_arr.append(msg)
        #json_str = json.dumps(msg_arr)
        #return json_str
    elif skeleton == 'genea':
        from helper.genea_skeleton import GENEA_SIMPLIFIED_INDEX, GENEA_UE_ROOT_OFFSET
        rotation = rotation[:,GENEA_SIMPLIFIED_INDEX,:]
        root_position -= GENEA_UE_ROOT_OFFSET
        root_position *= 100
        msg_arr = []
        for idx in range(len(rotation)):
            msg = send_motion_genea(rotation[idx], root_position[idx])
            msg_arr.append(msg)
    else:
        assert False, f'skeleton={skeleton}'
    return msg_arr


def test_for_ue_smpl():
    # 给UE测试用
    import glob, tqdm
    data_path = './results/generated/gen_frankenstein_v2/bvh'
    bvh_files = glob.glob(os.path.join(data_path, '*.bvh'))
    bvh_files.sort()
    for bvh_file in tqdm.tqdm(bvh_files):
        motion_name = os.path.basename(bvh_file)
        motion_name = motion_name.split('.')[0]
        motion_name = motion_name[0:-4]
        json_data = bvh2uedata(bvh_file, skeleton='smpl')
        #json_str = json.dumps(json_data)
        save_file = os.path.join(data_path, f'{motion_name}.json')
        with open(save_file, "w") as json_file:
            json.dump(json_data, json_file, indent=4)
    return

def test_for_ue_genea():
    # 给UE测试用
    import glob, tqdm
    data_path = './results/test/genea'
    bvh_files = glob.glob(os.path.join(data_path, '*.bvh'))
    bvh_files.sort()
    for bvh_file in tqdm.tqdm(bvh_files):
        motion_name = os.path.basename(bvh_file)
        motion_name = motion_name.split('.')[0]
        motion_name = motion_name[0:-4]
        json_data = bvh2uedata(bvh_file, skeleton='genea')
        #json_str = json.dumps(json_data)
        save_file = os.path.join(data_path, f'{motion_name}.json')
        with open(save_file, "w") as json_file:
            json.dump(json_data, json_file, indent=4)
    return


if __name__ == "__main__":
    dic = {'a':1}
    a = 'a1' in dic.keys()
    test_for_ue_genea()
    cache_all_models()
    #generate_action_for_audio('test_gOK.wav', style_token='gOK', start_seconds=0, gen_seconds=10)
    generate_action_for_audio('test_gFF.wav', style_token='gFF', start_seconds=0, gen_seconds=10)
    #generate_action_for_audio('test_gSP.wav', style_token='gSP', start_seconds=0, gen_seconds=10)
    a = 1
