import os
import numpy as np
from pytorch_lightning import seed_everything
from models.LightningModel import LitLDA
import scipy.io.wavfile as wav
from synthesize import do_synthesize, get_cond


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


if __name__ == "__main__":
    use_raw_data = False
    if use_raw_data:
        checkpoint = '../pretrained_models/dance_LDA.ckpt'
        data_dir = '../data/motorica_dance'
        wav_dir = '../data/motorica_dance/wav'
        basenames = np.loadtxt('../data/motorica_dance/gen_files.txt', dtype=str).tolist()
    else:
        checkpoint = '../pretrained_models/my_results/checkpoints/epoch=9-step=403080.ckpt'
        data_dir = '../data/my_wav'
        wav_dir = '../data/my_wav'
        basenames = np.loadtxt('../data/my_wav/my_gen_files.txt', dtype=str).tolist()

    dest_dir = '../results/generated/dance_LDA'
    os.makedirs(dest_dir, exist_ok=True)

    start = 0
    seed = 150
    fps = 30
    trim_s = 0  # 这个就不能不是零
    length_s = 10  # 每一段生成生成多少秒
    trim = trim_s * fps
    length = length_s * fps
    fixed_seed = False
    #gpu = 'cuda:0'
    gpu = 'cpu'
    render_video = True

    for wavfile in basenames:
        print( f"process {wavfile} ......")
        start = 0
        style_token = wavfile.split('_')[1]
        for postfix in range(12):  # 生成几段，每段长length_s秒
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
