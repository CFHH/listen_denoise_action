import os
import numpy as np
from pytorch_lightning import seed_everything
from models.LightningModel import LitLDA
from synthesize import do_synthesize, get_cond


if __name__ == "__main__":
    checkpoint = '../pretrained_models/dance_LDA.ckpt'
    dest_dir = '../results/generated/dance_LDA'
    os.makedirs(dest_dir, exist_ok=True)

    data_dir = '../data/motorica_dance'
    wav_dir = '../data/motorica_dance/wav'
    basenames = np.loadtxt('../data/motorica_dance/gen_files.txt', dtype=str).tolist()

    start = 0
    seed = 150
    fps = 30
    trim_s = 0
    length_s = 10  # 生成多少秒
    trim = trim_s * fps
    length = length_s * fps
    fixed_seed = False
    #gpu = 'cuda:0'
    gpu = 'cpu'
    render_video = True

    for wavfile in basenames:
        start = 0
        style = wavfile.split('_')[1]
        for postfix in range(12):
            input_file = f'{wavfile}.audio29_{fps}fps.pkl'
            output_file = f'{wavfile[0:-3]}_{postfix}_{style}'

            checkpoints = [checkpoint]
            data_dirs = [data_dir]
            input_files = [input_file]
            style_tokens = [style]
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

            do_synthesize(models, l_conds, g_conds, out_file_name, postfix, trim, dest_dir, guidance_factors, gpu,
                          render_video, outfile)
