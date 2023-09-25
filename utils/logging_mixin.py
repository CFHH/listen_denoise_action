# Copyright 2023 Motorica AB, Inc. All Rights Reserved.

import io
import json
from pathlib import Path
import joblib as jl
import numpy as np
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.pipeline import Pipeline
from pymo.writers import *
from pymo.data import Joint, MocapData
from pymo.preprocessing import *
from pymo.viz_tools import *
from scipy import interpolate
from pymo.parsers import BVHParser


parameterizer = None
mocap_data_sample = None
train_columns = None
ignored_columns = None
full_columns = None

def gesture_feats_to_bvh(pred_clips, dataset_root, from_train=False):
    """
    :param pred_clips:
    :param dataset_root:
    :param from_train: LitLDA.synthesize()，加了3帧
    :return:
    """
    global parameterizer, mocap_data_sample, train_columns, ignored_columns, full_columns
    if parameterizer is None:
        # parameterizer, mocap_data_sample
        skeleton_bvh = os.path.join(dataset_root, 'skeleton.bvh')
        bvh_parser = BVHParser()
        bvh_data = bvh_parser.parse(skeleton_bvh)
        bvh_datas = [bvh_data]
        parameterizer = MocapParameterizer('expmap')
        expmap_datas = parameterizer.fit_transform(bvh_datas)
        mocap_data_sample = expmap_datas[0]

        # train_columns, ignored_columns, full_columns
        bone_feature_filename = os.path.join(dataset_root, 'pose_features.expmap.txt')
        train_columns = np.loadtxt(bone_feature_filename, dtype=str).tolist()

        ignored_joints = ['RightToeBase', 'RightForeFoot', 'LeftToeBase', 'LeftForeFoot', 'pCube4']
        hands = ['LeftHand', 'RightHand']
        fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        indexs = [1, 2, 3]
        for hand in hands:
            for finger in fingers:
                for index in indexs:
                    name = f'{hand}{finger}{index}'
                    ignored_joints.append(name)
        ignored_columns = []
        for joint in ignored_joints:
            ignored_columns.append(joint + '_alpha')
            ignored_columns.append(joint + '_beta')
            ignored_columns.append(joint + '_gamma')

        full_columns = train_columns + ignored_columns

    mocap_datas = []
    for clip in pred_clips:
        if from_train:
            clip = clip[:, :-3]  # TODO ZZW
        frames = clip.shape[0]
        zeros = np.zeros((frames, len(ignored_columns)))
        channels = np.concatenate([clip, zeros], axis=-1)

        fps = 30
        time_list = [i / fps for i in range(frames)]  # 以秒计算的各帧时间，shape=(frame,)
        time_index = pd.to_timedelta(time_list, unit='s')  # 转成panda的数据
        panda_data = pd.DataFrame(data=channels, index=time_index, columns=full_columns)  # 就是pkl

        new_data = MocapData()
        new_data.skeleton = copy.deepcopy(mocap_data_sample.skeleton)
        new_data.channel_names = copy.deepcopy(mocap_data_sample.channel_names)
        new_data.root_name = copy.deepcopy(mocap_data_sample.root_name)
        new_data.values = panda_data
        new_data.framerate = 1 / fps
        # new_data.take_name = ''

        mocap_datas.append(new_data)

    my_bvh_datas = parameterizer.inverse_transform(mocap_datas)
    return my_bvh_datas


class LoggingMixin:

    def log_results(self, pred_clips, file_name, log_prefix, logdir=None, render_video=True):

        if logdir is None:
            logdir = f"{self.logger.save_dir}/{self.logger.name}/version_{self.logger.version}/"

        if len(log_prefix.strip())>0:
            file_name = file_name + "_" + log_prefix
            
        bvh_data = self.feats_to_bvh(pred_clips)
        nclips = len(bvh_data)
        framerate = np.rint(1/bvh_data[0].framerate)
        
        if self.hparams.Validation["max_render_clips"]:
            nclips = min(nclips, self.hparams.Validation["max_render_clips"])
        
        self.write_bvh(bvh_data[:nclips], log_dir=logdir, name_prefix=file_name)
        
        if render_video:
            pos_data = self.bvh_to_pos(bvh_data)
            
        if render_video:
            self.render_video(pos_data[:nclips], log_dir=logdir, name_prefix=file_name)
                    
        
    def feats_to_bvh(self, pred_clips):
        if self.hparams.Data["datapipe_filename"] == 'inverse_transform_gesture':
            return gesture_feats_to_bvh(pred_clips, self.hparams.dataset_root, from_train=True)
        #import pdb;pdb.set_trace()
        data_pipeline = jl.load(Path(self.hparams.dataset_root) / self.hparams.Data["datapipe_filename"])
        n_feats = data_pipeline["cnt"].n_features
        data_pipeline["root"].separate_root=False

        print('inverse_transform...')
        bvh_data=data_pipeline.inverse_transform(pred_clips[:,:,:n_feats])
        return bvh_data


    def write_bvh(self, bvh_data, log_dir="", name_prefix=""):
        writer = BVHWriter()
        nclips = len(bvh_data)
        for i in range(nclips):        
            if nclips>1:
                fname = f"{log_dir}/{name_prefix}_{str(i).zfill(3)}.bvh"
            else:
                fname = f"{log_dir}/{name_prefix}.bvh"
            print('writing:' + fname)
            with open(fname,'w') as f:
                writer.write(bvh_data[i], f)
        
    def bvh_to_pos(self, bvh_data):        
        # convert to joint positions
        return MocapParameterizer('position').fit_transform(bvh_data)
                
    def render_video(self, pos_data, log_dir="", name_prefix=""):
        # write bvh and skeleton motion
        nclips = len(pos_data)
        for i in range(nclips):        
            if nclips>1:
                fname = f"{log_dir}/{name_prefix}_{str(i).zfill(3)}"
            else:
                fname = f"{log_dir}/{name_prefix}"
            print('writing:' + fname + ".mp4")
            render_mp4(pos_data[i], fname + ".mp4", axis_scale=200)
        
            
    def log_jerk(self, x, log_prefix):

        deriv = x[:, 1:] - x[:, :-1]
        acc = deriv[:, 1:] - deriv[:, :-1]
        jerk = acc[:, 1:] - acc[:, :-1]
        self.log(f'{log_prefix}_jerk', torch.mean(torch.abs(jerk)), sync_dist=True)
        
