import os
import random
import argparse
import json
import torch
import torch.utils.data
import sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle as pkl
import pandas as pd
from scipy import interpolate
import joblib as jl
import librosa
import copy
from pymo.data import MocapData
from pymo.parsers import BVHParser
from pymo.writers import BVHWriter
from pymo.preprocessing import JointSelector, RootTransformer, MocapParameterizer, ConstantsRemover, FeatureCounter, Numpyfier
from pymo.Quaternions import Quaternions
from sklearn.pipeline import Pipeline

pipeline = None

def get_pipeline(is_dance_skeleton):
    global pipeline
    if pipeline is not None:
        return pipeline

    dance_joints = ['Spine', 'Spine1', 'Neck', 'Head',
                    'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                    'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
                    'RightUpLeg', 'RightLeg', 'RightFoot',
                    'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'reference']
    gesture_joints = ['Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head',
                      'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                      'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
                      'RightUpLeg', 'RightLeg', 'RightFoot',
                      'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'reference']
    joints = gesture_joints
    if is_dance_skeleton:
        joints = dance_joints

    t1 = JointSelector(joints, include_root=True)
    t2 = RootTransformer('pos_rot_deltas', separate_root=False)
    t3 = MocapParameterizer('expmap', ref_pose=None)
    t4 = ConstantsRemover()
    t5 = FeatureCounter()
    t6 = Numpyfier()

    pipeline = [t1, t2, t3, t4, t5, t6]
    #pipe = Pipeline([('joint', t1), ('root', t2), ('mocap', t3), ('constant', t4), ('counter', t5), ('numpy', t6)])
    return pipeline


def transform(pipe, bvh_datas):
    datas = bvh_datas
    for tr in pipe:
        datas = tr.fit_transform(datas)
    return datas


def inverse_transform(pipe, pred_clips):
    datas = pred_clips
    for tr in reversed(pipe):
        datas = tr.inverse_transform(datas)
    return datas