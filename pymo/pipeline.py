from pymo.preprocessing import JointSelector, RootTransformer, MocapParameterizer, ConstantsRemover, FeatureCounter, Numpyfier
from sklearn.pipeline import Pipeline


def get_pipeline(is_dance_skeleton):
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
    t6 = Numpyfier()  # 从这个的self.org_mocap_.values.columns获得列表，存入pose_features.expmap.txt

    pipeline = [t1, t2, t3, t4, t5, t6]
    #pipe = Pipeline([('joint', t1), ('root', t2), ('mocap', t3), ('constant', t4), ('counter', t5), ('numpy', t6)])
    return pipeline


def transform(pipe, bvh_datas):
    datas = bvh_datas
    for tr in pipe:
        datas = tr.fit_transform(datas)
    return datas

def transform2pkl(pipe, bvh_datas):
    datas = bvh_datas
    for i in range(4):
        tr = pipe[i]
        datas = tr.fit_transform(datas)
    return datas

def inverse_transform(pipe, pred_clips):
    datas = pred_clips
    for tr in reversed(pipe):
        datas = tr.inverse_transform(datas)
    return datas
