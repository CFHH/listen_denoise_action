from pymo.preprocessing import JointSelector, RootTransformer, MocapParameterizer, ConstantsRemover, FeatureCounter, Numpyfier
from sklearn.pipeline import Pipeline


def get_pipeline(skeleton_type):
    if skeleton_type == 'default':
        # 作者的数据对应的骨骼
        joints = ['Spine', 'Spine1', 'Neck', 'Head',
                  'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                  'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
                  'RightUpLeg', 'RightLeg', 'RightFoot',
                  'LeftUpLeg', 'LeftLeg', 'LeftFoot',
                  'reference']
        root_joint = 'Hips'  # joints里不含这个
    elif skeleton_type == 'GENEA':
        # Trinity Speech-Gesture I\GENEA_Challenge_2020_data_release 的骨骼
        joints = ['Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head',
                  'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                  'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
                  'RightUpLeg', 'RightLeg', 'RightFoot',
                  'LeftUpLeg', 'LeftLeg', 'LeftFoot',
                  'reference']
        root_joint = 'Hips'
    elif skeleton_type == 'smpl':
        joints = ['l_hip', 'l_knee', 'l_ankle', 'l_foot',
                  'r_hip', 'r_knee', 'r_ankle', 'r_foot',
                  'spine1', 'spine2', 'spine3', 'neck', 'head',
                  'l_collar', 'l_shoulder', 'l_elbow', 'l_wrist', 'l_hand',
                  'r_collar', 'r_shoulder', 'r_elbow', 'r_wrist', 'r_hand',
                  'reference']
        root_joint = 'pelvis'

    t1 = JointSelector(joints, include_root=True)
    t2 = RootTransformer('pos_rot_deltas', separate_root=False)
    t3 = MocapParameterizer('expmap', ref_pose=None)
    t4 = ConstantsRemover(eps=1e-9)  # TODO 默认的1e-6会把smpl的l_foot/r_foot/l_hand/r_hand也删掉，其实这个类的功能，应该是删掉根节点的XZposition
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
