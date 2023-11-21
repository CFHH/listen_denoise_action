import numpy as np

########################################################################################################################
# SMPL顺序
########################################################################################################################
# SMPL的默认骨骼顺序【这个顺序也是aist++等一些数据集中的顺序，不能改】
SMPL_JOINTS_NAMES = [
    "pelvis",       #0
    "l_hip",        #1
    "r_hip",        #2
    "spine1",       #3
    "l_knee",       #4
    "r_knee",       #5
    "spine2",       #6
    "l_ankle",      #7
    "r_ankle",      #8
    "spine3",       #9
    "l_foot",       #10
    "r_foot",       #11
    "neck",         #12
    "l_collar",     #13
    "r_collar",     #14
    "head",         #15
    "l_shoulder",   #16
    "r_shoulder",   #17
    "l_elbow",      #18
    "r_elbow",      #19
    "l_wrist",      #20
    "r_wrist",      #21
    "l_hand",       #22 四根手指与手掌的连接处
    "r_hand",       #23
]

# SMPL_JOINTS_NAMES中各节点的父节点的索引
SMPL_JOINTS_PARENTS = [
    -1, 0, 0, 0, 1,
    2, 3, 4, 5, 6,
    7, 8, 9, 9, 9,
    12, 13, 14, 16, 17,
    18, 19, 20, 21,
]

# SMPL顺序下，T-pos状态各节点的一组世界坐标（不是相对父节点的坐标）
SMPL_JOINTS_TPOS = np.array(
    [
        [-8.76308970e-04, -2.11418723e-01, 2.78211200e-02],     #0
        [7.04848876e-02, -3.01002533e-01, 1.97749280e-02],      #1
        [-6.98883278e-02, -3.00379160e-01, 2.30254335e-02],     #2
        [-3.38451650e-03, -1.08161861e-01, 5.63597909e-03],     #3
        [1.01153808e-01, -6.65211904e-01, 1.30860155e-02],      #4
        [-1.06040718e-01, -6.71029623e-01, 1.38401121e-02],     #5
        [1.96440985e-04, 1.94957852e-02, 3.92296547e-03],       #6
        [8.95999143e-02, -1.04856032e00, -3.04155922e-02],      #7
        [-9.20120818e-02, -1.05466743e00, -2.80514913e-02],     #8
        [2.22362284e-03, 6.85680141e-02, 3.17901760e-02],       #9
        [1.12937580e-01, -1.10320516e00, 8.39545265e-02],       #10
        [-1.14055299e-01, -1.10107698e00, 8.98482216e-02],      #11
        [2.60992373e-04, 2.76811197e-01, -1.79753042e-02],      #12
        [7.75218998e-02, 1.86348444e-01, -5.08464100e-03],      #13
        [-7.48091986e-02, 1.84174211e-01, -1.00204779e-02],     #14
        [3.77815350e-03, 3.39133394e-01, 3.22299558e-02],       #15
        [1.62839013e-01, 2.18087461e-01, -1.23774789e-02],      #16
        [-1.64012068e-01, 2.16959041e-01, -1.98226746e-02],     #17
        [4.14086325e-01, 2.06120683e-01, -3.98959248e-02],      #18
        [-4.10001734e-01, 2.03806676e-01, -3.99843890e-02],     #19
        [6.52105424e-01, 2.15127546e-01, -3.98521818e-02],      #20
        [-6.55178550e-01, 2.12428626e-01, -4.35159074e-02],     #21
        [7.31773168e-01, 2.05445019e-01, -5.30577698e-02],      #22
        [-7.35578759e-01, 2.05180646e-01, -5.39352281e-02],     #23
    ]
)

# SMPL顺序下，各节点相对父节点的的偏移
def get_smpl_offsets(default_root_y=0.0):
    smpl_offset = np.zeros([24, 3])
    smpl_offset[0] = [0.0, default_root_y, 0.0]
    for idx, pid in enumerate(SMPL_JOINTS_PARENTS[1:]):
        smpl_offset[idx + 1] = SMPL_JOINTS_TPOS[idx + 1] - SMPL_JOINTS_TPOS[pid]
    return smpl_offset


########################################################################################################################
# BVH顺序
# 也就是bvh文件中各骨骼节点的出场次序，即：
# ['pelvis', 'l_hip', 'l_knee', 'l_ankle', 'l_foot', 'r_hip', 'r_knee', 'r_ankle', 'r_foot', 'spine1',
#  'spine2', 'spine3', 'neck', 'head', 'l_collar', 'l_shoulder', 'l_elbow', 'l_wrist', 'l_hand',
#  'r_collar', 'r_shoulder', 'r_elbow', 'r_wrist', 'r_hand']
########################################################################################################################
# 把smpl顺序变成bvh顺序，来源最早是write_smpl_bvh()的中间产物
ROTATION_SMPL2BVH = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23]
ROTATION_SEQ = ROTATION_SMPL2BVH

# 把bvh顺序变成smpl顺序
# ROTATION_BVH2SMPL = ROTATION_SMPL2BVH.copy()
# for i in range(24):
#     ROTATION_BVH2SMPL[ROTATION_SMPL2BVH[i]] = i
ROTATION_BVH2SMPL = [0, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 14, 19, 13, 15, 20, 16, 21, 17, 22, 18, 23]
ROTATION_SEQ_INV = ROTATION_BVH2SMPL

# bvh顺序下，左右互换（动作镜像）
BVH_MIRROR_SEQ = [0, 5, 6, 7, 8, 1, 2, 3, 4, 9, 10, 11, 12, 13, 19, 20, 21, 22, 23, 14, 15, 16, 17, 18]

# bvh顺序下，各骨骼节点的出场次序
BVH_JOINTS_NAMES = [SMPL_JOINTS_NAMES[i] for i in ROTATION_SMPL2BVH]
def get_bvh_names():
    return BVH_JOINTS_NAMES.copy()

# bvh顺序下，可以和地面解除的节点（'l_ankle', 'l_foot', 'r_ankle', 'r_foot'）
BVH_CONTACT_ID = [3, 4, 7, 8]

# bvh顺序下，各骨骼的父节点的索引，即BVH_JOINTS_NAMES中各节点的父节点
BVH_JOINTS_PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 17, 11, 19, 20, 21, 22]
def get_bvh_parents():
    return BVH_JOINTS_PARENTS.copy()

# SMPL顺序下，各节点相对父节点的的偏移
def get_bvh_offsets(default_root_y=1.0):
    smpl_offset = get_smpl_offsets(default_root_y=default_root_y)
    return smpl_offset[ROTATION_SMPL2BVH]


########################################################################################################################
# 写bvh文件
#    如果用bvhacker打开文件报错: Problem with file: was the file saved using Mac line ending?
#    并不是行结尾是\n还是\r\n的问题，还是一行超过了2048个字符
########################################################################################################################
def write_smpl_bvh(filename, names, parent, offset, xyz_ratation_order, positions, rotations, frametime,
                   scale100=False, rotation_order='smpl'):
    """
    用下面的save_motion_as_bvh吧，这个参数太复杂
    :param filename          : 待保存的文件名
    :param names             : smpl顺序所有骨骼的名字，SMPL_JOINTS_NAMES
    :param parent            : smpl顺序的父节点索引，SMPL_JOINTS_PARENTS
    :param offset            : SMPL顺序各节点相对父节点的的偏移，get_smpl_offsets
    :param xyz_ratation_order: rotations中的3个欧拉角的顺序，如'xyz'
    :param positions         : 根骨骼的位置，单位米【TODO 注意！这里已经受offset修正】
    :param rotations         : 各骨骼的旋转，xyz欧拉角，用角度表示
    :param frametime         : 1/fps
    :param scale100          : 保存为文件时是否把表示距离的数据放大100倍（当作是单位从米变成厘米）；为了在bvhacker里显示，默认True
    :param rotation_order    : ['smpl', 'bvh'], rotations中骨骼节点的顺序，模式是smpl顺序，写入bvh需要转成bvh顺序
    :return:
    """
    file = open(filename, 'w')
    joints_num = len(names)
    xyz_ratation_order = xyz_ratation_order.upper()
    frames = rotations.shape[0]

    file_string = 'HIERARCHY\n'
    seq = [] # 最终就是 ROTATION_SMPL2BVH/ROTATION_SEQ

    def write_static(idx, prefix):
        nonlocal names, parent, offset, xyz_ratation_order, positions, rotations, frames, file_string, seq
        seq.append(idx)
        if idx == 0:
            name_label = 'ROOT ' + names[idx]
            channel_label = 'CHANNELS 6 Xposition Yposition Zposition {}rotation {}rotation {}rotation'.format(*xyz_ratation_order)
        else:
            name_label = 'JOINT ' + names[idx]
            channel_label = 'CHANNELS 3 {}rotation {}rotation {}rotation'.format(*xyz_ratation_order)
        if scale100:
            offset_label = 'OFFSET %.6f %.6f %.6f' % (offset[idx][0] * 100.0, offset[idx][1] * 100.0, offset[idx][2] * 100.0)
        else:
            offset_label = 'OFFSET %.6f %.6f %.6f' % (offset[idx][0], offset[idx][1], offset[idx][2])

        file_string += prefix + name_label + '\n'
        file_string += prefix + '{\n'
        file_string += prefix + '\t' + offset_label + '\n'
        file_string += prefix + '\t' + channel_label + '\n'

        has_child = False
        for y in range(idx+1, joints_num):
            if parent[y] == idx:
                has_child = True
                write_static(y, prefix + '\t')
        if not has_child:
            file_string += prefix + '\t' + 'End Site\n'
            file_string += prefix + '\t' + '{\n'
            file_string += prefix + '\t\t' + 'OFFSET 0 0 0\n'
            file_string += prefix + '\t' + '}\n'

        file_string += prefix + '}\n'

    write_static(0, '')

    file_string += 'MOTION\n' + 'Frames: {}\n'.format(frames) + 'Frame Time: %.8f\n' % frametime
    for i in range(frames):
        if scale100:
            file_string += '%.6f %.6f %.6f ' % (positions[i][0] * 100.0, positions[i][1] * 100.0, positions[i][2] * 100.0)
        else:
            file_string += '%.6f %.6f %.6f ' % (positions[i][0], positions[i][1], positions[i][2])
        for j in range(joints_num):
            if rotation_order == 'smpl':
                idx = seq[j]
            elif rotation_order == 'bvh':
                idx = j
            else:
                raise Exception('Unknown rotation order')
            file_string += '%.6f %.6f %.6f ' % (rotations[i][idx][0], rotations[i][idx][1], rotations[i][idx][2])
        file_string += '\n'

    file.write(file_string)
    file.close()

    """
    seq_string = '['
    for i in range(len(seq)):
        if i != len(seq) -1:
            seq_string += '%d, ' % seq[i]
        else:
            seq_string += '%d' % seq[i]
    seq_string += ']'
    """
    return

def save_motion_as_bvh(filename, positions, rotations, frametime, scale100=True, rotation_order='smpl'):
    """
    :param filename      : 待保存的文件名
    :param positions     : 根骨骼的位置，单位米【TODO 注意！这里需要包含bvh文件中根骨骼offset的值】
    :param rotations     : 各骨骼的旋转，xyz欧拉角，用角度表示
    :param frametime     : 1/fps
    :param scale100      : 保存为文件时是否把表示距离的数据放大100倍（当作是单位从米变成厘米）；为了在bvhacker里显示，默认True
    :param rotation_order: ['smpl', 'bvh'], rotations中骨骼节点的顺序，模式是smpl顺序，写入bvh需要转成bvh顺序
    :return:
    """
    #################################################################################################
    # 关于positions和smpl_offsets[0]（即bvh文件中根节点的OFFSET）的关系：
    # 一、不同的软件和开源代码对此的处理也不相同
    #     1、bvhhacker中，根节点（即pelvis）的OFFSET中Y越大，人越高
    #     2、blender中，pelvis的OFFSET中Y无论是多少，人的位置没变化，相当于是默认100的情形
    #     3、代码ForwardKinematicsJoint类中，构造函数中的offsets[0, 1]（即pelvis的OFFSET的Y）是没有意义的
    #     4、AIST数据集（非bvh格式），人物跟骨骼的Y在1.9左右
    #     TODO 但实际情况，pelvis是胯部，确实0.9米左右比较合适，1.9米属于不正常
    # 二、由于历史关系，受最初的ganimator和aist的影响，暂时规定格式positions和smpl_offsets[0]需要满足下列条件：
    #     基本原则:
    #           当输入参数或返回值，同时有root_position和offset时，root_position不含offset[0]
    #           当输入参数或返回值，只有root_position（没有offset）时，root_position含offset[0]
    #           读写文件也类似处理
    #     比如
    #     调用load_bvh_motion时：
    #           1、同时返回两者，所以root_position不包含offset[0]
    #     调用ForwardKinematicsJoint时：
    #           1、构造函数中需要offset，forward函数需要root_position，所以root_position应该不含offset[0]
    #     以scale100=True调用save_motion_as_bvh保存为bvh文件时：
    #           1、调用参数没有offset，所以positions应该加上smpl_offsets[0]
    #           2、保存bvh文件中，pelvis的OFFSET填(0, 100, 0)，帧数据中的Y就是90左右而不是190
    #           3、帧数据中的Y，需要满足在blender里查看，认为此帧脚该着地时，脚应该差不多刚好在地面上
    #     以scale100=False调用save_motion_as_bvh保存为bvh文件时【目前没有用处】：
    #           1、调用参数没有offset，所以positions应该加上smpl_offsets[0]
    #           2、保存bvh文件中，pelvis的OFFSET填(0, 0, 0)，TODO 帧数据该如何
    #           3、TODO blender显示什么情况没试过
    #################################################################################################

    smpl_offsets = get_smpl_offsets(default_root_y=0.0)
    if scale100:
        smpl_offsets[0][1] = 1.0
        positions[:, 1] -= 1.0
    write_smpl_bvh(filename, SMPL_JOINTS_NAMES, SMPL_JOINTS_PARENTS, smpl_offsets, 'xyz', positions, rotations,
                   frametime, scale100=scale100, rotation_order=rotation_order)
    if scale100:
        positions[:, 1] += 1.0

def test():
    dummy_fps = 60
    dummy_frames = 10
    base_position = [0.0, 1.9, 0.0]
    positions = np.zeros([dummy_frames, 3])
    for i in range(dummy_frames):
        positions[i] = base_position
    ratations = np.zeros([dummy_frames, 24, 3])
    # 实测得，旋转角度的含义是绕XYZ轴，按右手螺旋旋转，是intrinsic rotations
    # intrinsic rotations的含义就是旋转轴是跟随物体的，见
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_euler.html#r72d546869407-1
    # X：人物自身的右脚朝左脚；Y：指向人物头顶方向；Z：右手系根据XY确定Z，人物前方
    # 先X再Y（x=75，y=90），先Y再Z（y=75，z=90），先X再Z（x=90，z=90）
    #
    # UE坐标系，X是人物前方，Y是右方，Z是上方
    # bvh的x旋转对应UE的y旋转，bvh的y旋转对应UE的-z旋转，bvh的z旋转对应UE的-x旋转
    # UE的x旋转对应bvh的-z旋转，UE的y旋转对应bvh的x旋转，UE的z旋转对应bvh的-y旋转
    # bvh里按xyz旋转（a,b,c），对应UE里按yzx旋转（a,-b,-c）
    #
    rot_x = 75
    rot_y = 90
    rot_z = 0
    ratations[:, 0, 0] = rot_x
    ratations[:, 0, 1] = rot_y
    ratations[:, 0, 2] = rot_z
    filename = './root_x%dy%dz%d.bvh' % (rot_x, rot_y, rot_z)
    save_motion_as_bvh(filename, positions, ratations, 1.0/dummy_fps, scale100=True, rotation_order='smpl')

if __name__ == '__main__':
    test()