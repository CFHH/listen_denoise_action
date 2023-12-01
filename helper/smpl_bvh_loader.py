import numpy as np
import re, threading
import math
import torch
from .bvh import Bvh
from .smpl_bvh_writer import BVH_JOINTS_NAMES, BVH_JOINTS_PARENTS

def load_bvh_motion_new(filename, data_scaled100 = True, fill_to_2exp = False):
    '''
    新写的版本
    '''
    with open(filename) as f:
        bvh = Bvh(f.read())
        frame_time = bvh.frame_time
        all_nodes = bvh.search('ROOT') +  bvh.search('JOINT')
        parents = []
        names = []
        offsets = np.array([]).reshape((0, 3))
        for n in all_nodes:
            # names
            names.append(n.name)
 
            # parents
            ji = bvh.joint_parent_index(n.name)
            parents.append(ji)

            # offsets
            off = bvh.joint_offset(n.name)
            off = np.array([off])
            offsets = np.append(offsets, off, axis=0)

        node_cnt = len(names)
        positions = np.array([]).reshape((0, 3))
        rotations = np.array([]).reshape((0, node_cnt * 3))
 
        # frame data 
        for frame in bvh.frames:
            data_block = np.array(list(map(float, frame)))
            assert len(data_block) == node_cnt * 3 + 3
            position = data_block[0:3]
            rotation = data_block[3:]
            positions = np.append(positions, position.reshape(1, 3), axis=0)
            rotations = np.append(rotations, rotation.reshape(1, node_cnt * 3), axis=0)

    #剩余空间都用最后的数据填满，凑成2的N次方个
    # if fill_to_2exp:
    #     while i < frames:
    #         positions[i] = position
    #         rotations[i] = rotation
    #         i += 1

    if data_scaled100:
        positions /= 100.0
        offsets /= 100.0
    rotations = rotations.reshape(-1, node_cnt, 3)

    return positions, rotations, frame_time, names, parents, offsets

def load_bvh_motion(filename, data_scaled100 = True, fill_to_2exp = False):
    # positions, rotations, frametime, names, parents, offsets = load_bvh_motion_old(filename, data_scaled100, fill_to_2exp)
    root_positions, rotations, frametime, names, parents, offsets = load_bvh_motion_new(filename, data_scaled100, fill_to_2exp)
    return root_positions, rotations, frametime, names, parents, offsets

def load_bvh_motion_old(filename, data_scaled100 = True, fill_to_2exp = False):
    """
    老的简单的版本
    :param filename: bvh文件的完整路径名字
    :param data_scaled: bvh中的offset、position数据是否已经放大100倍
    :return:
        positions     : np,(帧数,3),各帧根骨骼的位移（单位是米）
        rotations     : np,(帧数,24,3),各帧各骨骼节点的欧拉角（单位是角度）
        frametime     : float,每帧时间
        bvh_names     : list,(24,),各骨骼节点的名字,24按bvh中出场次序(下同)
        bvh_parents   : list,(24,),各骨骼节点的父节点索引
        offsets       : np,(24,3),各骨骼节点在初始状态下的对父节点的位移（单位是米）,也就是bvh中的OFFSET
    """
    cur_step = 0
    offsets = np.array([]).reshape((0, 3))
    endsites = np.array([]).reshape((0, 3))
    read_offset = False
    read_for_endsite = False
    f = open(filename, "r")
    for line in f:
        if cur_step == 0:
            if read_offset:
                offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
                if offmatch:
                    read_offset = False
                    if read_for_endsite:
                        endsites = np.append(endsites, np.array([[0, 0, 0]]), axis=0)
                        endsites[-1] = np.array([list(map(float, offmatch.groups()))])
                    else:
                        offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
                        offsets[-1] = np.array([list(map(float, offmatch.groups()))])
            elif "ROOT" in line or "JOINT" in line:
                read_offset = True
                read_for_endsite = False
            elif "End Site" in line:
                read_offset = True
                read_for_endsite = True
            elif "MOTION" in line:
                cur_step = 1
            continue
        if cur_step == 1:
            cur_step = 2
            fmatch = re.match("\s*Frames:\s+(\d+)", line)
            frames = int(fmatch.group(1))
            if fill_to_2exp:
                # 需要的情况下凑成2的N次方个，默认关闭
                frames = 2 ** (int(math.log2(frames))+1)
            positions = np.zeros((frames, 3), dtype=np.float32)
            rotations = np.zeros((frames, 72), dtype=np.float32)
            continue
        if cur_step == 2:
            cur_step = 3
            i = 0
            fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
            frametime = float(fmatch.group(1))
            continue
        if cur_step == 3:
            dmatch = line.strip().split()
            data_block = np.array(list(map(float, dmatch)))
            assert len(data_block) == 75
            position = data_block[0:3]
            rotation = data_block[3:]
            positions[i] = position
            rotations[i] = rotation
            i += 1
    f.close()

    # 处理 End Site ，大小是(n,3)，n = 5
    extra_offset = np.average(endsites, axis=0)
    offsets[0] += extra_offset

    #剩余空间都用最后的数据填满，凑成2的N次方个
    if fill_to_2exp:
        while i < frames:
            positions[i] = position
            rotations[i] = rotation
            i += 1

    if data_scaled100:
        positions /= 100.0
        offsets /= 100.0
    rotations = rotations.reshape(-1, 24, 3)

    bvh_parents = BVH_JOINTS_PARENTS.copy()
    bvh_names = BVH_JOINTS_NAMES.copy()
    # 也可以这样： offsets = smpl_bvh_writer.get_bvh_offsets()
    return positions, rotations, frametime, bvh_names, bvh_parents, offsets


def _load_bvh_meta_info(file, device):
    _, _, _, names, parents, offsets = load_bvh_motion(file)

    parents = torch.tensor(parents, dtype=torch.int, device=device)
    offsets = torch.tensor(offsets, dtype=torch.float, device=device)

    meta_info = {
        'names': names,
        'parents':parents,
        'offsets':offsets
    }
    return meta_info

smpl_meta_infos = None

_lock=threading.Lock() #创建线程锁

def get_bvh_meta_info(device, name = 'standard'):
    global smpl_meta_infos, _lock
    _lock.acquire()

    if smpl_meta_infos is None:
        smpl_meta_infos = {}

    if not device.type in smpl_meta_infos:
        smpl_meta_infos[device.type] = {}

    if not name in smpl_meta_infos[device.type]:
        smpl_meta_infos[device.type][name] = _load_bvh_meta_info('./configs/standard_skeleton.bvh', device=device)

    item = smpl_meta_infos[device.type][name]

    _lock.release()

    return item['parents'], item['offsets'], item['names']


