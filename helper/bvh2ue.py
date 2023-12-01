import numpy as np
from helper.smpl_bvh_writer import ROTATION_SMPL2BVH, ROTATION_BVH2SMPL, SMPL_JOINTS_NAMES
from scipy.spatial.transform import Rotation


"""
FOR_OUR_UE_ACTOR
当前美术制作的UE模型，在TPose下各骨骼的相对旋转
单位是弧度
顺序是SMPL顺序，smpl_bvh_writer.SMPL_JOINTS_NAMES
"""
SMPL_JOINTS_REST = np.array(
    [
        [-2.7061, 0.0087, 0.0019],       #0
        [-0.0201, -0.4286, -3.0492],     #1
        [-0.0280, 0.4962, 3.0310],       #2
        [-0.0134, -0.0002, -0.0280],     #3
        [-3.0116, 0.5173, 0.0344],       #4
        [-3.0055, -0.6438, -0.0454],     #5
        [0.5161, 0.0095, -0.0359],       #6
        [1.9630, 0.2827, -0.1897],       #7
        [1.9017, -0.2484, 0.1778],       #8
        [-0.2830, 0.0011, 0.0080],       #9
        [-0.0000, 0.0000, -0.0000],      #10
        [0.0000, -0.0000, -0.0000],      #11
        [0.6775, 0.0155, -0.0439],       #12
        [-0.0801, -0.0554, -1.2102],     #13
        [-0.1030, 0.0714, 1.2112],       #14
        [0.0000, -0.0000, -0.0000],      #15
        [-0.1097, -0.1136, -1.6059],     #16
        [-0.0820, 0.0859, 1.6172],       #17
        [0.0002, 0.0002, -1.5330],       #18
        [-0.0144, 0.0139, 1.5354],       #19
        [-0.1658, -0.1815, -1.6615],     #20
        [-0.1296, 0.1392, 1.6426],       #21
        [0.0000, -0.0000, 0.0000],       #22
        [-0.0000, -0.0000, -0.0000],     #23
    ]
)

SMPL_UE_ROOT_OFFSET = np.array([-0.0363, 91.213097, 4.3399]) / 100


def bvh2ueactor(root_position, rotation):
    """
    :param root_position: [frame, 3]
    :param rotation: [frame, 24, 3]，单位角度
    :return:
    """
    frame, nodes, _ = rotation.shape

    root_position[..., 0] *= -1
    root_position[..., 2] *= -1

    rest_euler = SMPL_JOINTS_REST[ROTATION_SMPL2BVH] #smpl->bvh
    rest_rot = Rotation.from_euler('xyz', rest_euler, degrees=False)
    mat_rest = rest_rot.as_matrix() # [24, 3, 3]
    mat_rest_inv = np.linalg.inv(mat_rest) # [24, 3, 3]

    rotation_euler = rotation.reshape(-1, 3) # [frame*24, 3]
    rot = Rotation.from_euler('XYZ', rotation_euler, degrees=True)
    rot_mat = rot.as_matrix() # [frame*24, 3, 3]
    rot_mat = rot_mat.reshape(frame, -1, 3, 3) # [frame, 24, 3, 3]

    rotation_quat_ue = np.zeros((frame, nodes, 4)) # [frame, 24, 4]
    for i in range(frame):
        rot_mat_i = rot_mat[i] # [24, 3, 3]
        rot_mat_i = (mat_rest_inv @ rot_mat_i @ mat_rest) # [24, 3, 3]
        rot_end = Rotation.from_matrix(rot_mat_i)
        rotation_quat_ue[i] = rot_end.as_quat() # [24, 4]

    rotation_quat_ue[..., 0] *= -1
    rotation_quat_ue[..., 2] *= -1
    return root_position, rotation_quat_ue


def send_motion(rotations, root_pos):
    msg = ""

    for i in range(len(rotations)):
        locationWS = np.zeros(3)
        if i==0:
            locationWS = root_pos
            #print(locationWS)

        rest_rot = Rotation.from_euler('xyz', SMPL_JOINTS_REST[i], degrees=False)
        mat_rest = rest_rot.as_matrix()
        mat_rest_inv = np.linalg.inv(mat_rest)

        rot = Rotation.from_euler('XYZ', rotations[ROTATION_BVH2SMPL[i]], degrees=True)
        rot_mat = rot.as_matrix()

        rot_mat = (mat_rest_inv @ rot_mat @ mat_rest)
        rot_end = Rotation.from_matrix(rot_mat)
        quaternionWS = rot_end.as_quat()

        msg = msg + SMPL_JOINTS_NAMES[i] +  ":" + "{:.9f}".format(locationWS[0])+ "," + "{:.9f}".format(locationWS[2]) +  "," + "{:.9f}".format(locationWS[1]) +  "," + "{:.9f}".format(-quaternionWS[0]) +  "," + "{:.9f}".format(quaternionWS[1])+  "," + "{:.9f}".format(-quaternionWS[2])+ "," + "{:.9f}".format(quaternionWS[3]) + "|"
    msg = msg + "|"
    return msg


from helper.genea_skeleton import GENEA_JOINTS_NAMES_SIMPLIFIED, GENEA_JOINTS_REST_SIMPLIFIED, GENEA_ROTATION_ORDER
def send_motion_genea(rotations, root_pos):
    msg = ""

    for i in range(len(rotations)):
        locationWS = np.zeros(3)
        if i==0:
            locationWS = root_pos
            #print(locationWS)

        rest_rot = Rotation.from_euler('xyz', GENEA_JOINTS_REST_SIMPLIFIED[i], degrees=False)
        mat_rest = rest_rot.as_matrix()
        mat_rest_inv = np.linalg.inv(mat_rest)

        rot = Rotation.from_euler(GENEA_ROTATION_ORDER, rotations[i], degrees=True)
        rot_mat = rot.as_matrix()

        rot_mat = (mat_rest_inv @ rot_mat @ mat_rest)
        rot_end = Rotation.from_matrix(rot_mat)
        quaternionWS = rot_end.as_quat()

        msg = msg + GENEA_JOINTS_NAMES_SIMPLIFIED[i] +  ":" + "{:.9f}".format(locationWS[0])+ "," + "{:.9f}".format(locationWS[2]) +  "," + "{:.9f}".format(locationWS[1]) +  "," + "{:.9f}".format(-quaternionWS[0]) +  "," + "{:.9f}".format(quaternionWS[1])+  "," + "{:.9f}".format(-quaternionWS[2])+ "," + "{:.9f}".format(quaternionWS[3]) + "|"
    msg = msg + "|"
    return msg