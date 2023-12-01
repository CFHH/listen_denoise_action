import numpy as np


GENEA_JOINTS_NAMES_FULL = [
    'Hips',             #00
        'Spine',            #01
            'Spine1',           #02
                'Spine2',           #03
                    'Spine3',           #04
                        'Neck',             #05
                            'Neck1',            #06
                                'Head',             #07
                        'RightShoulder',    #08
                            'RightArm',         #09
                                'RightForeArm',     #10
                                    'RightHand',        #11
                                        'RightHandThumb1',  #12
                                        'RightHandThumb2',  #13
                                        'RightHandThumb3',  #14
                                        'RightHandIndex1',  #15
                                        'RightHandIndex2',  #16
                                        'RightHandIndex3',  #17
                                        'RightHandMiddle1', #18
                                        'RightHandMiddle2', #19
                                        'RightHandMiddle3', #20
                                        'RightHandRing1',   #21
                                        'RightHandRing2',   #22
                                        'RightHandRing3',   #23
                                        'RightHandPinky1',  #24
                                        'RightHandPinky2',  #25
                                        'RightHandPinky3',  #26
                        'LeftShoulder',     #27
                            'LeftArm',          #28
                                'LeftForeArm',      #29
                                    'LeftHand',         #30
                                        'LeftHandThumb1',   #31
                                        'LeftHandThumb2',   #32
                                        'LeftHandThumb3',   #33
                                        'LeftHandIndex1',   #34
                                        'LeftHandIndex2',   #35
                                        'LeftHandIndex3',   #36
                                        'LeftHandMiddle1',  #37
                                        'LeftHandMiddle2',  #38
                                        'LeftHandMiddle3',  #39
                                        'LeftHandRing1',    #40
                                        'LeftHandRing2',    #41
                                        'LeftHandRing3',    #42
                                        'LeftHandPinky1',   #43
                                        'LeftHandPinky2',   #44
                                        'LeftHandPinky3',   #45
                        'pCube4',           #46
        'RightUpLeg',       #47
            'RightLeg',         #48
                'RightFoot',        #49
                    'RightForeFoot',    #50
                        'RightToeBase',     #51
        'LeftUpLeg',        #52
            'LeftLeg',          #53
                'LeftFoot',         #54
                    'LeftForeFoot',     #55
                        'LeftToeBase'       #56
]

GENEA_JOINTS_PARENTS_FULL = [
    -1,  0,  1,  2,  3,
     4,  5,  6,  4,  8,
     9, 10, 11, 12, 13,
    11, 15, 16, 11, 18,
    19, 11, 21, 22, 11,
    24, 25,  4, 27, 28,
    29, 30, 31, 32, 30,
    34, 35, 30, 37, 38,
    30, 40, 41, 30, 43,
    44, 4,   0, 47, 48,
    49, 50,  0, 52, 53,
    54, 55
]

GENEA_JOINTS_NAMES_SIMPLIFIED = [
    'Hips',                                             #00
        'Spine',                                        #01
            'Spine1',                                   #02
                'Spine2',                               #03
                    'Spine3',                           #04
                        'Neck',                         #05
                            'Neck1',                    #06
                                'Head',                 #07
                        'RightShoulder',                #08
                            'RightArm',                 #09
                                'RightForeArm',         #10
                                    'RightHand',        #11
                        'LeftShoulder',                 #27     #12
                            'LeftArm',                  #28     #13
                                'LeftForeArm',          #29     #14
                                    'LeftHand',         #30     #15
        'RightUpLeg',                                   #47     #16
            'RightLeg',                                 #48     #17
                'RightFoot',                            #49     #18
        'LeftUpLeg',                                    #52     #19
            'LeftLeg',                                  #53     #20
                'LeftFoot',                             #54     #21
]

GENEA_SIMPLIFIED_INDEX = [0, 1, 2, 3, 4, 5, 6, 7, 8 , 9, 10, 11, 27, 28, 29, 30, 47, 48, 49, 52, 53, 54]

GENEA_JOINTS_PARENTS_SIMPLIFIED = [
    -1,  0,  1,  2,  3,
     4,  5,  6,  4,  8,
     9, 10,  4, 12, 13,
    14,  0, 16, 17,  0,
    19, 20
]

#bvh下的旋转顺序
GENEA_ROTATION_ORDER = 'ZXY'

# TPOS下的相对旋转，单位是弧度，旋转顺序是xyz
GENEA_JOINTS_REST_FULL = np.array([
    [-0.1209, 0.0000, 0.0000],
    [-0.0000, -0.0000, 0.0000],
    [-0.0000, -0.0000, 0.0000],
    [0.0000, -0.0000, 0.0000],
    [0.3219, -0.0000, 0.0000],
    [0.0000, -0.0000, 0.0000],
    [0.0000, -0.0000, 0.0000],
    [0.0000, -0.0000, 0.0000],
    [-0.0000, 0.0000, 1.5708],
    [-0.0000, 0.0000, 1.5708],
    [0.0000, 0.0000, 1.5708],
    [-0.0404, 0.0410, 1.5840],
    [0.0000, 0.0000, 1.5708],
    [0.0000, 0.0000, 1.5708],
    [-0.0000, 0.0000, -0.0000],
    [0.0000, 0.0000, 1.5708],
    [0.0000, 0.0000, 1.5708],
    [0.0000, 0.0000, -0.0000],
    [0.0000, 0.0000, 1.5708],
    [0.0000, 0.0000, 1.5708],
    [0.0000, 0.0000, -0.0000],
    [-0.0000, 0.0000, 1.5708],
    [-0.0000, 0.0000, 1.5708],
    [0.0000, 0.0000, -0.0000],
    [0.0000, 0.0000, 1.5708],
    [0.0000, 0.0000, 1.5708],
    [-0.0000, 0.0000, -0.0000],
    [0.0000, -0.0000, -1.5708],
    [0.0000, -0.0000, -1.5708],
    [0.0000, -0.0000, -1.5708],
    [-0.0404, -0.0410, -1.5840],
    [-0.0000, -0.0000, -1.5708],
    [0.0000, -0.0000, -1.5708],
    [0.0000, -0.0000, 0.0000],
    [0.0000, -0.0000, -1.5708],
    [0.0000, -0.0000, -1.5708],
    [-0.0000, -0.0000, 0.0000],
    [0.0000, -0.0000, -1.5708],
    [0.0000, -0.0000, -1.5708],
    [0.0000, -0.0000, 0.0000],
    [0.0000, -0.0000, -1.5708],
    [0.0000, -0.0000, -1.5708],
    [0.0000, -0.0000, 0.0000],
    [-0.0000, -0.0000, -1.5708],
    [-0.0000, -0.0000, -1.5708],
    [0.0000, -0.0000, 0.0000],
    [-0.0000, -0.0000, 0.0000],
    [0.0000, 0.0000, 3.1416],
    [-0.0000, 0.0000, 3.1416],
    [0.0000, 0.0000, 3.1416],
    [1.5664, -0.0000, -0.0000],
    [0.0000, -0.0000, -0.0000],
    [-0.0000, 0.0000, 3.1416],
    [0.0000, -0.0000, -3.1416],
    [0.0000, -0.0000, -3.1416],
    [1.5664, -0.0000, -0.0000],
    [0.0000, 0.0000, 0.0000]
])

GENEA_JOINTS_REST_SIMPLIFIED = GENEA_JOINTS_REST_FULL[GENEA_SIMPLIFIED_INDEX]
