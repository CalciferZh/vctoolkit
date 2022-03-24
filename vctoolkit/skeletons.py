import numpy as np


def get_finger_color(labels):
  color = []
  for l in labels:
    if 'T' in l:
      c = [255, 0, 0]
    elif 'I' in l:
      c = [0, 255, 0]
    elif 'M' in l:
      c = [0, 0, 255]
    elif 'R' in l:
      c = [0, 255, 255]
    elif 'L' in l:
      c = [255, 0, 255]
    else:
      c = [255, 255, 0]
    color.append(c)
  return color


def get_body_color(labels):
  color = []
  for l in labels:
    if 'left' in l:
      c = [255, 0, 0]
    elif 'right' in l:
      c = [0, 255, 0]
    else:
      c = [0, 0, 255]
    color.append(c)
  return color


def convert_skeleton(data, src_skeleton, tgt_skeleton, axis=0):
  src_data = np.swapaxes(data, 0, axis)
  tgt_data = []
  for l in tgt_skeleton.labels:
    tgt_data.append(src_data[src_skeleton.labels.index(l)])
  tgt_data = np.swapaxes(np.stack(tgt_data), 0, axis).copy()
  return tgt_data


class BaseSkeleton:
  labels = []

  parents = [] # parent keypoint in the kinematic tree

  n_keypoints = 0

  center = None # for alignment

  root = None # the root joint in the kinematics tree


class COCOBody(BaseSkeleton):
  n_keypoints = 17

  labels = [
    "nose", # 0
    "left_eye", "right_eye", # 2
    "left_ear", "right_ear", # 4
    "left_shoulder", "right_shoulder", # 6
    "left_elbow", "right_elbow", # 8
    "left_wrist", "right_wrist", # 10
    "left_hip", "right_hip", # 12
    "left_knee", "right_knee", # 14
    "left_ankle", "right_ankle" # 17
  ]

  parents = [
    None,
    0, 0,
    1, 2,
    None, None,
    5, 6,
    7, 8,
    None, None,
    11, 12,
    13, 14
  ]

  center = root = None

  colors = get_body_color(labels)


class MPII2DBody(BaseSkeleton):
  n_keypoints = 16

  labels = [
    'right_ankle', 'right_knee', 'right_hip', # 2
    'left_hip', 'left_knee', 'left_ankle', # 5
    'pelvis', 'thorax', 'upper_neck', 'head_top', # 9
    'right_wrist', 'right_elbow', 'right_shoulder', # 12
    'left_shoulder', 'left_elbow', 'left_wrist' # 15
  ]

  parents = [
    1, 2, 6,
    6, 3, 4,
    None, 6, 7, 8,
    11, 12, 7,
    7, 13, 14
  ]

  center = root = 6

  colors = get_body_color(labels)


class MPII3DBody28(BaseSkeleton):
  n_keypoints = 28

  labels = [
    'spine3', 'spine4', 'spine2', 'spine', 'pelvis', # 4
    'neck', 'head', 'head_top', # 7
    'left_clavical', 'left_shoulder', 'left_elbow', 'left_wrist', 'left_hand', # 12
    'right_clavical', 'right_shoulder', 'right_elbow', 'right_wrist', 'right_hand', # 17
    'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe', # 22
    'right_hip', 'right_knee', 'right_ankle', 'right_foot', 'right_toe', # 27
  ]

  parents = [
    2, 0, 3, 4,
    None, 1, 5, 6,
    1, 8, 9, 10, 11,
    1, 13, 14, 15, 16,
    4, 18, 19, 20, 21,
    4, 23, 24, 25, 26
  ]

  center = root = 4

  colors = get_body_color(labels)


class MPII3DBody17(BaseSkeleton):
  n_keypoints = 17

  labels = [
    'head_top', 'neck', # 1
    'right_shoulder', 'right_elbow', 'right_wrist', # 4
    'left_shoulder', 'left_elbow', 'left_wrist', # 7
    'right_hip', 'right_knee', 'right_ankle', # 10
    'left_hip', 'left_knee', 'left_ankle', # 13
    'pelvis', 'spine', 'head' # 16
  ]

  parents = [
    16, 15,
    1, 2, 3,
    1, 5, 6,
    14, 8, 9,
    14, 11, 12,
    None, 14, 1
  ]

  center = root = 14

  colors = get_body_color(labels)


class HM36MBody23(BaseSkeleton):
  labels = [
    'pelvis', 'spine', 'neck', 'head', 'head_top', # 4
    'left_shoulder', 'left_elbow', 'left_wrist', 'left_hand', # 8
    'right_shoulder', 'right_elbow', 'right_wrist', 'right_hand', # 12
    'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe', # 17
    'right_hip', 'right_knee', 'right_ankle', 'right_foot', 'right_toe' # 22
  ]

  n_keypoints = 23

  parents = [
    None, 0, 1, 2, 3,
    2, 5, 6, 7,
    2, 9, 10, 11,
    0, 13, 14, 15, 16,
    0, 18, 19, 20, 21
  ]

  center = root = 0

  colors = get_body_color(labels)


class HM36MBody17(BaseSkeleton):
  labels = [
    'pelvis', 'spine', 'neck', 'head', 'head_top', # 4
    'left_shoulder', 'left_elbow', 'left_wrist', # 7
    'right_shoulder', 'right_elbow', 'right_wrist', # 10
    'left_hip', 'left_knee', 'left_ankle',  # 13
    'right_hip', 'right_knee', 'right_ankle', # 16
  ]

  n_keypoints = 17

  parents = [
    None, 0, 1, 2, 3,
    2, 5, 6,
    2, 8, 9,
    0, 11, 12,
    0, 14, 15
  ]

  center = root = 0

  colors = get_body_color(labels)


class HM36MBody32(BaseSkeleton):
  labels = [
    'pelvis', # 0
    'right_hip', 'right_knee', 'right_ankle', 'right_foot', 'right_toe', # 5
    'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe', # 10
    'pelvis_duplicate', 'spine', 'neck', 'head', 'head_top', # 15
    'unknown', # 16
    'left_shoulder', 'left_elbow', 'left_wrist', # 19
    'unknown', 'unknown', # 21
    'left_hand', # 22
    'unknown', 'unknown', # 24
    'right_shoulder', 'right_elbow', 'right_wrist', # 27
    'unknown', 'unknown', # 29
    'right_hand', # 30
    'unknown', 'unknown', # 32
  ]

  n_keypoints = 32

  parents = [
    None,
    0, 1, 2, 3, 4,
    0, 6, 7, 8, 9,
    None, 0, 12, 13, 14,
    None,
    12, 17, 18,
    None, None,
    19,
    None, None,
    12, 25, 26,
    None, None,
    27,
    None, None
  ]

  center = root = 0

  colors = get_body_color(labels)


class HUMBIBody33(BaseSkeleton):
  n_keypoints = 33

  labels = [
    'pelvis', # 0
    'left_hip', 'right_hip', # 2
    'lowerback', # 3
    'left_knee', 'right_knee', # 5
    'upperback', # 6
    'left_ankle', 'right_ankle', # 8
    'thorax', # 9
    'left_toes', 'right_toes', # 11
    'lowerneck', # 12
    'left_clavicle', 'right_clavicle', # 14
    'upperneck', # 15
    'left_shoulder', 'right_shoulder', # 17
    'left_elbow', 'right_elbow', # 19
    'left_wrist', 'right_wrist', # 21
    # the fake hand joints in SMPL are removed
    # following are extended keypoints
    'head_top', 'left_eye', 'right_eye', # 24
    'left_hand_I0', 'left_hand_L0', # 26
    'right_hand_I0', 'right_hand_L0', # 28
    'left_foot_T0', 'left_foot_L0', # 30
    'right_foot_T0', 'right_foot_L0', # 32
  ]

  parents = [
    None,
    0, 0,
    0,
    1, 2,
    3,
    4, 5,
    6,
    7, 8,
    9,
    9, 9,
    12,
    13, 14,
    16, 17,
    18, 19,
    # extended
    15, 15, 15,
    20, 20,
    21, 21,
    7, 7,
    8, 8
  ]

  center = root = 0

  # the vertex indices of the extended keypoints
  extended_verts = {
    22: 411, 23: 2800, 24: 6260,
    25: 2135, 26: 2062,
    27: 5595, 28: 5525,
    29: 3292, 30: 3318,
    31: 6691, 32: 6718
  }

  colors = get_body_color(labels)


class MTCBody(BaseSkeleton):
  n_keypoints = 19

  labels = [
    'neck', 'nose', 'pelvis', # 2
    'left_shoulder', 'left_elbow', 'left_wrist', # 5
    'left_hip', 'left_knee', 'left_ankle', # 8
    'right_shoulder', 'right_elbow', 'right_wrist', # 11
    'right_hip', 'right_knee', 'right_ankle', # 14
    'left_eye', 'left_ear', # 16
    'right_eye', 'right_ear' # 18
  ]

  parents = [
    2, 0, None,
    0, 3, 4,
    2, 6, 7,
    0, 9, 10,
    2, 12, 13,
    1, 15,
    1, 17
  ]

  center = root = 2

  colors = get_body_color(labels)


class MANOHand(BaseSkeleton):
  n_keypoints = 21

  center = 4

  root = 0

  labels = [
    'W', #0
    'I0', 'I1', 'I2', #3
    'M0', 'M1', 'M2', #6
    'L0', 'L1', 'L2', #9
    'R0', 'R1', 'R2', #12
    'T0', 'T1', 'T2', #15
    'I3', 'M3', 'L3', 'R3', 'T3' #20, tips are manually added (not in MANO)
  ]

  # finger tips are not keypoints in MANO, we label them on the mesh manually
  extended_keypoints = {16: 333, 17: 444, 18: 672, 19: 555, 20: 744}

  parents = [
    None,
    0, 1, 2,
    0, 4, 5,
    0, 7, 8,
    0, 10, 11,
    0, 13, 14,
    3, 6, 9, 12, 15
  ]

  colors = get_finger_color(labels)


class MPIIHand(BaseSkeleton):
  n_keypoints = 21

  center = 9

  root = 0

  labels = [
    'W', #0
    'T0', 'T1', 'T2', 'T3', #4
    'I0', 'I1', 'I2', 'I3', #8
    'M0', 'M1', 'M2', 'M3', #12
    'R0', 'R1', 'R2', 'R3', #16
    'L0', 'L1', 'L2', 'L3', #20
  ]

  parents = [
    None,
    0, 1, 2, 3,
    0, 5, 6, 7,
    0, 9, 10, 11,
    0, 13, 14, 15,
    0, 17, 18, 19
  ]

  colors = get_finger_color(labels)


class SMPLH65(BaseSkeleton):
  n_keypoints = 65

  center = root = 0

  labels = [
    'pelvis', # 0
    'left_hip', 'right_hip', # 2
    'lowerback', # 3
    'left_knee', 'right_knee', # 5
    'upperback', # 6
    'left_ankle', 'right_ankle', # 8
    'thorax', # 9
    'left_toes', 'right_toes', # 11
    'lowerneck', # 12
    'left_clavicle', 'right_clavicle', # 14
    'upperneck', # 15
    'left_shoulder', 'right_shoulder', # 17
    'left_elbow', 'right_elbow', # 19
    'left_wrist', 'right_wrist', # 21
    # left hand
    'LI0', 'LI1', 'LI2', # 24
    'LM0', 'LM1', 'LM2', # 27
    'LL0', 'LL1', 'LL2', # 30
    'LR0', 'LR1', 'LR2', # 33
    'LT0', 'LT1', 'LT2', # 36
    # right hand
    'RI0', 'RI1', 'RI2', # 39
    'RM0', 'RM1', 'RM2', # 42
    'RL0', 'RL1', 'RL2', # 45
    'RR0', 'RR1', 'RR2', # 48
    'RT0', 'RT1', 'RT2', # 51
    # extended keypoints
    # hand tips
    'LI3', 'LM3', 'LL3', 'LR3', 'LT3', # 56
    'RI3', 'RM3', 'RL3', 'RR3', 'RT3', # 61
    # foot middle toe root
    'LMT', 'RMT' # 63
    # head top
    'head_top' # 64
  ]

  parents = [
    None,
    0, 0,
    0,
    1, 2,
    3,
    4, 5,
    6,
    7, 8,
    9,
    9, 9,
    12,
    13, 14,
    16, 17,
    18, 19,
    # left hand
    20, 22, 23,
    20, 25, 26,
    20, 28, 29,
    20, 31, 32,
    20, 34, 35,
    # right hand
    21, 37, 38,
    21, 40, 41,
    21, 43, 44,
    21, 46, 47,
    21, 49, 50,
    # extended keypoints
    24, 27, 30, 33, 36,
    39, 42, 45, 48, 51,
    10, 11,
    15
  ]

  # the vertex indices of the extended keypoints
  extended_keypoints = {
    52: 2319, 53: 2445, 54: 2673, 55: 2556, 56: 2746,
    57: 5782, 58: 5905, 59: 6133, 60: 6016, 61: 6206,
    62: 3255, 63: 6655, 64: 411
  }

  colors = get_body_color(labels)


class SMPLBody26(BaseSkeleton):
  n_keypoints = 26

  center = root = 0

  labels = [
    'pelvis', # 0
    'left_hip', 'right_hip', # 2
    'lowerback', # 3
    'left_knee', 'right_knee', # 5
    'upperback', # 6
    'left_ankle', 'right_ankle', # 8
    'thorax', # 9
    'left_toes', 'right_toes', # 11
    'lowerneck', # 12
    'left_clavicle', 'right_clavicle', # 14
    'upperneck', # 15
    'left_shoulder', 'right_shoulder', # 17
    'left_elbow', 'right_elbow', # 19
    'left_wrist', 'right_wrist', # 21
    'left_hand', 'right_hand', # 23
    'left_hand_tip', 'right_hand_tip' # 25
  ]

  parents = [
    None,
    0, 0,
    0,
    1, 2,
    3,
    4, 5,
    6,
    7, 8,
    9,
    9, 9,
    12,
    13, 14,
    16, 17,
    18, 19,
    20, 21,
    22, 23
  ]

  # the vertex indices of the extended keypoints
  extended_keypoints = {
    24: 2445, 25: 5905
  }

  colors = get_body_color(labels)


class LSPBody(BaseSkeleton):
  n_keypoints = 14

  labels = [
    'right_ankle', 'right_knee', 'right_hip', # 2
    'left_hip', 'left_knee', 'left_ankle', # 5
    'right_wrist', 'right_elbow', 'right_shoulder', # 8
    'left_shoulder', 'left_elbow', 'left_wrist', # 11
    'neck', 'head_top' # 13
  ]

  parents = [
    1, 2, 12,
    6, 3, 12,
    11, 12, 7,
    7, 13, 14,
    None, 12
  ]

  center = root = 12

  colors = get_body_color(labels)
