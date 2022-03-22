import numpy as np


def convert(rot, src, tar):
  # https://www.euclideanspace.com/maths/geometry/rotations/conversions/
  eps = np.finfo(np.float32).eps
  if src == 'axangle':
    data_shape = rot.shape[:-1]
    rot = np.reshape(rot, [-1, 3])
    if tar == 'quat':
      rad = np.linalg.norm(rot, axis=-1, keepdims=True)
      ax = rot / np.maximum(rad, eps)
      w = np.cos(rad / 2)
      xyz = np.sin(rad / 2) * ax
      quat = np.concatenate([w, xyz], -1)
      quat = np.reshape(quat, data_shape + (4,))
      return quat

    if tar == 'rotmat' or tar == 'rot6d':
      theta = np.linalg.norm(rot, axis=-1, keepdims=True)
      c = np.cos(theta)
      s = np.sin(theta)
      t = 1 - c
      x, y, z = np.split(rot / np.maximum(theta, eps), 3, axis=-1)
      rotmat = np.stack([
        t*x*x + c, t*x*y - z*s, t*x*z + y*s,
        t*x*y + z*s, t*y*y + c, t*y*z - x*s,
        t*x*z - y*s, t*y*z + x*s, t*z*z + c
      ], 1)
      rotmat = np.reshape(rotmat, data_shape + (3, 3))
      if tar == 'rotmat':
        return rotmat
      if tar == 'rot6d':
        return convert(rotmat, 'rotmat', 'rot6d')

  if src == 'quat':
    data_shape = rot.shape[:-1]
    rot = np.reshape(rot, [-1, 4])
    if tar == 'rotmat' or tar == 'rot6d':
      w, x, y, z = np.split(rot, 4, axis=-1)
      rotmat = np.stack([
        1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w,
        2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w,
        2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y
      ], -1)
      rotmat = np.reshape(rotmat, data_shape + (3, 3))
      if tar == 'rotmat':
        return rotmat
      if tar == 'rot6d':
        return convert(rotmat, 'rotmat', 'rot6d')

    if tar == 'axangle':
      angle = 2 * np.arccos(rot[:, 0:1])
      axis = rot[:, 1:] / np.sqrt(1 - np.square(rot[:, 0]))
      axangle = axis * angle
      axangle = np.reshape(axangle, data_shape + (3,))
      return axangle

  if src == 'rotmat':
    data_shape = rot.shape[:-2]
    rot = np.reshape(rot, [-1, 3, 3])
    if tar == 'rot6d':
      rot6d = np.reshape(
        np.transpose(rot[:, :, :2], [0, 2, 1]), data_shape + (6,)
      )
      return rot6d

    if tar == 'axangle':
      angle = np.arccos((rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2] - 1) / 2)
      angle = np.expand_dims(angle, -1)
      norm = np.sqrt(
        np.square(rot[:, 2, 1] - rot[:, 1, 2]) + \
        np.square(rot[:, 0, 2] - rot[:, 2, 0]) + \
        np.square(rot[:, 1, 0] - rot[:, 0, 1])
      )
      norm = np.maximum(norm, np.finfo(np.float32).eps)
      x = (rot[:, 2, 1] - rot[:, 1, 2]) / norm
      y = (rot[:, 0, 2] - rot[:, 2, 0]) / norm
      z = (rot[:, 1, 0] - rot[:, 0, 1]) / norm
      axangle = np.stack([x, y, z], -1) * angle
      axangle = np.reshape(axangle, data_shape + (3,))
      return axangle

    if tar == 'quat':
      quat = []
      for i in range(rot.shape[0]):
        tr = rot[i, 0, 0] + rot[i, 1, 1] + rot[i, 2, 2]
        if tr > 0:
          S = np.sqrt(tr + 1.0) * 2
          qw = 0.25 * S
          qx = (rot[i, 2, 1] - rot[i, 1, 2]) / S
          qy = (rot[i, 0, 2] - rot[i, 2, 0]) / S
          qz = (rot[i, 1, 0] - rot[i, 0, 1]) / S
        elif rot[i, 0, 0] > rot[i, 1, 1] and rot[i, 0, 0] > rot[i, 2, 2]:
          S = np.sqrt(1.0 + rot[i, 0, 0] - rot[i, 1, 1] - rot[i, 2, 2]) * 2
          qw = (rot[i, 2, 1] - rot[i, 1, 2]) / S
          qx = 0.25 * S
          qy = (rot[i, 0, 1] + rot[i, 1, 0]) / S
          qz = (rot[i, 0, 2] + rot[i, 2, 0]) / S
        elif rot[i, 1, 1] > rot[i, 2, 2]:
          S = np.sqrt(1.0 + rot[i, 1, 1] - rot[i, 0, 0] - rot[i, 2, 2]) * 2
          qw = (rot[i, 0, 2] - rot[i, 2, 0]) / S
          qx = (rot[i, 0, 1] + rot[i, 1, 0]) / S
          qy = 0.25 * S
          qz = (rot[i, 1, 2] + rot[i, 2, 1]) / S
        else:
          S = np.sqrt(1.0 + rot[i, 2, 2] - rot[i, 0, 0] - rot[i, 1, 1]) * 2
          qw = (rot[i, 1, 0] - rot[i, 0, 1]) / S
          qx = (rot[i, 0, 2] + rot[i, 2, 0]) / S
          qy = (rot[i, 1, 2] + rot[i, 2, 1]) / S
          qz = 0.25 * S
        quat.append(np.array([qw, qx, qy, qz]))
      quat = np.array(quat)
      quat = np.reshape(quat, data_shape + (4,))
      return quat

  if src == 'rot6d':
    data_shape = rot.shape[:-1]
    rot = np.reshape(rot, [-1, 6])
    col0 = rot[:, 0:3] / \
        np.maximum(np.linalg.norm(rot[:, 0:3], axis=-1, keepdims=True), eps)
    col1 = rot[:, 3:6] - np.sum((col0 * rot[:, 3:6]), axis=-1, keepdims=True) * col0
    col1 = col1 / np.maximum(np.linalg.norm(col1, axis=-1, keepdims=True), eps)
    col2 = np.cross(col0, col1)
    rotmat = np.stack([col0, col1, col2], -1)
    rotmat = np.reshape(rotmat, data_shape + (3, 3))
    if tar == 'rotmat':
      return rotmat
    return convert(rotmat, 'rotmat', tar)

  raise NotImplementedError(f'Unsupported conversion: from {src} to {tar}.')


def rotmat_rel_to_abs(rel_rotmat, parents, batch=False):
  if not batch:
    rel_rotmat = np.expand_dims(rel_rotmat, 0)
  n_joints = len(parents)
  abs_rotmat = [None] * n_joints
  for c in range(n_joints):
    abs_rotmat[c] = rel_rotmat[:, c]
    p = parents[c]
    while p is not None:
      abs_rotmat[c] = \
        np.einsum('nhw, nwk -> nhk', rel_rotmat[:, p], abs_rotmat[c])
      p = parents[p]
  abs_rotmat = np.stack(abs_rotmat, 1)
  if not batch:
    abs_rotmat = abs_rotmat[0]
  return abs_rotmat


def rotmat_abs_to_rel(abs_rotmat, parents, batch=False):
  if not batch:
    abs_rotmat = np.expand_dims(abs_rotmat, 0)
  n_joints = len(parents)
  rel_rotmat = [None] * n_joints
  for c in range(n_joints):
    p = parents[c]
    if p is None:
      rel_rotmat[c] = abs_rotmat[:, c]
    else:
      rel_rotmat[c] = np.einsum(
        'nhw, nwd -> nhd',
        np.transpose(abs_rotmat[:, p], (0, 2, 1)), abs_rotmat[:, c]
      )
  rel_rotmat = np.stack(rel_rotmat, 1)
  if not batch:
    rel_rotmat = rel_rotmat[0]
  return rel_rotmat


def keypoints_to_bones(keypoints, parents, batch=False):
  if not batch:
    keypoints = np.expand_dims(keypoints, 0)
  bones = []
  for c, p in enumerate(parents):
    if p is None:
      bones.append(keypoints[:, c])
    else:
      bones.append(keypoints[:, c] - keypoints[:, p])
  bones = np.stack(bones, 1)
  if not batch:
    bones = bones[0]
  return bones


def bones_to_keypoints(bones, parents, batch=False):
  if not batch:
    bones = np.expand_dims(bones)
  keypoints = []
  for c, p in enumerate(parents):
    if p is None:
      keypoints.append(bones[:, c])
    else:
      keypoints.append(bones[:, c] + keypoints[p])
  keypoints = np.stack(keypoints, 1)
  if not batch:
    keypoints = keypoints[0]
  return keypoints


def forward_kinematics(ref_bones, abs_rotmat, parents, batch=False):
  if not batch:
    abs_rotmat = np.expand_dims(abs_rotmat, 0)
    ref_bones = np.expand_dims(ref_bones, 0)
  bones = np.einsum('njhw, njw -> njh', abs_rotmat, ref_bones)
  keypoints = bones_to_keypoints(bones, parents, batch=True)
  if not batch:
    bones = bones[0]
    keypoints = keypoints[0]
  return keypoints, bones


def sphere_sampling(n):
  theta = np.random.uniform(0, 2 * np.pi, size=n)
  phi = np.random.uniform(0, np.pi, size=n)
  v = np.stack(
    [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1
  )
  return v


def random_rotation(n):
  axis = sphere_sampling(n)
  angle = np.random.uniform(np.pi, size=[n, 1])
  return convert(axis * angle, 'axangle', 'rotmat')


def slerp(a, b, t):
  shape = list(a.shape[:-1])
  a = np.reshape(a, [-1, 4])
  b = np.reshape(b, [-1, 4])
  t = np.reshape(t, [-1, 1])

  dot = np.einsum('nd, nd -> n', a, b)
  omega = np.expand_dims(np.arccos(np.clip(dot, -1, 1), dtype=np.float32), -1)
  so = np.sin(omega, dtype=np.float32)
  so[so == 0] = np.finfo(np.float32).eps
  p = np.sin((1 - t) * omega, dtype=np.float32) / so * a + \
      np.sin(t * omega, dtype=np.float32) / so * b
  mask = np.expand_dims(np.abs(dot) >= 1, -1)
  p = np.where(mask, a, p)

  p = np.reshape(p, shape + [4])
  return p
