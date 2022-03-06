import pickle
import numpy as np

from . import math_np


class LBSMesh():
  def __init__(self, model_path, skeleton, dtype=np.float32):
    with open(model_path, 'rb') as f:
      data = pickle.load(f, encoding='latin1')

    self.mesh = data['v_template'].astype(dtype)
    self.n_verts = self.mesh.shape[0]

    self.keypoints_mean = np.empty([skeleton.n_keypoints, 3], dtype)
    self.keypoints_mean[:data['J'].shape[0]] = data['J']
    for k, v in skeleton.extended_keypoints.items():
      self.keypoints_mean[k] = self.mesh[v]

    self.j_regressor = np.zeros([skeleton.n_keypoints, self.n_verts], dtype)
    self.j_regressor[:data['J_regressor'].shape[0]] = \
      data['J_regressor'].toarray()
    for k, v in skeleton.extended_keypoints.items():
      self.j_regressor[k, v] = 1
    self.keypoints_std = np.einsum(
      'vdc, jv -> vjd', np.array(data['shapedirs'], dtype), self.j_regressor
    )

    self.parents = skeleton.parents
    self.children = [[] for _ in skeleton.parents]
    for c, p in enumerate(self.parents):
      if p is not None:
        self.children[p].append(c)

    # translate skinning weight: we use child joint
    self.skinning_weights = np.zeros(
      [data['weights'].shape[0], skeleton.n_keypoints], dtype=np.float32
    )
    for c, p in enumerate(self.parents):
      if p is not None:
        self.skinning_weights[:, c] = \
          data['weights'][:, p] / len(self.children[p])

    self.faces = data['f']
    self.shape_std = np.array(data['shapedirs'], dtype)
    self.ones = np.ones([self.n_verts, 1], dtype)
    self.skeleton = skeleton
    self.shape_dim = self.shape_std[-1]
    self.n_faces = self.faces.shape[0]
    self.dtype = dtype

  def pose_parent_to_children(self, pose):
    # convert pose from children style to parent style
    outputs = [np.zeros(3) for _ in range(self.skeleton.n_keypoints)]
    for c, p in enumerate(self.parents):
      if p is not None:
        outputs[c] = pose[p]
    return np.stack(outputs)

  def set_params(self, pose=None, shape=None, format='rotmat', relative=False,
                 reference='child', use_j_regressor=False):
    verts = self.mesh.copy()
    if shape is not None:
      verts = verts + np.einsum('c, vdc -> vd', shape, self.shape_std)

    keypoints = np.einsum('vd, jv -> jd', verts, self.j_regressor)
    if pose is None:
      return verts, keypoints

    if reference == 'parent':
      pose = self.pose_parent_to_children(pose)

    if format != 'rotmat':
      pose = math_np.convert(pose, format, 'rotmat')

    if relative:
      pose = math_np.rotmat_rel_to_abs(pose, self.parents)

    bones = math_np.keypoints_to_bones(keypoints, self.parents)
    posed_keypoints, _ = \
      math_np.forward_kinematics(bones, pose, self.parents)
    j_mat = posed_keypoints - np.einsum('jhw, jw -> jh', pose, keypoints)
    g_mat = np.concatenate([pose, np.expand_dims(j_mat, -1)], -1)
    verts = np.concatenate([verts, self.ones], 1)
    posed_verts = np.einsum(
      'vj, jvd -> vd',
      self.skinning_weights, np.einsum('jhw, vw -> jvh', g_mat, verts)
    )
    if use_j_regressor:
      posed_keypoints = np.dot(self.j_regressor, posed_verts)

    return posed_keypoints, posed_verts
