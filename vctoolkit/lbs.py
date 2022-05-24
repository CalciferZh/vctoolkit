import pickle
import numpy as np

from . import math_np


class LBSMesh():
  def __init__(self, model_path, skeleton, dtype=np.float32, fix_shape_bug=False):
    with open(model_path, 'rb') as f:
      data = pickle.load(f, encoding='latin1')

    if fix_shape_bug:
      data['shapedirs'] = np.array(data['shapedirs'])
      data['shapedirs'][:, 0, :] *= -1

    self.mesh = data['v_template'].astype(dtype)
    self.n_verts = self.mesh.shape[0]

    self.keypoints_mean = np.empty([skeleton.n_keypoints, 3], dtype)
    self.keypoints_mean[:data['J'].shape[0]] = data['J']
    for k, v in skeleton.extended_keypoints.items():
      self.keypoints_mean[k] = self.mesh[v]

    self.j_regressor = np.zeros([skeleton.n_keypoints, self.n_verts], dtype)

    # some early version uses scipy sparse array
    if type(data['J_regressor']) != np.ndarray:
      data['J_regressor'] = data['J_regressor'].toarray()

    self.j_regressor[:data['J_regressor'].shape[0]] = data['J_regressor']
    for k, v in skeleton.extended_keypoints.items():
      self.j_regressor[k] = 0
      self.j_regressor[k, v] = 1
    self.keypoints_std = np.einsum(
      'vdc, jv -> cjd', np.array(data['shapedirs'], dtype), self.j_regressor
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
    self.ones = np.ones([1, self.n_verts, 1], dtype)
    self.skeleton = skeleton
    self.shape_dim = self.shape_std[-1]
    self.n_faces = self.faces.shape[0]
    self.dtype = dtype

  def pose_parent_to_children(self, pose, batch=False):
    if not batch:
      pose = np.expand_dims(pose, 0)

    # convert pose from children style to parent style
    outputs = [
      np.zeros([pose.shape[0]] + list(pose[0][0].shape), self.dtype) \
        for _ in range(self.skeleton.n_keypoints)
    ]
    for c, p in enumerate(self.parents):
      if p is not None and p < pose.shape[1]:
        outputs[c] = pose[:, p]

    outputs = np.stack(outputs, 1)

    if not batch:
      outputs = outputs[0]

    return outputs

  def set_params(self, pose=None, shape=None, format='rotmat', relative=False,
                 reference='child', use_j_regressor=False, batch=False,
                 return_details=False, return_mesh=False):
    if pose is not None:
      pose = pose.astype(self.dtype)
    if shape is not None:
      shape = shape.astype(self.dtype)

    if not batch:
      if pose is not None:
        pose = np.expand_dims(pose, 0)
      if shape is not None:
        shape = np.expand_dims(shape, 0)

    verts = np.expand_dims(self.mesh.copy(), 0)

    keypoints = np.einsum('nvd, jv -> njd', verts, self.j_regressor)

    if shape is not None:
      verts = verts + np.einsum('nc, vdc -> nvd', shape.astype(self.dtype), self.shape_std)

    keypoints = np.einsum('nvd, jv -> njd', verts, self.j_regressor)

    if pose is None:
      if not batch:
        verts = verts[0]
        keypoints = keypoints[0]
      return keypoints, verts, None

    if reference == 'parent':
      pose = self.pose_parent_to_children(pose, batch=True)

    if format != 'rotmat':
      pose = math_np.convert(pose, format, 'rotmat')

    if relative:
      pose = math_np.rotmat_rel_to_abs(pose, self.parents, batch=True)

    bones = math_np.keypoints_to_bones(keypoints, self.parents, batch=True)
    posed_keypoints, _ = \
      math_np.forward_kinematics(bones, pose, self.parents, batch=True)

    if return_mesh:
      j_mat = posed_keypoints - np.einsum('njhw, njw -> njh', pose, keypoints)
      g_mat = np.concatenate([pose, np.expand_dims(j_mat, -1)], -1)
      verts = np.concatenate([verts, np.tile(self.ones, [pose.shape[0], 1, 1])], -1)
      posed_verts = np.einsum(
        'vj, njvd -> nvd',
        self.skinning_weights, np.einsum('njhw, nvw -> njvh', g_mat, verts)
      )

    if use_j_regressor:
      posed_keypoints = \
        np.einsum('jv, nvd -> njd', self.j_regressor, posed_verts)

    if not batch:
      posed_keypoints = posed_keypoints[0]
      pose = pose[0]
      if return_mesh:
        posed_verts = posed_verts[0]

    if return_details:
      pack = {
        'posed_keypoints': posed_keypoints,
        'abs_rotmat': pose,
        'ref_bones': bones
      }
      if return_mesh:
        pack['posed_verts'] = posed_verts
      return pack

    if return_mesh:
      return posed_keypoints, posed_verts
    else:
      return posed_keypoints