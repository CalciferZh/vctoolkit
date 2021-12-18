import numpy as np
from .io import *
from tqdm import tqdm
from transforms3d.axangles import axangle2mat


def joints_to_mesh(joints, parents, color=None, thickness=0.2, save_path=None):
  """
  Produce a mesh representing the skeleton with given joint coordinates.
  Bones are represented by prisms.

  Parameters
  ----------
  joints : np.ndarray, shape [k, 3]
    Joint coordinates.
  parents : list
    Parent joint of each child joint.
  color : list
    Color for each bone. A list of list in range [0, 255].
  thickness : float, optional
    The thickness of the bone relative to length, by default 0.2

  Returns
  -------
  np.ndarray, shape [v, 3]
    Vertices of the mesh.
  np.ndarray, shape [f, 3]
    Face indices of the mesh.
  """
  n_bones = len(list(filter(lambda x: x is not None, parents)))
  faces = np.empty([n_bones * 8, 3], dtype=np.int32)
  verts = np.empty([n_bones * 6, 3], dtype=np.float32)
  face_color = []
  bone_idx = -1
  for child, parent in enumerate(parents):
    if parent is None:
      continue
    bone_idx += 1
    a = joints[parent]
    b = joints[child]
    ab = b - a
    f = a + thickness * ab

    if ab[0] == 0:
      ax = [0, 1, 0]
    else:
      ax = [-ab[1]/ab[0], 1, 0]

    fd = np.transpose(axangle2mat(ax, -np.pi/2).dot(np.transpose(ab))) \
         * thickness / 1.2
    d = fd + f
    c = np.transpose(axangle2mat(ab, -np.pi/2).dot(np.transpose(fd))) + f
    e = np.transpose(axangle2mat(ab, np.pi/2).dot(np.transpose(fd))) + f
    g = np.transpose(axangle2mat(ab, np.pi).dot(np.transpose(fd))) + f

    verts[bone_idx*6+0] = a
    verts[bone_idx*6+1] = b
    verts[bone_idx*6+2] = c
    verts[bone_idx*6+3] = d
    verts[bone_idx*6+4] = e
    verts[bone_idx*6+5] = g

    faces[bone_idx*8+0] = \
      np.flip(np.array([0, 2, 3], dtype=np.int32), axis=0) + bone_idx * 6
    faces[bone_idx*8+1] = \
      np.flip(np.array([0, 3, 4], dtype=np.int32), axis=0) + bone_idx * 6
    faces[bone_idx*8+2] = \
      np.flip(np.array([0, 4, 5], dtype=np.int32), axis=0) + bone_idx * 6
    faces[bone_idx*8+3] = \
      np.flip(np.array([0, 5, 2], dtype=np.int32), axis=0) + bone_idx * 6
    faces[bone_idx*8+4] = \
      np.flip(np.array([1, 4, 3], dtype=np.int32), axis=0) + bone_idx * 6
    faces[bone_idx*8+5] = \
      np.flip(np.array([1, 3, 2], dtype=np.int32), axis=0) + bone_idx * 6
    faces[bone_idx*8+6] = \
      np.flip(np.array([1, 5, 4], dtype=np.int32), axis=0) + bone_idx * 6
    faces[bone_idx*8+7] = \
      np.flip(np.array([1, 2, 5], dtype=np.int32), axis=0) + bone_idx * 6

    if color is not None:
      for _ in range(8):
        face_color.append(color[child])

  if color is not None:
    return verts, faces, face_color

  if save_path is not None:
    save(save_path, (verts, faces))

  return verts, faces
