import json
import os
import pickle

import cv2
import h5py
import imageio
import numpy as np
import scipy.io
import yaml


def load(path):
  ext = os.path.splitext(path)[1][1:]
  if ext == 'json':
    return load_json(path)
  elif ext == 'mat':
    return load_mat(path)
  elif ext == 'h5':
    return load_hdf5(path)
  elif ext == 'txt':
    return load_txt(path)
  elif ext == 'pkl':
    return load_pkl(path)
  elif ext in ['png', 'jpg', 'jpeg']:
    return np.array(load_img(path))
  elif ext == 'obj':
    return load_obj(path)
  elif ext == 'npy' or ext == 'npz':
    return np.load(path)
  elif ext == 'yaml':
    return load_yaml(path)
  else:
    raise NotImplementedError('Unsupported file extension: ' + path)


def save(path, data):
  ext = os.path.splitext(path)[1][1:]
  if ext == 'json':
    return save_json(path, data)
  elif ext == 'h5':
    return save_hdf5(path, data)
  elif ext == 'txt':
    return save_txt(path, data)
  elif ext == 'pkl':
    return save_pkl(path, data)
  elif ext in ['png', 'jpg', 'jpeg']:
    return save_img(path, data)
  elif ext == 'obj':
    return save_obj(path, *data)
  elif ext == 'npy':
    return np.save(path, data)
  elif ext == 'npz':
    return np.savez(path, data)
  elif ext == 'yaml':
    return save_yaml(path, data)
  else:
    raise NotImplementedError('Unsupported file extension: ' + path)


def load_yaml(path):
  with open(path) as f:
    data = yaml.load(f)
  return data


def save_yaml(path, data):
  with open(path, 'w') as f:
    documents = yaml.dump(data, f)


def load_json(path):
  """
  Load json data from file.

  Parameters
  ----------
  path : str
    Path to the json file.

  Returns
  -------
  dict or list
    Json data.
  """
  with open(path) as f:
    data = json.load(f)
  return data


def save_json(path, data):
  """
  Save dict data into json file.

  Parameters
  ----------
  path : str
    Path to save.
  data : dict or list
    Data to save.
  """
  with open(path, 'w') as f:
    f.write(json.dumps(data, indent=2))


def load_mat(path):
  """
  Load matlab .mat data.

  Parameters
  ----------
  path : str
    Path to data.

  Returns
  -------
  obj
    Data.
  """
  return scipy.io.loadmat(path)


def save_hdf5(path, data):
  """
  Save data into hdf5 format. If any data is string in a numpy array, make sure
  set the dtype to 'S'.

  Parameters
  ----------
  path : str
    Path to save the data.
  data : dict
    Keys and data.
  """
  f = h5py.File(path, 'w')
  for k, v in data.items():
    f.create_dataset(k, v.shape, data=v)
  f.close()


def load_hdf5(path):
  """
  Load data from hdf5 file.

  Parameters
  ----------
  path : str
    Path to the file.

  Returns
  -------
  dict
    Data.
  """
  f = h5py.File(path, 'r')
  data = {}
  for k in f.keys():
    data[k] = np.array(f[k])
  return data


def save_txt(path, s):
  with open(path, 'w') as f:
    f.write(s)


def load_txt(path):
  """
  Read all lines from a text file.

  Parameters
  ----------
  path : str
    Path to the text file.

  Returns
  -------
  list
    A list of lines.
  """
  with open(path, 'r') as f:
    lines = f.read().splitlines()
  return lines


def pkl_load(path):
  """
  Load pickle data.

  Parameters
  ----------
  path : str
    Path to the file.

  Returns
  -------
  object
    Loaded data.
  """

  with open(path, 'rb') as f:
    data = pickle.load(f)
  return data


load_pkl = pkl_load


def pkl_save(path, data):
  """
  Save pickle data.

  Parameters
  ----------
  path : str
    Path to save file.
  data : object
    Data to save.
  """
  with open(path, 'wb') as f:
    pickle.dump(data, f)


save_pkl = pkl_save


def obj_save(path, vertices, faces=None):
  """
  Save .obj mesh file.

  Parameters
  ----------
  path : str
    Path to save the file.
  vertices : np.ndarray
    Mesh vertices.
  faces : np.ndarray, optional
    Mesh faces, by default None
  """
  with open(path, 'w') as fp:
    for v in vertices:
      v = np.ravel(v)
      fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    if faces is not None:
      for f in faces + 1:
        f = np.ravel(f)
        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


save_obj = obj_save


def imread(path):
  """
  Read image.

  Parameters
  ----------
  path : str
    Path to read the image.

  Returns
  -------
  np.ndarray
    Read image.
  """
  return imageio.imread(path)


load_img = imread


def imsave(path, img):
  """
  Save image.

  Parameters
  ----------
  path : str
    Path to load image.
  img : np.ndarray
    Image to save.
  """
  imageio.imsave(path, img)


save_img = imsave


class VideoReader:
  """
  Read frames from video.

  Call `next_frame` to read next frame.
  Call `all_frames` to read all remaining frames.
  Call `close` to release resource.

  """
  def __init__(self, path):
    """
    Parameters
    ----------
    path : str
      Path to the video file.
    """
    self.video = cv2.VideoCapture(path)
    self.fps = self.video.get(cv2.CAP_PROP_FPS)
    self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
    self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    self.n_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)) + 1
    self.frame_idx = 0

  def next_frame(self):
    """
    Read next frame. Return None if no frame left.

    Returns
    -------
    np.ndarray
      Image in RGB format.
    """
    if not self.video.isOpened():
      return None
    ret, frame = self.video.read()
    self.frame_idx += 1
    if not ret:
      return None
    frame = np.flip(frame, axis=-1).copy()
    return frame

  def all_frames(self):
    """
    Read all remaining frames.

    Returns
    -------
    list
      A list of frames.
    """
    frames = []
    while self.video.isOpened():
      ret, frame = self.video.read()
      self.frame_idx += 1
      if not ret:
        break
      frame = np.flip(frame, axis=-1).copy()
      frames.append(frame)
    return frames

  def close(self):
    """
    Release video resource.
    """
    self.video.release()


class VideoWriter:
  """
  Write frames to a video.

  Call `write_frame` to write a single frame.
  Call `close` to release resource.

  """
  def __init__(self, path, width, height, fps, codec='H264'):
    """
    Parameters
    ----------
    path : str
      Path to the video.
    width : int
      Frame width.
    height : int
      Frame height.
    fps : int
      Video frame rate.
    codec : str, optional
      Video codec, by default H264.
    """
    self.fps = fps
    self.width = width
    self.height = height
    self.video = cv2.VideoWriter(
      path, cv2.VideoWriter_fourcc(*codec), fps, (width, height)
    )
    self.frame_idx = 0

  def write_frame(self, frame):
    """
    Write one frame.

    Parameters
    ----------
    frame : np.ndarray
      Frame to write.
    """
    self.video.write(np.flip(frame, axis=-1).copy())
    self.frame_idx += 1

  def close(self):
    """
    Release resource.
    """
    self.video.release()


def ply_save_color_face(path, verts, faces, colors):
  """
  Save mesh into a .ply file with colored faces.

  Parameters
  ----------
  path : str
    Path to save the file.
  verts : np.ndarray, shape [v, 3]
    Mesh vertices.
  faces : np.ndarray, shape [f, 3]
    Mesh faces.
  colors : np.ndarray, shape [f, 3], dtype uint8
    RGB color for each face.
  """
  num_verts = verts.shape[0]
  num_faces = faces.shape[0]
  with open(path, 'wb') as f:
    f.write(b'ply\n')
    f.write(b'format ascii 1.0\n')
    f.write(b'element vertex %d\n' % num_verts)
    f.write(b'property float32 x\n')
    f.write(b'property float32 y\n')
    f.write(b'property float32 z\n')
    f.write(b'element face %d\n' % num_faces)
    f.write(b'property list uint8 int32 vertex_index\n')
    f.write(b'property uchar red\n')
    f.write(b'property uchar green\n')
    f.write(b'property uchar blue\n')
    f.write(b'end_header\n')
    for i in range(num_verts):
      v = verts[i]
      f.write(b'%f %f %f\n' % (v[0], v[1], v[2]))
    for i in range(num_faces):
      face = faces[i]
      c = colors[i]
      f.write(
        b'3 %d %d %d %d %d %d\n' % (face[0], face[1], face[2], c[0], c[1], c[2])
      )


save_ply_color_face = ply_save_color_face


def ply_save_color_pcloud(path, pcloud, color):
  """
  Save point cloud into a .ply file with per-point color.

  Parameters
  ----------
  path : str
    Path to save the file.
  pcloud : np.ndarray, shape [n, 3]
    Point cloud to save.
  color : np.ndarray, shape [n, 3], dtype uint8
    Color for each point.
  """
  num_points = pcloud.shape[0]
  with open(path, 'wb') as f:
    f.write(b'ply\n')
    f.write(b'format ascii 1.0\n')
    f.write(b'element vertex %d\n' % num_points)
    f.write(b'property float32 x\n')
    f.write(b'property float32 y\n')
    f.write(b'property float32 z\n')
    f.write(b'property uchar red\n')
    f.write(b'property uchar green\n')
    f.write(b'property uchar blue\n')
    f.write(b'end_header\n')
    for i in range(num_points):
      v = pcloud[i]
      c = color[i]
      f.write(b'%f %f %f %d %d %d\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))


save_ply_color_pcloud = ply_save_color_pcloud


def load_obj(file_path):
  """
  Read the most naive .obj file. Only supports `v` and `f`.

  Parameters
  ----------
  file_path : str
    Path to the file.

  Returns
  -------
  [N, 3]
    Vertices.
  [N, 3]
    Faces (Vertex index starts from 0).
  """
  lines = load_txt(file_path)
  verts = []
  faces = []
  for l in lines:
    if l[0] == 'v':
      verts.append(np.array([float(v) for v in l.split()[1:]]))
    elif l[0] == 'f':
      faces.append(np.array([int(v) - 1 for v in l.split()[1:]]))
  verts = np.array(verts)
  faces = np.array(faces)
  return verts, faces
