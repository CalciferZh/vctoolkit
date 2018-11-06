import cv2
import pickle
import time
import numpy as np
import copy
from matplotlib import pyplot as plt


def imshow_cv(img, name='OpenCV Image Show'):
  """
  Display an image using opencv.

  Parameters
  ----------
  img: Numpy array image to be shown.

  name: Name of window.

  """
  cv2.imshow(name, img)
  cv2.waitKey()


def imshow(img):
  """
  Display an image using matplotlib.

  Parameter
  ---------
  img: Numpy array image to be shown.

  """
  plt.imshow(img)
  plt.show()


def imshow_multi(imgs, nrows, ncols):
  """
  Display images as grid.

  Parameters
  ----------
  imgs: A list of images to be displayed.

  nrows: How many rows.

  ncols: How many columns.
  """
  fig = plt.figure()
  for i, img in enumerate(imgs):
    fig.add_subplot(nrows, ncols, i+1)
    plt.imshow(img)
  plt.show()


def imshow_onerow(imgs):
  """
  Display images in a list in one row.

  Parameter
  ---------
  imgs: List of images to be displayed.

  """
  imshow_multi(imgs, 1, len(imgs))


def imshow_2x2(imgs):
  """
  Display at most 4 images in a 2x2 grid.

  Parameter
  ---------
  imgs: A list of images to be displayed.

  """
  imshow_multi(imgs, 2, 2)


def imshow_3x2(imgs):
  """
  Display at most 6 images in a 3x2 grid.

  Parameter
  ---------
  imgs: A list of images to be displayed.

  """
  imshow_multi(imgs, 3, 2)


def pkl_load(path):
  """
  Load pickle data.

  Parameter
  ---------
  path: Path to pickle file.

  Return
  ------
  Data in pickle file.

  """
  with open(path, 'rb') as f:
    data = pickle.load(f)
  return data


def pkl_save(path, data):
  """
  Save data to pickle file.

  Parameters
  ----------
  path: Path to save the pickle file.

  data: Data to be serialized.

  """
  with open(path, 'wb') as f:
    pickle.dump(data, f)


def obj_save(path, vertices, faces=None):
  """
  Save 3D model into .obj files.

  Parameters
  ----------
  path: Path to save.

  vertices: Vertices of the mesh. Can be a list of vertex or an Nx3 numpy array.

  faces: Vertex indices of each face. Should be Mx3 numpy array or `None`.

  """
  with open(path, 'w') as fp:
    for v in vertices:
      fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    if faces is not None:
      for f in faces + 1:
        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


def ply_save_color_face(path, verts, faces, colors):
  """
  Save 3D model into .ply files with colored faces.

  Parameters
  ----------
  path: Path to save.

  vertices: Vertices of the mesh. Can be a list of vertex or an Nx3 numpy array.

  faces: Vertex indices of each face. Can be a list of vertex indices or an Nx3 numpy array.

  colors: Faces' colors. Can be a list of vertex indices or an Nx3 numpy array.

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
      f.write(b'3 %d %d %d %d %d %d\n' % (face[0], face[1], face[2], c[0], c[1], c[2]))


def ply_save_color_pcloud(path, pcloud, color):
  """
  Save point cloud into .ply files with color.

  Parameters
  ----------
  path: Path to save.

  pcloud: Points of the cloud. Should be an Nx3 numpy array.

  color: Faces' colors. Can be a list of RGB color or an Nx3 numpy array.

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


def one_hot_decoding(array):
  """
  One-hot decoding.

  Parameter
  ---------
  array: Array encoded in the last dimension.

  """
  return np.argmax(array, axis=-1)


def one_hot_encoding(array, n_channels):
  """
  One-hot encode a given array.

  Parameters
  ----------
  array: Array to be encoded.
  n_channels: Number of channels to be encoded.

  Return
  ------
  One-hot encoded ndarray with new channels at the last dimension.

  """
  shape = list(copy.deepcopy(array.shape))
  array = np.reshape(array, [-1])
  array = np.eye(n_channels)[array]
  shape.append(n_channels)
  array = np.reshape(array, shape)
  return array


class VideoReader:
  """
  Read frames from video.
  """
  def __init__(self, path):
    """
    Parameters
    ----------
    path: Path to the video.
    """
    self.video = cv2.VideoCapture(path)
    self.fps = self.video.get(cv2.CAP_PROP_FPS)
    self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
    self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

  def next_frame(self):
    """
    Read next frame.
    """
    if not self.video.isOpened():
      return None
    ret, frame = self.video.read()
    if ret:
      return frame
    return None

  def next_frames(self, n_frames):
    """
    Read next several frames.

    Parameter
    ---------
    n_frames: How many frames to be read.

    Return
    ------
    A list of read frames. Could be less than `n_frames` if video ends.
    """
    frames = []
    for _ in range(n_frames):
      if not self.video.isOpened():
        break
      ret, frame = self.video.read()
      if ret:
        frames.append(frame)
      else:
        break
    return frames

  def all_frames(self):
    """
    Read all (remained) frames from video.

    Return
    ------
    A list of frames.
    """
    frames = []
    while self.video.isOpened():
      ret, frame = self.video.read()
      if ret:
        frames.append(frame)
      else:
        break
    return frames

  def sequence(self, start, end):
    self.video.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(start, end):
      ret, frame = self.video.read()
      if ret:
        frames.append(frame)
      else:
        break
    return frames

  def close(self):
    """
    Release video resource.
    """
    self.video.release()


class VideoWriter:
  """
  Write frames to a video.
  """
  def __init__(self, path, width, height, fps):
    """
    Parameters
    ----------
    path: Path to the video.

    width: Width of each frame.

    height: Height of each frame.

    fps: Frame per second.
    """
    self.video = cv2.VideoWriter(
      path,
      cv2.VideoWriter_fourcc(*'XVID'),
      fps,
      (width, height)
    )

  def write_frame(self, frame):
    """
    Write single frame.

    Parameters
    ----------
    frame: Frame to be written.
    """
    self.video.write(frame)

  def close(self):
    """
    Release resource.
    """
    self.video.release()


class Timer():
  def __init__(self):
    """
    Simple timer.

    """
    self.start = None
    self.end = None
    self.interval = None

  def tic(self):
    """
    Start timing.

    """
    self.start = time.time()

  def toc(self):
    """
    End timing.

    Return
    ------
    Time since last `tic` called.

    """
    self.end = time.time()
    self.interval = self.end - self.start
    return self.interval
