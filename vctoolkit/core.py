import cv2
import pickle
import time
import numpy as np
import copy
import imageio
from matplotlib import pyplot as plt
from tqdm import tqdm
import open3d as o3d


color_lib = [
  [255, 0, 0], [230, 25, 75], [60, 180, 75], [255, 225, 25],
  [0, 130, 200], [245, 130, 48], [145, 30, 180], [70, 240, 240],
  [240, 50, 230], [210, 245, 60], [250, 190, 190], [0, 128, 128],
  [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
  [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128],
  [128, 128, 128]
]


def read_all_lines(path):
  with open(path, 'r') as f:
    lines = f.read().splitlines()
  return lines


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


def imread(path):
  return imageio.imread(path)


def imsave(path, img):
  return imageio.imsave(path, img)


def imresize(img, size):
  """
  Resize an image with cv2.INTER_LINEAR.

  Parameters
  ----------
  size: (width, height)

  """
  return cv2.resize(img, size, cv2.INTER_LINEAR)


def tensor_shape(t):
  return t.get_shape().as_list()


def print_important(s):
  print('='*80)
  print(s)
  print('='*80)


def arr_identical(a, b, print_info=True):
  if a.shape != b.shape:
    if print_info:
      print('Different shape: a: {}, b: {}'.format(a.shape, b.shape))
    return False
  else:
    return np.allclose(a, b)


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
    self.n_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)) + 1

  def next_frame(self):
    """
    Read next frame in RGB format.
    """
    if not self.video.isOpened():
      return None
    ret, frame = self.video.read()
    if not ret:
      return None
    frame = np.flip(frame, axis=-1).copy()
    return frame

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
      frame = np.flip(frame, axis=-1).copy()
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
      if not ret:
        break
      frame = np.flip(frame, axis=-1).copy()
      frames.append(frame)
    return frames

  def sequence(self, start, end):
    self.video.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(start, end):
      ret, frame = self.video.read()
      frame = np.flip(frame, axis=-1).copy()
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
      path, cv2.VideoWriter_fourcc(*'H264'), fps, (width, height)
    )

  def write_frame(self, frame):
    """
    Write single frame in RGB format.

    Parameters
    ----------
    frame: Frame to be written.
    """
    self.video.write(np.flip(frame, axis=-1).copy())

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


class LowPassFilter:
  def __init__(self):
    self.prev_raw_value = None
    self.prev_filtered_value = None

  def process(self, value, alpha):
    if self.prev_raw_value is None:
      s = value
    else:
      s = alpha * value + (1.0 - alpha) * self.prev_filtered_value
    self.prev_raw_value = value
    self.prev_filtered_value = s
    return s


class OneEuroFilter:
  def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0, freq=30):
    self.freq = freq
    self.mincutoff = mincutoff
    self.beta = beta
    self.dcutoff = dcutoff
    self.x_filter = LowPassFilter()
    self.dx_filter = LowPassFilter()

  def compute_alpha(self, cutoff):
    te = 1.0 / self.freq
    tau = 1.0 / (2 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / te)

  def process(self, x):
    prev_x = self.x_filter.prev_raw_value
    dx = 0.0 if prev_x is None else (x - prev_x) * self.freq
    edx = self.dx_filter.process(dx, self.compute_alpha(self.dcutoff))
    cutoff = self.mincutoff + self.beta * np.abs(edx)
    return self.x_filter.process(x, self.compute_alpha(cutoff))


def render_sequence_3d(verts, faces, width, height, video_path, fps=30,
                       visible=False):
  writer = VideoWriter(video_path, width, height, fps)
  mesh = o3d.geometry.TriangleMesh()
  mesh.triangles = o3d.utility.Vector3iVector(faces)
  mesh.vertices = o3d.utility.Vector3dVector(verts[0])

  vis = o3d.visualization.Visualizer()
  vis.create_window(width=width, height=height, visible=visible)
  vis.add_geometry(mesh)

  for v in tqdm(verts, ascii=True):
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.compute_vertex_normals()
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()
    frame = (np.asarray(vis.capture_screen_float_buffer()) * 255).astype(np.uint8)
    writer.write_frame(frame)

  writer.close()
