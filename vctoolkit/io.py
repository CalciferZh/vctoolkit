import pickle
import imageio
import cv2
import numpy as np
import uuid
from .misc import imresize


def read_all_lines(path):
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
      fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
    if faces is not None:
      for f in faces + 1:
        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


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


class VideoReader:
  """
  Read frames from video.
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
  """
  def __init__(self, path, width, height, fps):
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
    """
    self.video = cv2.VideoWriter(
      path, cv2.VideoWriter_fourcc(*'H264'), fps, (width, height)
    )

  def write_frame(self, frame):
    """
    Write one frame.

    Parameters
    ----------
    frame : np.ndarray
      Frame to write.
    """
    self.video.write(np.flip(frame, axis=-1).copy())

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


def save_video_frames(size, video_path, save_prefix=None, fps=60):
  """
  Read video, display frame by frame, and save selected frames.

  w: previous frame
  s: next frame
  a: revert
  d: forward
  q: quit
  space: save this frame

  Parameters
  ----------
  size : tuple
    Screen size, (width, height)
  video_path : str
    Path to the video.
  save_prefix : str, optional
    Path prefix to save the frames, if None, will be the same as video_path,
    by default None
  fps : int, optional
    Display framerate, by default 60
  """
  import pygame

  if save_prefix is None:
    save_prefix = video_path

  pygame.init()
  display = pygame.display.set_mode(size)
  pygame.display.set_caption('loading')

  reader = VideoReader(video_path)
  frames = reader.all_frames()
  reader.close()

  idx = 0
  done = False
  clock = pygame.time.Clock()
  while not done:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        done = True
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_w:
          idx -= 1
        elif event.key == pygame.K_s:
          idx += 1
        elif event.key == pygame.K_SPACE:
          imsave(video_path + '.frame%d' % idx + '.png')
        elif event.key == pygame.K_q:
          done = True
    pressed = pygame.key.get_pressed()
    if pressed[pygame.K_a]:
      idx -= 1
    if pressed[pygame.K_d]:
      idx += 1

    idx = min(max(idx, 0), len(frames) - 1)
    pygame.display.set_caption('%s %d/%d' % video_path, idx, len(frames))
    display.blit(
      pygame.surfarray.make_surface(
        imresize(frames[idx], size).transpose((1, 0, 2))
      ), (0, 0)
    )
    pygame.display.update()

    clock.tick(fps)

  pkl_save(save_path, selected)
