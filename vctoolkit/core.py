import cv2
import pickle
import time
import numpy as np
import copy
import imageio
from matplotlib import pyplot as plt
from tqdm import tqdm
import open3d as o3d
from transforms3d.axangles import axangle2mat


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


class Timer:
  def __init__(self):
    self.last_tic = None
    self.memory = {}

  def tic(self, key=None):
    curr_time = time.time()
    interval = 0.0
    if self.last_tic is not None:
      interval = curr_time - self.last_tic
      if key is not None:
        self.memory[key] = interval
    self.last_tic = curr_time
    return interval


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
    """
    Parameters
    ----------
    mincutoff : float, optional
      Decrease mincutoff to decrease slow speed jittering, by default 1.0
    beta : float, optional
      Increase beta to decrease speed lag, by default 0.0
    dcutoff : float, optional
        [description], by default 1.0
    freq : int, optional
        [description], by default 30
    """
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
                       visible=False, need_norm=True):
  if need_norm:
    if type(verts) == list:
      verts = np.stack(verts, 0)

    scale = np.max(np.max(verts, axis=(0, 1)) - np.min(verts, axis=(0, 1)))
    mean = np.mean(verts)
    verts = (verts - mean) / scale

  cam_offset = 1.2

  writer = VideoWriter(video_path, width, height, fps)

  mesh = o3d.geometry.TriangleMesh()
  mesh.triangles = o3d.utility.Vector3iVector(faces)
  mesh.vertices = o3d.utility.Vector3dVector(verts[0])

  vis = o3d.visualization.Visualizer()
  vis.create_window(width=width, height=height, visible=visible)
  vis.add_geometry(mesh)
  view_control = vis.get_view_control()
  cam_params = view_control.convert_to_pinhole_camera_parameters()
  cam_params.extrinsic = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, cam_offset],
    [0, 0, 0, 1],
  ])
  view_control.convert_from_pinhole_camera_parameters(cam_params)

  for v in tqdm(verts, ascii=True):
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.compute_vertex_normals()
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()
    frame = (np.asarray(vis.capture_screen_float_buffer()) * 255).astype(np.uint8)
    writer.write_frame(frame)

  writer.close()


def render_gaussian_hmap(centers, shape, sigma=None):
  """
  Render gaussian heat maps from given centers.

  Parameters
  ----------
  centers : np.ndarray, shape [N, 2]
    Gussian centers, (row, column).
  shape : tuple
    Heatmap shape, (height, width)
  sigma : float, optional
    sigma, if None, would be height / 40

  Returns
  -------
  np.ndarray, float32, shape [height, width, N]
    Rendered heat maps
  """
  if sigma is None:
    sigma = shape[0] / 40
  x = [i for i in range(shape[1])]
  y = [i for i in range(shape[0])]
  xx, yy = np.meshgrid(x, y)
  xx = np.reshape(xx.astype(np.float32), [shape[0], shape[1], 1])
  yy = np.reshape(yy.astype(np.float32), [shape[0], shape[1], 1])
  x = np.reshape(centers[:,1], [1, 1, -1])
  y = np.reshape(centers[:,0], [1, 1, -1])
  distance = np.square(xx - x) + np.square(yy - y)
  hmap = np.exp(-distance / (2 * sigma**2 )) / np.sqrt(2 * np.pi * sigma**2)
  hmap /= (
    np.max(hmap, axis=(0, 1), keepdims=True) + np.finfo(np.float32).eps
  )
  return hmap


def hmap_to_uv(hmap):
  """
  Compute uv coordinates of heat map maxima in each layer.

  Parameters
  ----------
  hmap : np.ndarray, shape [h, w, k]
    Heat maps.

  Returns
  -------
  np.ndarray, shape [k, 2]
    UV coordinates in the order of (row, column).
  """
  shape = hmap.shape
  hmap = np.reshape(hmap, [-1, shape[-1]])
  v, h = np.unravel_index(np.argmax(hmap, 0), shape[:-1])
  coord = np.stack([v, h], 1)
  return coord


def render_bones_from_uv(uv, canvas, parents, thickness=None):
  """
  Render bones from joint uv coordinates.

  Parameters
  ----------
  uv : np.ndarray, shape [k, 2]
    UV coordinates of joints.
  canvas : np.ndarray, dtype uint8
    Canvas to draw on.
  parents : list
    The parent joint for each joint. Root joint's parent should be None.
  thickness : int, optional
    Thickness of the line, by default None

  Returns
  -------
  np.ndarray
    The canvas after rendering.
  """
  if canvas.dtype != np.uint8:
    print('canvas must be uint8 type')
    exit(0)
  if thickness is None:
    thickness = max(canvas.shape[0] // 128, 1)
  for c, p in enumerate(parents):
    if p is None:
      continue
    color = color_lib[p]
    start = (int(uv[p][1]), int(uv[p][0]))
    end = (int(uv[c][1]), int(uv[c][0]))

    anyzero = lambda x: x[0] * x[1] == 0
    if anyzero(start) or anyzero(end):
      continue

    cv2.line(canvas, start, end, color, thickness)
  return canvas


def render_bones_from_hmap(hmap, canvas, parents, thickness=None):
  """
  Render bones from heat maps.

  Parameters
  ----------
  hmap : np.ndarray, shape [h, w, k]
    Heat maps.
  canvas : np.ndarray, shape [h, w, 3], dtype uint8
    Canvas to render.
  parents : list
    Parent joint of each child joint.
  thickness : int, optional
    Thickness of each bone, by default None

  Returns
  -------
  np.ndarray
    Canvas after rendering.
  """
  coords = hmap_to_uv(hmap)
  bones = render_bones_from_uv(coords, canvas, parents, thickness)
  return bones


def joints_to_mesh_prism(joints, parents, thickness=0.2):
  """
  Produce a mesh representing the skeleton with given joint coordinates.

  Parameters
  ----------
  joints : np.ndarray, shape [k, 3]
    Joint coordinates.
  parents : list
    Parent joint of each child joint.
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
  for c, p in enumerate(parents):
    if p is None:
      continue
    a = joints[p]
    b = joints[c]
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

    verts[c*6+0] = a
    verts[c*6+1] = b
    verts[c*6+2] = c
    verts[c*6+3] = d
    verts[c*6+4] = e
    verts[c*6+5] = g

    faces[c*8+0] = np.flip(np.array([0, 2, 3], dtype=np.int32), axis=0) + c * 6
    faces[c*8+1] = np.flip(np.array([0, 3, 4], dtype=np.int32), axis=0) + c * 6
    faces[c*8+2] = np.flip(np.array([0, 4, 5], dtype=np.int32), axis=0) + c * 6
    faces[c*8+3] = np.flip(np.array([0, 5, 2], dtype=np.int32), axis=0) + c * 6
    faces[c*8+4] = np.flip(np.array([1, 4, 3], dtype=np.int32), axis=0) + c * 6
    faces[c*8+5] = np.flip(np.array([1, 3, 2], dtype=np.int32), axis=0) + c * 6
    faces[c*8+6] = np.flip(np.array([1, 5, 4], dtype=np.int32), axis=0) + c * 6
    faces[c*8+7] = np.flip(np.array([1, 2, 5], dtype=np.int32), axis=0) + c * 6

  return verts, faces


def joints_to_mesh_cylinder(joints, parents, thickness=0.2):
  faces = []
  verts = []
  v_cnt = 0

  for c, p in enumerate(parents):
    if p is None:
      continue
    parent = joints[p]
    child = joints[c]
    delta = parent - child
    delta_len = np.linalg.norm(delta)
    delta_unit = delta / delta_len

    radius = delta_len * thickness
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
    s_verts = np.asarray(sphere.vertices)
    s_faces = np.asarray(sphere.triangles)

    # bone cylinder

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
      radius / 2, height=delta_len, split=1
    )
    c_verts = np.asarray(cylinder.vertices) + np.array([0, 0, delta_len / 2])
    c_faces = np.asarray(cylinder.triangles)
    c_delta = np.array([0, 0, 1])

    v_dot = np.dot(c_delta, delta_unit)
    v_cross = np.linalg.norm(np.cross(c_delta, delta_unit))
    rot_mat = np.array([[v_dot, -v_cross, 0], [v_cross, v_dot, 0], [0, 0, 1]])

    v = np.cross(c_delta, delta_unit)
    s = np.linalg.norm(v)
    c = np.dot(c_delta, delta_unit)
    vx = '{} {} {}; {} {} {}; {} {} {}'.format(
      0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0
    )
    vx = np.matrix(vx)
    rot_mat = np.eye(3) + vx + np.matmul(vx, vx) * ((1 - c) / (s ** 2))

    v = np.matmul(rot_mat, c_verts.T).T + child
    verts.append(v)
    faces.append(c_faces + v_cnt)
    v_cnt += v.shape[0]

    # joint sphere

    v = s_verts + child
    verts.append(v)
    faces.append(s_faces + v_cnt)
    v_cnt += v.shape[0]

    v = s_verts + parent
    verts.append(v)
    faces.append(s_faces + v_cnt)
    v_cnt += v.shape[0]

  faces = np.concatenate(faces, 0)
  verts = np.concatenate(verts, 0)
  return verts, faces


def joints_to_mesh(joints, parents, style='prism', thickness=0.2):
  if style == 'prism':
    return joints_to_mesh_prism(joints, parents, thickness)
  elif style == 'cylinder':
    return joints_to_mesh_cylinder(joints, parents, thickness)
  else:
    raise RuntimeError('Invalid style: ' % style)


def vis_mesh(verts, faces):
  mesh = o3d.geometry.TriangleMesh()
  mesh.triangles = o3d.utility.Vector3iVector(faces)
  mesh.vertices = o3d.utility.Vector3dVector(verts)
  mesh.compute_vertex_normals()

  viewer = o3d.visualization.Visualizer()
  viewer.create_window(width=1080, height=1080, visible=True)
  viewer.add_geometry(mesh)

  render_option = viewer.get_render_option()
  render_option.load_from_json('./vis_render_option.json')
  viewer.update_renderer()

  viewer.run()


def save_video_frames(size, path, name, fps=30):
  save_path = './selected_frames.pkl'
  if os.path.isfile(save_path):
    selected = pkl_load(save_path)
  else:
    selected = {}

  pygame.init()
  display = pygame.display.set_mode(size)
  pygame.display.set_caption('idle')

  pygame.display.set_caption('loading')
  reader = VideoReader(path)
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
          selected[name] = selected.get(name, []) + [idx]
          imsave('%s_%d.png' % (name, idx), frames[idx])
        elif event.key == pygame.K_q:
          done = True
    pressed = pygame.key.get_pressed()
    if pressed[pygame.K_a]:
      idx -= 1
    if pressed[pygame.K_d]:
      idx += 1

    idx = min(max(idx, 0), len(frames) - 1)
    pygame.display.set_caption('%s %d/%d' % (name, idx, len(frames)))
    display.blit(
      pygame.surfarray.make_surface(frames[idx].transpose((1, 0, 2))), (0, 0)
    )
    pygame.display.update()

    clock.tick(fps)

  pkl_save(save_path, selected)
