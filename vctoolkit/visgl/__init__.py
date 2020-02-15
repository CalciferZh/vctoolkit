import pygame
import numpy as np
import datetime

from transforms3d.axangles import axangle2mat
from OpenGL.GL import *
from OpenGL.GLU import *
from ..io import imsave, VideoWriter


class TriMeshViewer:
  """
  Render and visualize a triangle mesh with animation. See `TriMeshViewer.run`.

  num-pad 5: reset view.
  space: play / pause.
  d: revert.
  f: forward.
  p: save screenshot.

  """
  def __init__(self, size=(512, 512)):
    self.rot_mat = np.eye(3)
    self.trans_v = np.zeros(3)
    self.speed_rot = np.pi / 45
    self.speed_trans = 0.02
    self.done = False
    self.playing = False
    self.window_size = size
    self.cam_offset = 1.5
    self.verts = []
    self.faces = np.empty([3, 3]) # just whatever
    self.n_verts = None
    self.n_faces = None
    self.frame_idx = None
    self.clock = pygame.time.Clock()

    self.dup_faces = None
    self.dup_normals = None
    self.verts_mapping = None
    self.normal_mapping = None

    self.n_frames = -1
    self.setup_opengl()

  def get_normal(self, faces):
    '''Given faces, return per face unit normal vector.'''
    p1x, p1y, p1z = faces[:, 0, 0], faces[:, 0, 1], faces[:, 0, 2]
    p2x, p2y, p2z = faces[:, 1, 0], faces[:, 1, 1], faces[:, 1, 2]
    p3x, p3y, p3z = faces[:, 2, 0], faces[:, 2, 1], faces[:, 2, 2]
    x_stick = (p2y - p1y) * (p3z - p1z) - (p2z - p1z) * (p3y - p1y)
    y_stick = (p2z - p1z) * (p3x - p1x) - (p2x - p1x) * (p3z - p1z)
    z_stick = (p2x - p1x) * (p3y - p1y) - (p2y - p1y) * (p3x - p1x)
    normals = np.stack((x_stick, y_stick, z_stick), axis=1)
    normals /=np.linalg.norm(normals, axis=1, keepdims=True)
    return normals

  def setup_dup(self):
    """
    For vertices normal computation.
    Since OpenGL only supports per vertex normal, here we duplicate the vertices
    so that each face has 3 same-normal-vertices belong to itself only.
    """
    self.dup_faces = np.empty([self.n_faces, 3], np.int32)
    self.verts_mapping = np.empty(self.n_faces * 3, np.int32)
    self.normal_mapping = np.empty(self.n_faces * 3, np.int32)

    for i in range(self.n_faces):
      for j in range(3):
        self.dup_faces[i, j] = i * 3 + j
        self.verts_mapping[i * 3 + j] = self.faces[i][j]
        self.normal_mapping[i * 3 + j] = i

  def setup_opengl(self):
    pygame.init()
    self.screen = \
      pygame.display.set_mode(self.window_size, pygame.DOUBLEBUF|pygame.OPENGL)
    pygame.display.set_caption('Frame %d / %d' % (0, self.n_frames))
    self.clock = pygame.time.Clock()

    glClearColor(1.0, 1.0, 1.0, 0)
    glShadeModel(GL_FLAT)

    glMaterialfv(GL_FRONT, GL_AMBIENT, np.array([.3]*3, dtype=np.float32))
    glMaterialfv(GL_FRONT, GL_DIFFUSE, np.array([1.0]*3, dtype=np.float32))
    glMaterialfv(GL_FRONT, GL_SPECULAR, np.array([.0]*3, dtype=np.float32))
    glMaterialf(GL_FRONT, GL_SHININESS,.4 * 128.0)

    glLightfv(GL_LIGHT0,GL_POSITION,np.array([0.0,-1.0,0.0,0.0],dtype=np.float32))
    glLightfv(GL_LIGHT0,GL_SPECULAR,np.array([0.0,0.0,0.0,1],dtype=np.float32))
    glLightfv(GL_LIGHT0,GL_DIFFUSE,np.array([1.0,1.0,1.0,1],dtype=np.float32))
    glLightfv(GL_LIGHT0,GL_AMBIENT,np.array([0.3,0.3,0.3,1],dtype=np.float32))
    glEnable(GL_LIGHT0)

    glLightfv(GL_LIGHT1,GL_POSITION,np.array([0.0,0.0,-1.0,0.0],dtype=np.float32))
    glLightfv(GL_LIGHT1,GL_SPECULAR,np.array([0.0,0.0,0.0,1],dtype=np.float32))
    glLightfv(GL_LIGHT1,GL_DIFFUSE,np.array([1.0,1.0,1.0,1],dtype=np.float32))
    glLightfv(GL_LIGHT1,GL_AMBIENT,np.array([0.3,0.3,0.3,1],dtype=np.float32))
    glEnable(GL_LIGHT1)

    glLightfv(GL_LIGHT2,GL_POSITION,np.array([-1.0,0.0,0.0,0.0],dtype=np.float32))
    glLightfv(GL_LIGHT2,GL_SPECULAR,np.array([0.0,0.0,0.0,1],dtype=np.float32))
    glLightfv(GL_LIGHT2,GL_DIFFUSE,np.array([1.0,1.0,1.0,1],dtype=np.float32))
    glLightfv(GL_LIGHT2,GL_AMBIENT,np.array([0.3,0.3,0.3,1],dtype=np.float32))
    glEnable(GL_LIGHT2)

    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_VERTEX_ARRAY)
    glEnable(GL_NORMAL_ARRAY)
    gluPerspective(45, (self.window_size[0]/self.window_size[1]), 0.001, 10.0)

  def process_events(self):
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        self.done = True
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_KP5:
          self.trans_v = np.zeros(3)
          self.rot_mat = np.eye(3)
        elif event.key == pygame.K_SPACE:
          self.playing = not self.playing
        elif event.key == pygame.K_d:
          self.frame_idx -= 1
        elif event.key == pygame.K_f:
          self.frame_idx += 1
        elif event.key == pygame.K_p:
          img = pygame.image.tostring(self.screen, 'RGB')
          img = pygame.image.fromstring(img, self.window_size, 'RGB')
          img = np.transpose(pygame.surfarray.array3d(img), [1, 0, 2])
          imsave(
            '{date:%Y_%M_%d_%H_%M_%S}.png'.format(date=datetime.datetime.now()),
            img
          )
    pressed = pygame.key.get_pressed()

    # rotation
    if pressed[pygame.K_KP2]:
      rot_ax = [-1, 0, 0]
    elif pressed[pygame.K_KP8]:
      rot_ax = [1, 0, 0]
    elif pressed[pygame.K_KP4]:
      rot_ax = [0, -1, 0]
    elif pressed[pygame.K_KP6]:
      rot_ax = [0, 1, 0]
    elif pressed[pygame.K_KP1]:
      rot_ax = [0, 0, 1]
    elif pressed[pygame.K_KP3]:
      rot_ax = [0, 0, -1]
    else:
      rot_ax = None
    if rot_ax is not None:
      self.rot_mat = \
        np.matmul(axangle2mat(rot_ax, self.speed_rot), self.rot_mat)

    # translation
    if pressed[pygame.K_LEFT]:
      self.trans_v[0] += self.speed_trans
    if pressed[pygame.K_RIGHT]:
      self.trans_v[0] -= self.speed_trans
    if pressed[pygame.K_UP]:
      self.trans_v[1] -= self.speed_trans
    if pressed[pygame.K_DOWN]:
      self.trans_v[1] += self.speed_trans
    if pressed[pygame.K_q]:
      self.trans_v[2] += self.speed_trans
    if pressed[pygame.K_e]:
      self.trans_v[2] -= self.speed_trans

    if pressed[pygame.K_a]:
      self.frame_idx -= 1
    if pressed[pygame.K_s]:
      self.frame_idx += 1

    self.frame_idx = max(self.frame_idx, 0)
    self.frame_idx = min(self.frame_idx, self.n_frames)

  def norm_verts(self):
    maxi = np.max(self.verts, axis=(0, 1))
    mini = np.min(self.verts, axis=(0, 1))
    scale = np.max(maxi - mini)
    mean = np.mean(self.verts)
    self.verts = (self.verts - mean) / scale

  def render(self):
    verts = np.matmul(self.rot_mat, self.verts[self.frame_idx].T).T
    verts -= self.trans_v
    verts[..., 2] -= self.cam_offset

    face_coords = verts[self.faces]
    face_normals = self.get_normal(face_coords)

    dup_verts = verts[self.verts_mapping].copy()
    dup_normals = face_normals[self.normal_mapping].copy()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glVertexPointerf(dup_verts)
    glNormalPointerf(dup_normals)
    glDrawElementsui(GL_TRIANGLES, self.dup_faces)

  def run(self, verts, faces, video_path=None, video_fps=30, run_fps=30):
    """
    Render mesh animation.

    Parameters
    ----------
    verts : np.ndarray, [n, v, 3]
      Mesh vertices.
    faces : np.ndarray, [f, 3]
      Mesh faces.
    video_path : str, optional
      Path to save the rendered video if not None, by default None
    video_fps : int, optional
      Frame rate of saved video., by default 30
    run_fps : int, optional
      Frame rate of renderer, by default 30
    """
    self.faces = faces.astype(np.int32).copy()
    if type(verts) == list:
      verts = np.stack(verts, 0)
    self.verts = verts.copy()
    self.n_frames, self.n_verts, _ = self.verts.shape
    self.n_faces = self.faces.shape[0]
    self.norm_verts()
    self.setup_dup()

    self.frame_idx = -1

    if video_path is not None:
      video = VideoWriter(
        video_path, self.window_size[0], self.window_size[1], video_fps
      )

    if video_path is not None:
      self.playing = True

    while not self.done:
      self.process_events()

      if self.frame_idx >= self.n_frames:
        if video_path is not None:
          self.done = True
        self.frame_idx = self.n_frames - 1
      pygame.display.set_caption(
        'Frame %d / %d' % (self.frame_idx + 1, self.n_frames)
      )

      self.render()

      if video_path is not None:
        frame = pygame.image.tostring(self.screen, 'RGB')
        frame = pygame.image.fromstring(frame, self.window_size, 'RGB')
        frame = np.transpose(pygame.surfarray.array3d(frame), [1, 0, 2])
        frame = np.flip(frame, -1).copy()
        video.write_frame(frame)

      pygame.display.flip()

      if self.playing:
        self.frame_idx += 1
      if self.frame_idx >= self.n_frames and video_path is not None:
        self.done = True

      if run_fps is not None and video_path is None:
        self.clock.tick(run_fps)

    pygame.quit()
    if video_path is not None:
      video.close()
