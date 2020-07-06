import numpy as np
import open3d as o3d
from .io import *
from tqdm import tqdm
from transforms3d.axangles import axangle2mat


def render_sequence_3d(verts, faces, width, height, video_path, fps=30,
                       visible=False, need_norm=True):
  """
  Render mesh animation using open3d.

  Parameters
  ----------
  verts : np.ndarray, shape [n, v, 3]
    Mesh vertices for each frame.
  faces : np.ndarray, shape [f, 3]
    Mesh faces.
  width : int
    Width of video.
  height : int
    Height of video.
  video_path : str
    Path to save the rendered video.
  fps : int, optional
    Video framerate, by default 30
  visible : bool, optional
    Wether to display rendering window, by default False
  need_norm : bool, optional
    Normalizing the vertices and locate camera automatically or not, by default
    True
  """
  if need_norm:
    if type(verts) == list:
      verts = np.stack(verts, 0)

    scale = np.max(np.max(verts, axis=(0, 1)) - np.min(verts, axis=(0, 1)))
    mean = np.mean(verts, axis=(0, 1), keepdims=True)
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
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    frame = (np.asarray(vis.capture_screen_float_buffer()) * 255).astype(np.uint8)
    writer.write_frame(frame)

  writer.close()


def joints_to_mesh_prism(joints, parents, color=None, thickness=0.2):
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

  return verts, faces


def joints_to_mesh_cylinder(joints, parents, thickness=1.0):
  """
  Produce a mesh representing the skeleton with given joint coordinates.
  Bones are represented by cylinders and joints spheres.

  Parameters
  ----------
  joints : np.ndarray, shape [k, 3]
    Joint coordinates.
  parents : list
    Parent joint of each child joint.
  thickness : float, optional
    The size of the joint sphere relative to length, by default 0.2

  Returns
  -------
  np.ndarray, shape [v, 3]
    Vertices of the mesh.
  np.ndarray, shape [f, 3]
    Face indices of the mesh.
  """
  faces = []
  verts = []
  v_cnt = 0
  radius = None
  for c, p in enumerate(parents):
    if p is None:
      continue
    parent = joints[p]
    child = joints[c]
    delta = parent - child
    delta_len = np.linalg.norm(delta)
    delta_unit = delta / delta_len

    if radius is None:
      radius = delta_len * thickness / 2
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius, resolution=16)
    s_verts = np.asarray(sphere.vertices)
    s_faces = np.asarray(sphere.triangles)

    # bone cylinder

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(
      radius / 2, height=delta_len, split=1, resolution=16
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
  """
  Produce a mesh representing the skeleton with given joint coordinates.

  Parameters
  ----------
  joints : np.ndarray, shape [k, 3]
    Joint coordinates.
  parents : list
    Parent joint of each child joint.
  style : str
    'prism' or 'cylinder'.
  thickness : float, optional
    The thickness of the bone relative to length, by default 0.2

  Returns
  -------
  np.ndarray, shape [v, 3]
    Vertices of the mesh.
  np.ndarray, shape [f, 3]
    Face indices of the mesh.
  """
  if style == 'prism':
    return joints_to_mesh_prism(joints, parents, thickness=thickness)
  elif style == 'cylinder':
    return joints_to_mesh_cylinder(joints, parents, thickness=thickness)
  else:
    raise RuntimeError('Invalid style: ' % style)


def create_o3d_mesh(verts, faces):
  """
  Create a open3d mesh from vertices and faces.

  Parameters
  ----------
  verts : np.ndarray, shape [v, 3]
    Mesh vertices.
  faces : np.ndarray, shape [f, 3]
    Mesh faces.

  Returns
  -------
  o3d.geometry.TriangleMesh
    Open3d mesh.
  """
  mesh = o3d.geometry.TriangleMesh()
  mesh.triangles = o3d.utility.Vector3iVector(faces)
  mesh.vertices = o3d.utility.Vector3dVector(verts)
  mesh.compute_vertex_normals()
  return mesh


def vis_mesh(verts, faces, width=1080, height=1080):
  """
  Visualize mesh with open3d.

  Parameters
  ----------
  verts : np.ndarray, shape [v, 3]
    Mesh vertices.
  faces : np.ndarray, shape [f, 3]
    Mesh faces.
  width : int
    Window width, by default 1080.
  height : int
    Window height, by default 1080.
  """
  mesh = create_o3d_mesh(verts, faces)
  viewer = o3d.visualization.Visualizer()
  viewer.create_window(width=width, height=height, visible=True)
  viewer.add_geometry(mesh)
  viewer.run()


show_mesh = vis_mesh
