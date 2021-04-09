import pyrender
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import cv2

obj_trimesh = trimesh.load('Models/drill.obj')
mesh = pyrender.Mesh.from_trimesh(obj_trimesh)


cam1 = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))
cam1_pose = np.array([
    [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 0.5],
    [1.0, 0.0,           0.0,           0.0],
    [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 0.4],
    [0.0,  0.0,           0.0,          1.0]
])

cam2 = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))

cam2_pose = np.array([
    [0.0,  -np.sqrt(2)/2, np.sqrt(2)/2, 0.9],
    [1.0, 0.0,           0.0,           0.0],
    [0.0,  np.sqrt(2)/2,  np.sqrt(2)/2, 0.4],
    [0.0,  0.0,           0.0,          1.0]
])
setups = [[cam1_pose, cam2_pose]]

scene = pyrender.Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))
obj_pose = np.eye(4)
obj_pose[0, 3] = 0.1
obj_pose[2, 3] = -np.min(obj_trimesh.vertices[:, 2])
obj_node = scene.add(mesh, obj_pose)


for setup in setups:
    for i in range(1000):
        pos = setup[0]

        cam = pyrender.PerspectiveCamera(yfov=(np.pi/3.0))
        cam_node = scene.add(cam, pose=pos)

        r = pyrender.OffscreenRenderer(viewport_width=128, viewport_height=128)
        color, depth = r.render(scene)
        print(i)