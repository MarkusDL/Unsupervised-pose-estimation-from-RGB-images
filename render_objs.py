import random
from random import uniform
import pyrender
import trimesh
import numpy as np
import pandas
from scipy.spatial.transform import Rotation as R
from transform3d import Transform
import matplotlib.pyplot as plt
import cv2

def get_cam_setups(R, t, n):

    setups = []
    for i in range(n):
        x_min, x_max = -0.5,0.5
        y_min, y_max = -0.5,0.5
        z_min, z_max = 2.0,2.0

        cam1_x, cam1_y, cam1_z = uniform(x_min,x_max), uniform(y_min,y_max), uniform(z_min,z_max)

        T_cam1 = np.asarray([[1,0,0,cam1_x-t[0]/2],[0,1,0,cam1_y-t[1]/2], [0,0,1,cam1_z-t[2]/2],[0,0,0,1]])

        T_cam1_cam2 = np.eye(4)
        T_cam1_cam2[0:3,0:3]=R
        T_cam1_cam2[0:3,3] =t
        T_cam2 = T_cam1@T_cam1_cam2
        setup = [T_cam1,T_cam2, R, t]
        setups.append(setup)

    return setups


R_cam1_cam2 = np.eye(3)
t_cam1_cam2 = np.array((0.5,0,0))

setups = get_cam_setups(R_cam1_cam2, t_cam1_cam2, 1000)#[[cam1_pose, cam2_pose]]

scene = pyrender.Scene(ambient_light=np.array([0.02, 0.02, 0.02, 0.1]))
point_l = pyrender.PointLight(color=np.ones(3), intensity=50.0)

l_pose = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 2.0],
    [0.0, 0.0, 0.0, 1.0]
])
point_l_node = scene.add(point_l, pose=l_pose)

drillpink_trimesh = trimesh.load('Models/drillpink.obj')
drillpink_obj = pyrender.Mesh.from_trimesh(drillpink_trimesh)

drillred_trimesh = trimesh.load('Models/drillred.obj')
drillred_obj = pyrender.Mesh.from_trimesh(drillred_trimesh)

drillyellow_trimesh = trimesh.load('Models/drillyellow.obj')
drillyellow_obj = pyrender.Mesh.from_trimesh(drillyellow_trimesh)

obj_list = [drillpink_obj,drillred_obj,drillyellow_obj]

res = 128
save_path = "/home/markus/Documents/GitHub/Unsupervised-pose-estimation-from-RGB-images/3DdatasetImgs/"
count = 0
with open("setups.txt", "w") as out_file:

    for setup in setups:
        obj_node_list = []
        for obj in obj_list:
            obj_pose = np.eye(4)
            R_rand = np.eye(3) #R.random(random_state=random.randint(0,1213414)).as_matrix()
            obj_pose[0:3,0:3] = R_rand
            obj_pose[0:2, 3] = np.random.uniform(-0.5, 0.5, 2)
            obj_node = scene.add(obj, pose=obj_pose)
            obj_node_list.append(obj_node)

        out_file.write(np.array2string(np.ravel(setup[2])) +","+np.array2string(setup[3])+"\n")

        for cam_idx, cam_pos in enumerate(setup[0:2]):
            cam = pyrender.PerspectiveCamera(yfov=(np.pi/3.0))
            cam_node = scene.add(cam, pose=cam_pos)

            r = pyrender.OffscreenRenderer(viewport_width=res, viewport_height=res)
            color, depth = r.render(scene)
            scene.remove_node(cam_node)
            r.delete()

            cv2.imwrite(save_path+str(count).zfill(5)+"_cam"+str(cam_idx)+".png", cv2.cvtColor(color,cv2.COLOR_RGB2BGR))

        for obj_node in obj_node_list:
            scene.remove_node(obj_node)

        count += 1
        print(count)