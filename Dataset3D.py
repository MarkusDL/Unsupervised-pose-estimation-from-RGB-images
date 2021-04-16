import cv2
import torch
import numpy as np
import os
import random

class Dataset(torch.utils.data.Dataset):
    def __init__(self, epoch_size=1000, img_path="", setup_path =""):
        super().__init__()
        self.img_path = img_path
        self.setup_path = setup_path
        self.epoch_size = epoch_size
        self.img_files = os.listdir(img_path)
        self.n_img = int(len(self.img_files)/2)
        self.Rs, self.ts = self.get_setups(setup_path)

    def __len__(self):
        return self.epoch_size

    def get_setups(self, path):
        Rs =[]
        ts =[]
        with open(path, "r") as file:
            for line in file.read().splitlines():
                array = line.split(",")
                R = np.reshape(np.fromstring(array[0][1:-2], sep=' '), (3, 3))[0:2,0:2]
                Rs.append(R)
                t = np.fromstring(array[1][1:-2],sep=' ')[0:2]
                ts.append(t)

        return Rs, ts

    def __getitem__(self, idx):

        r_index = random.randint(0,self.n_img-1)
        cam0 = "cam0"
        cam1 = "cam1"

        image_A = cv2.imread(self.img_path+str(r_index).zfill(5)+"_"+cam0+".png").transpose(2, 0, 1)
        image_B = cv2.imread(self.img_path+str(r_index).zfill(5)+"_"+cam1+".png").transpose(2, 0, 1)
        R = self.Rs[r_index]
        t = self.ts[r_index]*(1/2.2857)

        return image_A, image_B, np.asarray(R,dtype=np.float32), np.asarray(t,dtype=np.float32)



setup_path = "/home/markus/Documents/GitHub/Unsupervised-pose-estimation-from-RGB-images/setups.txt"
img_path = "/home/markus/Documents/GitHub/Unsupervised-pose-estimation-from-RGB-images/3DdatasetImgs/"

dset = Dataset(1000, img_path=img_path, setup_path=setup_path)
img_a,_,_,_= dset.__getitem__(0)
print(img_a.shape)

