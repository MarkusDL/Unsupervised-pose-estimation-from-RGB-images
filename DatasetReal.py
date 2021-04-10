import numpy as np
import torch
import torchvision
import scipy
import os
import cv2


def crop_center(img, res):
    center = np.asarray(img.shape) / 2
    h,w = res,res
    x = center[1] - res / 2
    y = center[0] - res / 2

    return img[int(y):int(y + h), int(x):int(x + w)]

def rezise(img, res):
    return cv2.resize(img,(res,res))

def resize_and_crop_imgs(img,res):

    img_w = img.shape[0]
    img_h = img.shape[1]
    min_size = min(img_h, img_w)
    i = 0

    img = crop_center(img, min_size)
    img = rezise(img,res)
    return img




def nmz(a, axis=-1):
    return a / np.linalg.norm(a, axis=axis, keepdims=True)

def sample_disc(d):
    while True:
        p = np.random.randn(d)
        if np.linalg.norm(p) < 1:
            return p

def R_from_theta(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))

def theta_from_R(R):
    return np.arctan2(R[1, 0], R[0, 0])

assert np.allclose(theta_from_R(R_from_theta(1)), 1)

def random_pose():
    R = R_from_theta(np.random.uniform(0, np.pi * 2))
    t = sample_disc(2) * 0.25
    return R, t

def mat(R, t):
    m = np.empty((3, 3))
    m[:2, :2] = R
    m[:2, 2] = t
    m[2] = 0, 0, 1
    return m

def inv(R, t):
    return R.T, -t @ R

def mult(Ra, ta, Rb, tb):
    return Ra @ Rb, ta + tb @ Ra.T


class Dataset(torch.utils.data.Dataset):
    def __init__(self, res=128, epoch_size=1000, path="", augment = None):
        super().__init__()
        self.res = res
        self.root_dir = path
        self.u_shape = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * res * 0.05
        self.epoch_size = epoch_size
        self.files = os.listdir(path)
        self.nfiles = len(self.files)
        self.current_file = 0
        self.augment = augment
        self.static_pose = random_pose()


    def __len__(self):
        return self.epoch_size


    def get_tranformed_image(self, img_A, R, T):
        res = img_A.shape[0]
        T = [[1,0,T[0]*res],[0,1,T[1]*res]]

        image = cv2.copyMakeBorder(img_A, res, res, res, res, cv2.BORDER_CONSTANT)

        MT = np.asarray(T)
        img_B = cv2.warpAffine(image, MT, image.shape[0:2], cv2.INTER_CUBIC)

        center = tuple(np.asarray(image.shape[0:2])/2)
        MR = cv2.getRotationMatrix2D(center, np.rad2deg(theta_from_R(R)), 1)

        img_B = cv2.warpAffine(img_B, MR, image.shape[0:2], cv2.INTER_CUBIC)

        return crop_center(img_B, res)


    def augment(self, img):

        return img

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.files[self.current_file])
        image_A = cv2.imread(img_name)
        image_A = resize_and_crop_imgs(image_A, self.res)

        if self.augment:
            image_A = self.augment(image_A)

        R, T = random_pose()
        image_B = self.get_tranformed_image(image_A, R, T)


        self.current_file += 1

        if self.current_file >= self.nfiles:
            self.current_file = 0

        return image_A, image_B, R, T


path = "RealImgs_crop_rsize/"
realdataset = Dataset(path=path, res=512)


cv2.namedWindow("A", cv2.WINDOW_FREERATIO)

for i in range(1000):
    imgA, imgB, r, t = realdataset.__getitem__(1)

    img = np.hstack([imgA, imgB])
    cv2.imshow("A", img)
    cv2.waitKey(1)