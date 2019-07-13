import os
import numpy as np
import cv2
import torch
from torch.autograd import Variable
from retinaface import load_retinaface_mbnet
from utils import RetinaFace_Utils

class Retinaface_Detector(object):
    def __init__(self):
        self.threshold = 0.8
        self.model = load_retinaface_mbnet()
        self.pixel_means = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.pixel_stds = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.pixel_scale = float(1.0)
        self.utils = RetinaFace_Utils()

    def img_process(self, img):
        target_size = 1024
        max_size = 1980
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        im = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im = im.astype(np.float32)

        im_tensor = np.zeros((1, 3, im.shape[0], im.shape[1]), dtype=np.float32)
        for i in range(3):
            im_tensor[0, i, :, :] = (im[:, :, 2 - i] / self.pixel_scale - self.pixel_means[2 - i]) / \
                                    self.pixel_stds[2 - i]
        return im_tensor, im_scale

    def detect(self, img):
        results = []
        im, im_scale = self.img_process(img)
        im = torch.from_numpy(im)
        im_tensor = Variable(im)
        output = self.model(im_tensor)
        faces, landmarks = self.utils.detect(im, output, self.threshold, im_scale)
        
        if faces is None or landmarks is None:
            return results
        
        for face, landmark in zip(faces, landmarks):
            face = face.astype(np.int)
            landmark = landmark.astype(np.int)
            results.append([face, landmark])
        
        return results
        
