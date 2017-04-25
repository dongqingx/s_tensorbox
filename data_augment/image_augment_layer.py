"""
For image data augment layer

by Donny You

2016/12/08
"""

import caffe
import numpy as np
import numpy.random as nr
import yaml
from random import shuffle
import cv2
import os
from util import *
import pdb

from data_augment import *


class ImageAugmentNetTrain(caffe.Layer):
    """Data layer for training"""   
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self.batch_size = layer_params['batch_size']
        self.dir_path = layer_params['im_path']
        self.mode = layer_params['mode']
        self.labels, self.im_list = self.gait_dir_processor()
        self.idx = 0
        self.data_num = len(self.labels)
        self.rnd_list = np.arange(self.data_num)
        shuffle(self.rnd_list)
        
    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.im
        top[1].data[...] = self.label

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        # load image + label image pairs
        self.im = []
        self.label = []

        for i in xrange (self.batch_size):
            if self.idx != self.data_num:
                cur_idx = self.rnd_list[self.idx]
                im_path = self.im_list[cur_idx]
                for im in self.load_data(im_path)
                    self.im.append(im)
                    self.label.append(self.labels[cur_idx])
                self.idx +=1
                #pdb.set_trace()
            else:
                self.idx = 0
        self.im = np.array(self.im).astype(np.float32)
        self.label = np.array(self.label).astype(np.float32)
        # reshape tops to fit blobs
        
        top[0].reshape(*self.im.shape)
        top[1].reshape(*self.label.shape)
        
    def image_dir_processor(self):
        image_dic = './image_dic'
        if not os.path.exists(image_dic):
            labels = []
            im_path_list = []
            im_list = np.sort(os.listdir(self.dir_path))
            index = 0
            for image_file in im_list:
                label = int(image_file.split("_")[0])
                im_path_list.append(self.dir_path + "/" + image_file)
                labels.append(label)

            dic = {'im_path_list':im_path_list, 'labels':labels}
            pickle(image_dic, dic, compress=False)
        else:
            dic = unpickle(image_dic)
            im_list = dic['im_path_list']
            labels = dic['labels']
        return labels, im_path_list
    
    def load_data(self, image_path):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        image_list = list()
        image = DataAugmentation.open_image(image_path)

        for im in colorJitter(image):
            im = np.array(im, dtype=np.float32)
            im -= 127.5
            image_list.append(im)

        for im in addNoisy(image):
            im = np.array(im, dtype=np.float32)
            im -= 127.5
            image_list.append(im)

        for im in gradualBrightness(image):
            im = np.array(im, dtype=np.float32)
            im -= 127.5
            image_list.append(im)

        return image_list            
