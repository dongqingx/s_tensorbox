# -*- coding: utf-8 -*-

"""
Module implementing MainWindow.
"""

from PyQt4.QtCore import pyqtSignature
from PyQt4.QtGui import QMainWindow
from PyQt4 import QtCore
from PyQt4.QtCore import *
from PyQt4.QtGui import *



from PIL import Image,ImageDraw,ImageFont

import numpy
import cv2

import os
import scipy
import collections
import sys
import time
from Ui_mainWindow import Ui_MainWindow

from math import ceil

import subprocess
import thread

import Queue
import threading
import datetime

import cv2
import matplotlib.pyplot as plt
import json
import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize
from scipy import misc
import matplotlib.image as mpimg

from train import build_forward
from utils.annolist import AnnotationLib as al
from utils.train_utils import add_rectangles, rescale_boxes


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

weights_file = "output/lstm_googlenet/save.ckpt-90000"
hypes_file = "output/lstm_googlenet/hypes.json"

selectedFont1 = ImageFont.truetype('./font/guanjiaKai.ttf', 45)
selectedFont2 = ImageFont.truetype('./font/guanjiaKai.ttf', 25)

font = cv2.FONT_HERSHEY_SIMPLEX
green = (0, 255, 0)
red = (255, 0, 0)
x_width = 240 #300
y_height = 210 #240
messages=Queue.Queue()
messagesTime=Queue.Queue()



class MainWindow(QMainWindow, Ui_MainWindow):

    """
    Class documentation goes here.
    """
    def __init__(self, parent=None):
        """
        Constructor
        
        @param parent reference to the parent widget
        @type QWidget
        """
        QMainWindow.__init__(self, parent)
        self.setupUi(self)
        
        self.status = "IDLE"
        self.timeFlag = True
        self.videoSourse = "./data/test.avi"
        self.frame_id = 1
        self.stop = False
        self.clear = False
        self.reset = False
        self.start = True

        self.videoCapture = cv2.VideoCapture("video/test.avi")
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.playVideo)
        self._timer.start(500)

        self.video_output_height = 640  #850
        self.video_output_width = 1000  #1400 

        with open(hypes_file, 'r') as f:
            self.H = json.load(f)
           
        tf.reset_default_graph()
        self.x_in = tf.placeholder(tf.float32, name='x_in', shape=[self.H['image_height'], self.H['image_width'], 3])
        if self.H['use_rezoom']:
            self.pred_boxes, self.pred_logits, self.pred_confidences, self.pred_confs_deltas, self.pred_boxes_deltas = build_forward(self.H, 
                                        tf.expand_dims(self.x_in, 0), tf.expand_dims(self.x_in, 0), tf.expand_dims(self.x_in, 0), 
                                        tf.expand_dims(self.x_in, 0), tf.expand_dims(self.x_in, 0), tf.expand_dims(self.x_in, 0),
                                        tf.expand_dims(self.x_in, 0), tf.expand_dims(self.x_in, 0), tf.expand_dims(self.x_in, 0),
                                        tf.expand_dims(self.x_in, 0), 
                                        'test', reuse=None)
            grid_area = self.H['grid_height'] * self.H['grid_width']
            self.pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(self.pred_confs_deltas, 
                        [grid_area * self.H['rnn_len'], 2])), [grid_area, self.H['rnn_len'], 2])
            if self.H['reregress']:
                self.pred_boxes = self.pred_boxes + self.pred_boxes_deltas
        else:
            self.pred_boxes, self.pred_logits, self.pred_confidences = build_forward(self.H, tf.expand_dims(self.x_in, 0), tf.expand_dims(self.x_in, 0), 
                                                                        tf.expand_dims(self.x_in, 0), tf.expand_dims(self.x_in, 0),
                                                                        tf.expand_dims(self.x_in, 0), tf.expand_dims(self.x_in, 0),
                                                                        tf.expand_dims(self.x_in, 0), tf.expand_dims(self.x_in, 0),
                                                                        tf.expand_dims(self.x_in, 0), tf.expand_dims(self.x_in, 0), 'test', reuse=None)
        
        saver = tf.train.Saver()
        
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        saver.restore(self.sess, weights_file)

    def playVideo(self):
        weekValueShow = {
            '0' : u'星期日', 
            '1' : u'星期一', 
            '2' : u'星期二', 
            '3' : u'星期三', 
            '4' : u'星期四', 
            '5' : u'星期五', 
            '6' : u'星期六'
        }
        if self.timeFlag :
            T=time.localtime(time.time())
            self.label_date.setText(time.strftime('%Y-%m-%d',T))
            self.label_time.setText(time.strftime('%H:%M:%S', T))
            self.label_week.setText(weekValueShow.get(time.strftime('%w', T)))
        daytime = str(time.strftime("%H:%M:%S", time.localtime()))
        hours = int(daytime.split(':')[0])
        minutes = int(daytime.split(':')[1])
        seconds = int(daytime.split(':')[2])

        if not self.start:
            return
 
        ret, frame_bgr = self.videoCapture.read() # input size: 480x640
        # cv2.waitKey(500)
        if not ret or self.reset:
            self.videoCapture = cv2.VideoCapture(self.videoSourse)
            self.reset = False 
        if ret == True:
            self.frame_id += 1
            if not self.clear:
                frame_bgr = numpy.array(frame_bgr)
                img = imresize(frame_bgr, (self.H["image_height"], self.H["image_width"]), interp='cubic')
                #img = frame
                # print 'process begin'
                feed = {self.x_in: img}
                (np_pred_boxes, np_pred_confidences) = self.sess.run([self.pred_boxes, self.pred_confidences], feed_dict=feed)
                new_frame, rects, rect_count = add_rectangles(self.H, [img], np_pred_confidences, np_pred_boxes,
                        use_stitching=True, rnn_len=self.H['rnn_len'], min_conf=0.2, tau=0.25, show_suppressed=False)
                # print 'process end'
                # end_time = time.time()
                #print end_time - start_time
                font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 0.8, 0.5, 0, 2, 1)
                new_frame = cv2.cv.fromarray(new_frame)
                cv2.cv.PutText(new_frame, "number:  " + str(rect_count), (20,35), font, (255,0,0))
                cv2.cv.PutText(new_frame, "crowded rate: " + str(float(rect_count)/50.0), (20,60), font, (255,0,0))
                # print '\nframe: ', self.frame_id
                frame_bgr = new_frame

            frame_bgr = imresize(frame_bgr, (self.video_output_height, self.video_output_width))
        
            # face detect and align use RGB image
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) # (480,640,3)
            display_frame = numpy.array(frame_rgb)
            # self.cap_frame = display_frame.copy()
            
            #if self.frame_id % self.skip_frames == 1:
                # detect face
		    #time_1 = time.time()
            height, width = display_frame.shape[:2]
            img = QImage(display_frame, width, height, QImage.Format_RGB888)
            img = QPixmap.fromImage(img)
            self.DetectLabel.setPixmap(img)
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB) # (480,640,3)

    
    def on_stopVideo_clicked(self):
        self.start = False

    def on_startVideo_clicked(self):
        self.start = True
        self.clear = False

    def on_ResetVideo_clicked(self):
        self.reset = True

    def on_NoDisplay_clicked(self):
        self.clear = True
