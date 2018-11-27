#!/usr/bin/env python
# -*- coding: utf-8 -*-

'a hello world GUI example.'
import sys
sys.path.append("/home/workspace/tools/FRCN/tools/")
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
# import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, cv2
import argparse
import datetime
# import Tkinter
# import pymysql
import math
from Tkinter import *
import tkMessageBox


CLASSES = ('__background__',
           'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
           'truck', 'boat',  'traffic light', 'fire hydrant', 'stop sign',
           'parking meter', 'bench', 'bird',  'cat' ,'dog', 'horse', 'sheep',
           'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
           'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
           'sports ball', 'kite', 'baseball bat', 'baseball glove','skateboard',
           'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
           'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
           'broccoli', 'carrot' , 'hot dog', 'pizza' ,'donut' ,'cake', 'chair',
           'couch' , 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
           'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier', 'toothbrush')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        }

def im_resize(im_path,X_s,Y_s):
    im = cv2.imread(im_path)
    X_im = im.shape[1]
    Y_im = im.shape[0]
    print (X_im,Y_im)
    r = 0
    if ((X_im < X_s) | Y_im < Y_s):
        r =1
        # print (float(X_im) / X_s, float(Y_im) / Y_s)
        im_r = min(float(X_im) / X_s, float(Y_im) / Y_s)
        # print im_r
        im = cv2.resize(im, (int(X_im / im_r), int(Y_im / im_r)),interpolation=cv2.INTER_CUBIC)
        # print im.shape
    return im,r


def im_cut_scenery(im ,X_im,Y_im,X_s,Y_s):
    if ((float(X_im)/X_s)>(float(Y_im)/Y_s)):
        X1 = X_im/2 - X_s*(Y_im/Y_s)/2
        X2 = X_im/2 + X_s*(Y_im/Y_s)/2
        cut = im[:, X1:X2]
    else:
        Y1 = Y_im/2 - Y_s*(X_im/X_s)/2
        Y2 = Y_im / 2 + Y_s * (X_im / X_s) / 2
        cut = im[Y1:Y2,: ]
    return cut



def im_cut_whole(Xi,X_im,X_s,Y_s):
    X1 = Xi - X_s / 2
    print X1
    X1 = int(max(0, X1))
    X2 = Xi + X_s / 2
    print X2
    X2 = int(min(X_im, X2))

    Y1 = 0
    print Y1
    Y2 = Y_s
    print Y2
    return [X1, X2, Y1, Y2]


def im_cut_patch(Xi, X_im, X_r, Yi, Y_im, Y_r):
    X1 = Xi-X_r/2
    X2 = Xi + X_r / 2
    if(X1<0):
        X1 = 0
        X2 = X_r
    elif(X2>X_im):
        X2=X_im
        X1=X_im-X_r


    Y1 = Yi-Y_r/2
    Y2 = Yi + Y_r / 2
    if(Y1<0):
        Y1=0
        Y2=Y_r
    elif(Y2>Y_im):
        Y2=Y_im
        Y1=Y_im-Y_r


    return [int(X1), int(X2), int(Y1), int(Y2)]


def im_cut_rate_patch(Xi, X_im, X_s, Yi, Y_im, Y_s):
    X1 = Xi-X_s/2
    X2 = Xi + X_s / 2
    if(X1<0):
        X1 = 0
        X2 = x_S
    elif(X2>X_im):
        X2=X_im
        X1=X_im-X_s


    Y1 = Yi-Y_s/2
    Y2 = Yi + Y_s / 2
    if(Y1<0):
        Y1=0
        Y2=Y_s
    elif(Y2>Y_im):
        Y2=Y_im
        Y1=Y_im-Y_s


    return [int(X1), int(X2), int(Y1), int(Y2)]


def demo(net, im_path ,X_s,Y_s):
    """Detect object classes in an image using pre-computed object proposals."""
    im = cv2.imread(im_path)

    # Load the demo image
    X_im = im.shape[1]
    Y_im = im.shape[0]

    S=im.shape[0]*im.shape[1]
    # # print S

    X1 = X_im / 3
    X2 = X_im - X1
    Y1 = Y_im / 3
    Y2 = Y_im - Y1



    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    # print ('Detection took {:.3f}s for '
    #       '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    # im = im[:, :, (2, 1, 0)]
    class_name = ""
    class_list = []
    cut_list = []

    names = locals()
    j = 0

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]


            print (bbox[0], bbox[1], bbox[2], bbox[3])
            Xi = (bbox[2] + bbox[0]) / 2
            Yi = (bbox[3] + bbox[1]) / 2

            if (Yi < Y1):
                bbox[1] = bbox[1]/2
            elif (Yi > Y2):
                bbox[3] = bbox[3]+(Y_im-bbox[3])/2
            else:
                bbox[1] = bbox[1] - bbox[1] / 2
                bbox[3] = bbox[3] + (Y_im - bbox[3]) / 2

            X = (bbox[2] - bbox[0])
            Y = (bbox[3] - bbox[1])
            print (X, Y)

            X_rate = X/X_s
            Y_rate = Y/Y_s
            rate = max(X_rate,Y_rate)
            print rate

            #print cls



            X_r = rate*X_s
            Y_r = rate*Y_s



            box = im_cut_patch(Xi, X_im, X_r, Yi, Y_im, Y_r)
            print box
            cut = im[box[2]:box[3], box[0]:box[1]]


            # W1 = math.sqrt((1-abs(Xi-X)/X)*(1-abs(Yi-Y)/Y))
            # # print W1
            Si = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            W2=Si/S
            # print W2
            # Wi = W1*W2
            # # print Wi
            # Wi = round(math.sqrt(math.sqrt(Wi)),3)
            if(W2>0.1):
                class_list.append((cls, W2, j))
                cut_list.append(cut)
                j += 1


    # print class_n
    # print class_list.sort()
    if(cut_list==[]):
        cut_max =im_cut_scenery(im ,X_im,Y_im,X_s,Y_s)
    else:
        class_w = sorted(class_list, key=lambda class_list: class_list[1], reverse=True)
        # print class_list[[1]]
        print class_w
        S_max = class_w[0][-1]
        print S_max
        cut_max = cut_list[S_max]

    cut_max = cv2.resize(cut_max, (X_s, Y_s), interpolation=cv2.INTER_CUBIC)
    return cut_max

    # for class_i in class_w:
    #     print class_i
        # class_name += class_i + ','
    # class_n = class_name[:-1]





def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    # parser.add_argument('imagepath')
    # parser.add_argument('width')
    # parser.add_argument('height')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [1]',
                        default=2, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')


    args = parser.parse_args()

    return args


class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()

    def createWidgets(self):
        self.helloLabel = Label(self, text=u'输入你的图片路径!')
        self.helloLabel.pack()
        self.nameInput = Entry(self)
        self.nameInput.pack()
        self.helloLabel = Label(self, text=u'输入你的屏幕宽度!')
        self.helloLabel.pack()
        self.X_Input = Entry(self)
        self.X_Input.pack()
        self.helloLabel = Label(self, text=u'输入你的屏幕高度!')
        self.helloLabel.pack()
        self.Y_Input = Entry(self)
        self.Y_Input.pack()
        self.alertButton = Button(self, text=u'确认', command=self.hello)
        self.alertButton.pack()

    def hello(self):
        im_path = self.nameInput.get()
        X_s = int(self.X_Input.get())
        Y_s = int(self.Y_Input.get())
        print im_path
        # im_re,r = im_resize(im_path,X_s,Y_s)
        im_crop = demo(net, im_path, X_s, Y_s)
        im_name = im_path.split('/')[-1].split('.')[0]
        # im_root = im_path.replace(im_name,'')
        save_path = im_path.replace(im_name, 'cut/' + im_name + '_' + str(X_s) + '_' + str(Y_s))
        print save_path
        cv2.imwrite(save_path, im_crop)
        tkMessageBox.showinfo('Message', 'The picture is saved in %s' % save_path)

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    caffe_root = '/home/workspace/tools/caffe/'
    prototxt = '/home/workspace/tools/FRCN/models/coco/VGG16/faster_rcnn_end2end/test.prototxt'
    caffemodel = '/home/workspace/tools/FRCN/data/faster_rcnn_models/coco/coco_vgg16_faster_rcnn_final.caffemodel'

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id

    t1 = datetime.datetime.now()
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    t2 = datetime.datetime.now()
    print 'load model:', t2 - t1

    print '\n\nLoaded network {:s}'.format(caffemodel)

    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _ = im_detect(net, im)

    app = Application()
    app.master.title('Screen interception')
    # 主消息循环:
    app.mainloop()
