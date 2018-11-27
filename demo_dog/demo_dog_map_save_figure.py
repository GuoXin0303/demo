#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import sys
sys.path.append("/home/workspace/tools/FRCN/tools/")
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv 
import cPickle
import uuid
#import get_voc_results_file_template, im_detect
from voc_eval import voc_eval
import datetime
os.environ["CUDA_VISIBLE_DEVICES"]='3'
CLASSES = ('__background__',  # always index 0
         'Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese', 'Shih-Tzu', 'Blenheim_spaniel',
         'papillon', 'toy_terrier', 'Rhodesian_ridgeback', 'Afghan_hound', 'basset', 'beagle',
         'bloodhound', 'bluetick', 'black-and-tan_coonhound', 'English_foxhound', 'redbone', 'borzoi',
         'Irish_wolfhound', 'Italian_greyhound', 'whippet', 'Ibizan_hound', 'Norwegian_elkhound',
         'otterhound', 'Saluki', 'Scottish_deerhound', 'Weimaraner', 'Staffordshire_bullterrier',
         'American_Staffordshire_terrier', 'Border_terrier', 'Kerry_blue_terrier', 'Irish_terrier',
         'Norfolk_terrier', 'Norwich_terrier', 'Yorkshire_terrier', 'wire-haired_fox_terrier',
         'Lakeland_terrier', 'Sealyham_terrier', 'Airedale', 'cairn', 'Australian_terrier',
         'Dandie_Dinmont', 'Boston_bull', 'giant_schnauzer', 'standard_schnauzer', 'Scotch_terrier',
         'Tibetan_terrier', 'silky_terrier', 'soft-coated_wheaten_terrier',
         'West_Highland_white_terrier', 'Lhasa', 'flat-coated_retriever', 'curly-coated_retriever',
         'golden_retriever', 'Labrador_retriever', 'Chesapeake_Bay_retriever',
         'German_short-haired_pointer', 'Walker_hound', 'Bedlington_terrier', 'miniature_schnauzer',
         'vizsla', 'kelpie', 'bull_mastiff', 'English_setter', 'Irish_setter', 'Gordon_setter',
         'Brittany_spaniel', 'clumber', 'English_springer', 'Welsh_springer_spaniel', 'cocker_spaniel',
         'Sussex_spaniel', 'Irish_water_spaniel', 'kuvasz', 'schipperke', 'groenendael', 'malinois',
         'briard', 'komondor', 'Old_English_sheepdog', 'Shetland_sheepdog', 'collie', 'Border_collie',
         'Bouvier_des_Flandres', 'Rottweiler', 'German_shepherd', 'Doberman', 'miniature_pinscher',
         'Greater_Swiss_Mountain_dog', 'Bernese_mountain_dog', 'Appenzeller', 'EntleBucher', 'boxer',
         'Tibetan_mastiff', 'French_bulldog', 'Great_Dane', 'Saint_Bernard', 'Eskimo_dog', 'malamute',
         'Siberian_husky', 'affenpinscher', 'basenji', 'pug', 'Leonberg', 'Newfoundland',
         'Great_Pyrenees', 'Samoyed', 'Pomeranian', 'chow', 'keeshond', 'Brabancon_griffon', 'Pembroke',
         'Cardigan', 'toy_poodle', 'miniature_poodle', 'standard_poodle', 'Mexican_hairless', 'dingo',
         'dhole', 'African_hunting_dog')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

def vis_detections(im, class_name, dets, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

#     im = im[:, :, (2, 1, 0)]
#     fig, ax = plt.subplots(figsize=(12, 12))
#     ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=1) 
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('cat detections with '
                  'p(cat | box) >= {:.1f}').format(
                                                  thresh),
                  fontsize=14)

#     plt.axis('off')
#     plt.tight_layout()
#     plt.draw()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    
    im = im[:, :, (2, 1, 0)]
    fig,ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        
        vis_detections(im, cls, dets, ax,thresh=CONF_THRESH)
    
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

####mAP
def get_voc_results_file_template(cls):
    #comp_id = ('comp4' + '_' + str(uuid.uuid4()))
    #date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')##这里把原来的编码名称改为日期（年-月-日-时-分-秒），方便查看
    #filename = date + '_det_' + 'test' + cls + '.txt'
    filename ='_det_' + 'test' + cls + '.txt'
    path = os.path.join(save_prob_path, filename)
    return path

######override im_detect()
def my_im_detect(net, im):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs = {'data' : None, 'rois' : None}

    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []
##这里图片都是一样大小
#    for target_size in cfg.TEST.SCALES:
#        im_scale = float(target_size) / float(im_size_min)
#        # Prevent the biggest axis from being more than MAX_SIZE
#        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
#            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
#        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
#                        interpolation=cv2.INTER_LINEAR)
#        im_scale_factors.append(im_scale)
#        processed_ims.append(im)


    im_scale = 1.0
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)        

    max_shape = np.array([imn.shape for imn in processed_ims]).max(axis=0)
    num_images = len(processed_ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        imn = processed_ims[i]
        blob[i, 0:imn.shape[0], 0:imn.shape[1], :] = imn
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)

    blobs['data'] = blob
    im_scales =  np.array(im_scale_factors)

    im_blob = blobs['data']
    blobs['im_info'] = np.array([[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['im_info'].reshape(*(blobs['im_info'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)

    blobs_out = net.forward(**forward_kwargs)

    assert len(im_scales) == 1, "Only single-image batch implemented"
    rois = net.blobs['rois'].data.copy()
    # unscale back to raw image space
    boxes = rois[:, 1:5] / im_scales[0]

    scores = blobs_out['cls_prob']

    box_deltas = blobs_out['bbox_pred']
    pred_boxes = bbox_transform_inv(boxes, box_deltas)
    pred_boxes = clip_boxes(pred_boxes, im.shape)

    return scores, pred_boxes





if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()
    prototxt = '/home/workspace/zhudd/FRCN/models/dogs/test.prototxt'
    caffemodel = '/home/workspace/zhudd/FRCN/output/faster_rcnn_end2end/dog_trainval/vgg_cnn_m_1024_faster_rcnn_ms_iter_14000.caffemodel'
    
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    im_names = ['/home/data/StanfordDogsDataset/merge/test/n02085620_11818.jpg']
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        #demo(net, im_name)
    #plt.savefig('/home/workspace/zhudd/MyFig.jpg')
    #plt.show()






    file_path = 'dog_testvol'
    test_file = '/home/data/StanfordDogsDataset/merge/main/im_test_list.txt'
    file_path_img = '/home/data/StanfordDogsDataset/merge/test'
    save_prob_path = '/home/workspace/zhudd/label/dog' ##生成的结果文件都保存在output里，包括detections.pkl，class_pr.pkl，和txt文件
    annopath = '/home/data/StanfordDogsDataset/merge/main/test/{}.xml'
    thresh = 0.05
    max_per_image = 100
    num_classes = 121
    with open(test_file) as f:
        image_index = [x.strip() for x in f.readlines()]
    net.name = os.path.splitext(os.path.basename(caffemodel))[0]
    num_images = len(image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    for i in xrange(num_images):
        image_path = os.path.join(file_path_img, image_index[i] + '.jpg')
        im = cv2.imread(image_path)

        _t['im_detect'].tic()
        scores, boxes = my_im_detect(net, im)
        _t['im_detect'].toc()

        _t['misc'].tic()
        # skip j = 0, because it's the background class
        for j in xrange(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)
    if not os.path.exists(save_prob_path):
        os.mkdir(save_prob_path)
    det_file = os.path.join(save_prob_path, 'detections.pkl')
    import subprocess
    subprocess.call('rm /home/workspace/zhudd/label/dog/*',shell=True)
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    for cls_ind, cls in enumerate(CLASSES):
        if cls == '__background__':
            continue
        print 'Writing {} VOC results file'.format(cls)
        filename = get_voc_results_file_template(cls)
        if not os.path.exists(filename):
            os.mknod(filename) 

        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(image_index):
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in xrange(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))

    #annopath = os.path.join(file_path, 'Annotations', '{:s}.xml')
    imagesetfile = os.path.join(file_path, 'ImageSets', 'Main', 'test.txt')
    cachedir = os.path.join(save_prob_path)
    aps = []
    cla_names=[]

    # The PASCAL VOC metric changed in 2010
    use_07_metric = True #True
    print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')

    for i, cls in enumerate(CLASSES):
        if cls == '__background__':
            continue
        #filename = get_voc_results_file_template(cls)
        detpath=save_prob_path+'/'+"_det_test{}.txt"
        rec, prec, ap = voc_eval(
            detpath.format(cls), annopath, test_file, cls, cachedir, ovthresh = 0.5,
            use_07_metric = use_07_metric)
        #
        aps += [ap]
        cla_names += [cls]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(save_prob_path, cls + '_pr.pkl'), 'w') as f:
            cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    #plt.switch_backend('agg')
    name_list = cla_names
    num_list = aps
    plt.bar(range(len(num_list)), num_list,color='rgb',tick_label=name_list)
    plt.savefig('/home/workspace/shengong/mapFig.jpg')
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    







