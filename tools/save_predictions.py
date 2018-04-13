#!/usr/bin/env python

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

CLASSES = ('__background__', # always index 0
              'large-vehicle', 'swimming-pool', 'helicopter', 'bridge', 'plane',
              'ship', 'soccer-ball-field', 'basketball-court','ground-track-field',
              'small-vehicle', 'harbor', 'baseball-diamond','tennis-court', 'roundabout', 'storage-tank')

NETS = {'vgg16': ('VGG16',
                  'vgg16_faster_rcnn_iter_70000.caffemodel'),
        'zf': ('ZF',
                  'zf_faster_rcnn_iter_10000.caffemodel')}


def vis_detections(im, class_name, dets, image_name, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    directory = args.model_dir + 'prediction'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + '/' + 'pred_'+ image_name)
    #plt.draw()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(args.data_dir, image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.7
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, image_name, thresh=CONF_THRESH)


def demoCombined(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(args.data_dir, image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.7
    NMS_THRESH = 0.3
    im = im[:, :, (2, 1, 0)]

    _ , ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')

    plt.imshow(im)

    ax = plt.gca()
    #ax.set_autoscale_on(True)

    print image_name
    my_dpi = 96
    CONTAIN_OBJ = False
    for cls_ind, class_name in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #print dets
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0:
            continue
        CONTAIN_OBJ = True
        print class_name
        print(inds)

        for i in inds:
            #print inds
            bbox = dets[i, :4]
            score = dets[i, -1]

            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=3.5)
            )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f}'.format(class_name, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')

    ax.set_title(('object detections with '
                'p(object | box) >= {:.1f}').format(CONF_THRESH), fontsize=14)

    if CONTAIN_OBJ:
        plt.axis('off')
        plt.tight_layout()
        directory = args.model_dir + 'prediction'
        if not os.path.exists(directory):
            os.makedirs(directory)
        print('printing to {}'.format(directory + '/' + 'pred_' + image_name))
        #plt.figure(figsize=(800 / my_dpi, 800 / my_dpi), dpi=my_dpi)
        plt.savefig(directory + '/' + 'pred_' + image_name)
        plt.close()
        #plt.draw()
    plt.close('all')

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Proto directory',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--mddir', dest='model_dir', help='model directory',
                        choices=NETS.keys(), default='output/faster_rcnn_end2end/voc_2007_train')
    parser.add_argument('--dtdir', dest='data_dir', help='images directory',
                        choices=NETS.keys(), default='../preprocessing/DOTA/dota/original/patchs/test/images')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join('models/pascal_voc', NETS[args.demo_net][0] , 'faster_rcnn_end2end-24anchors', 'test.prototxt')
    caffemodel = os.path.join(args.model_dir, NETS[args.demo_net][1])


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

    print '\n\nLoaded network {:s}'.format(caffemodel)
    im_names = [name for name in os.listdir(args.data_dir) if name.endswith('.JPG')]
    print im_names
    for cls_ind, class_name in enumerate(CLASSES[1:]):
        print(class_name)
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demoCombined(net, im_name)

    #plt.show()
