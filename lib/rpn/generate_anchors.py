# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

#array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

# after adding more scales
'''
[ 1  2  3  4  8 16 32 64]
[ 1  2  3  4  8 16 32 64]
[ 1  2  3  4  8 16 32 64]
[ 1  2  3  4  8 16 32 64]

without 64
[[  -3.5    2.    18.5   13. ]
 [ -15.    -4.    30.    19. ]
 [ -26.5  -10.    41.5   25. ]
 [ -38.   -16.    53.    31. ]
 [ -84.   -40.    99.    55. ]
 [-176.   -88.   191.   103. ]
 [-360.  -184.   375.   199. ]
 [   0.     0.    15.    15. ]
 [  -8.    -8.    23.    23. ]
 [ -16.   -16.    31.    31. ]
 [ -24.   -24.    39.    39. ]
 [ -56.   -56.    71.    71. ]
 [-120.  -120.   135.   135. ]
 [-248.  -248.   263.   263. ]
 [   2.5   -3.    12.5   18. ]
 [  -3.   -14.    18.    29. ]
 [  -8.5  -25.    23.5   40. ]
 [ -14.   -36.    29.    51. ]
 [ -36.   -80.    51.    95. ]
 [ -80.  -168.    95.   183. ]
 [-168.  -344.   183.   359. ]]

'''
# add more anchors by adding more scales and ratios
#def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
#                     scales=2**np.arange(3, 6)):

def generate_anchors(base_size=8, ratios=[0.5, 1, 2],
                 scales= np.append(np.append(np.arange(1, 5), 2**np.arange(3, 8)), 224)):


    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
    return anchors

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print time.time() - t
    print a
    from IPython import embed; embed()
