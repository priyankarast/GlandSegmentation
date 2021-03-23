# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:58:00 2020

@author: livep
"""
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology, skimage.data

# the task is to read every img, predict the semantic mask and evaluate it and store it in the dataframe
def preprocessimg(imgname,testdirpath):
    img = cv2.imread(os.path.join(testdirpath,imgname+'.bmp'))
    img = cv2.resize(img, (480,480),interpolation = cv2.INTER_NEAREST)
    img = np.expand_dims(img,axis = 0)
    img = img.astype('float32')
    img /= 255.
    return img

def preprocessmask(imgname,testdirpath):
    maskname = imgname.split('.')[0] + '_anno.bmp' 
    mask = cv2.imread(os.path.join(testdirpath,maskname),cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (480,480))
    mask[mask != 0] = 1
    #mask = np.expand_dims(mask,axis = 0)
    #mask = np.expand_dims(mask,axis = 3)
    #mask = mask.astype('float32')
    return mask

#============= Connected Component labelling
def connectedcomplabel(img):
    #img = preds_val_t[:len(preds_val_t)][ix]
    num_labels, labels = cv2.connectedComponents(img)

    # Map component labels to hue val, 0-179 is the hue range in OpenCV
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Converting cvt to BGR
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    
    # Showing Original Image
    #plt.imshow(np.squeeze(img))
    #plt.axis("off")
    #plt.title("Orginal Image")
    #plt.show()
    
    #Showing Image after Component Labeling
    plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Image after Component Labeling")
    plt.show()
    return labeled_img

def remove_small_objects(img, min_size=150):
        # find all your connected components (white blobs in your image)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
        # connectedComponentswithStats yields every seperated component with information on each of them, such as size
        # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        # your answer image
        img2 = img
        # for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):
            if sizes[i] < min_size:
                img2[output == i + 1] = 0

        return img2,nb_components


def fill_holes(imgpath):
    img = skimage.data.imread(imgpath,1)
    labels = skimage.morphology.label(img)
    labelCount = np.bincount(labels.ravel())
    background = np.argmax(labelCount)
    img[labels != background] = 255
    return img

def addmask(y_true):
    not_mask = cv2.bitwise_not(y_true)
    not_mask = np.expand_dims(not_mask, axis=2)
    y_true = np.expand_dims(y_true, axis=2)
    im_softmax = np.concatenate([not_mask, y_true], axis=2)
    print(im_softmax.shape)
    return im_softmax
    
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, unary_from_softmax    
    
def crf(inpimg, im_pred):
    n_classes = 2
    feat_first = im_pred.transpose((2, 0, 1)).reshape((n_classes,-1))
    print(feat_first.shape)
    unary = unary_from_softmax(feat_first)
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF2D(inpimg.shape[1], inpimg.shape[0], n_classes)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=(5, 5), compat=10, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    d.addPairwiseBilateral(sxy=(10, 10), srgb=(13, 13, 13), rgbim=inpimg,
                       compat=10,
                       kernel=dcrf.DIAG_KERNEL,
                       normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)
    res = np.argmax(Q, axis=0).reshape((inpimg.shape[0], inpimg.shape[1]))
    print(res.shape) 
    return res
