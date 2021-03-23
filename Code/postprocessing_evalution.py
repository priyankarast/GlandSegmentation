# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 22:48:52 2020

@author: livep
"""

############# Validation using Test Data - A and B
from keras.models import load_model
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import tensorflow as tf
import keras.backend as K
from skimage import io  
import scipy
from metrics import ObjectHausdorff,F1score,ObjectDice
from helpfns import preprocessimg,preprocessmask,connectedcomplabel,remove_small_objects



def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

smooth = 1.

def my_loss(y_true,y_pred):
    bce = tf.keras.losses.BinaryCrossentropy() 
    return bce(y_true, y_pred)

#loading the model
modelname = 'Arch5_Unet_480_v8.h5' 
path = 'C:\\Users\\livep\\OneDrive\\Desktop\\' #Path where model is saved
modelpath = os.path.join(path,modelname)

model = load_model(modelpath, custom_objects={'dice_coef': dice_coef, 
                                              'dice_coef_loss':dice_coef_loss, 
                                              'my_loss': my_loss})
model.summary()

#############Prediction on test set; 1. result visualization

predictedmaskpath = 'Modelv8_postprocess'  
testdir = 'testA'
testdirpath = os.path.join(path,testdir)  #reading CSV file- TestA.csv
testdf = pd.read_csv("%s.csv" %(testdir))  
testimg_name = testdf['name']
 
    
testimg_name = testimg_name.tolist()
 
for imgname in testimg_name:
    print(imgname)
    processedimgarr = preprocessimg(imgname,testdirpath) #preprocess the test image

    #preprocess the GroundTruth
    y_true = preprocessmask(imgname,testdirpath) 
    y_true = np.squeeze(y_true)
    y_true_labelled = connectedcomplabel(y_true) #generate the labelled y_true mask

    #prediction from the model for the test img
    y_pred = model.predict(processedimgarr,verbose=1)
    y_pred_t = (y_pred > 0.5).astype(np.uint8) #thresholding
    y_pred_t = np.squeeze(y_pred_t)
    
    #Saving the thresholded binary predicted mask in the folder
    predmaskname = imgname+'_th_pred.bmp'
    plt.imsave(os.path.join(predictedmaskpath, predmaskname),y_pred_t) 

    #PostProcessing over predicted thresholded binary mask  
    y_pred_rsobj,n = remove_small_objects(y_pred_t,1500)
    y_pred_rso_bfh = scipy.ndimage.morphology.binary_fill_holes(y_pred_rsobj).astype('uint8')
    
    #Saving the labelled predicted mask in the folder
    predmask_label_name = imgname+'_labelled_pred_pp.bmp'
    y_pred_labelled_mask = connectedcomplabel(y_pred_rso_bfh)
    plt.imsave(os.path.join(predictedmaskpath, predmask_label_name),y_pred_labelled_mask) 
    
    p = np.squeeze(y_true_labelled)
    y = np.squeeze(y_pred_labelled_mask)

    objF1 = F1score(p,y)
    print(objF1)

    objDice = ObjectDice(p,y)
    print(objDice)

    objDist = ObjectHausdorff(p,y)
    print(objDist) 

    testdf.loc[testdf.name == imgname,
           ["F1score","ObjectDice","ObjectHausdorff"]]=objF1,objDice,objDist
    testdf['imageno'] = testdf.name.str.split('_').str[1]           
testdf.to_csv("%s\\%s_Output%s.csv" %(predictedmaskpath,testdir,modelname))           


#Visualization - Sanity Check
plt.imshow(processedimgarr[0,:,:,:])
plt.show()
plt.imshow(y_true[0,:,:,:])
plt.show()
plt.imshow(y_pred[0,:,:,:])
plt.show()
plt.imshow(y_pred_t[0,:,:,:])
plt.show()

####################### Accuracy on test set
results = model.evaluate(processedimgarr,y_true)
print("test loss, test acc:", results)



#Onetime activity - Component labelling of all the true masks in the TestA datset
labelledmaskpath = 'LabelledMasks\\TestA_labelled'
for img in testimg_name:
    y_true = preprocessmask(img)
    labelledmaskname = img + '_anno_labelled.bmp'
    labelledmask = connectedcomplabel(y_true[0,:,:,0])
    plt.imsave(os.path.join(labelledmaskpath, labelledmaskname),labelledmask) 


#Morphology operations - Post Processing operations - applied to some of the images

kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(preds_val_t[ix], cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(preds_val_t[ix], cv2.MORPH_CLOSE, kernel)
dialation = cv2.dilate(closing,kernel,iterations = 2)


