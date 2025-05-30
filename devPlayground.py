#%%
import tifffile as tif
import numpy as np
import pandas as pd
import os, glob, pickle
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.signal import fftconvolve
import cv2
from extractResGalvoROIs import make_annotation_tif
#%%
imageA = tif.imread('C:/temp/slice24.tif')
imageA = cv2.normalize(imageA, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
imageB = tif.imread('C:/temp/AVG_rT1_C1_ch2.tif')
imageB = cv2.normalize(imageB, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

#detect keyhpoints and descriptors
# orb = cv2.ORB_create(10000)
akaze = cv2.AKAZE_create()
kpA, desA = akaze.detectAndCompute(imageA, None)
kpB, desB = akaze.detectAndCompute(imageB, None)

#match features
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(desA, desB)
# matches = bf.knnMatch(desA, desB, k=2)
matches = sorted(matches, key=lambda x: x.distance)

#estimate an affube trabsfirnatuib (or homograpghy)
#location of good matches
ptsA = np.float32([kpA[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
ptsB = np.float32([kpB[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

#compute affine or homography matrix
matrix, mask = cv2.estimateAffinePartial2D(ptsA,ptsB) # Returns 2x3 affine matrix

#warp imageA to match imageB
alignedA = cv2.warpAffine(imageA, matrix, (imageB.shape[1], imageB.shape[0]))

#visualize
cv2.imshow('aligned imageA', alignedA)
cv2.imshow('image b', imageB)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
#if zoomed in a lot, then template matching is better than this method. therefore you will
#need to extract zoom factor for each trial and figure out a logic to use template matching
#vs affine transformation (i.e. if matches == 0 then do template matching)

#according to chat gpt
#do AKAZE FIRST
#2 metrics:
#if numMatches >= threshold
#and
#if inlier_ratio = np.sum(mask) / len(matches) > 0.5

#personally also include (if metadata reports zoom > 2x ->4x, then do template matching)

#%%

def make_annotation_tif(mIM, gcampSlice, wgaSlice, threshold, annTifFN, resolution):
    
    #zstack will always be at 2x

    if resolution[0] != mIM.shape[0]:
        mIM = resize(mIM, (resolution[0], resolution[1]), preserve_range=True, anti_aliasing=True)

    '''

    using cv2 since there is a possibility of scaling issues

    padding = (
        (np.abs(shifts[0])+25,np.abs(shifts[0])+25),#x shifts
        (np.abs(shifts[1])+25,np.abs(shifts[1])+25) #y shifts
    )
    paddedIM = np.pad(mIM, padding, mode='constant', constant_values=0)
    paddedWGASlice = np.pad(wgaSlice, padding, mode='constant', constant_values=0)
    paddedGCaMPSlice = np.pad(gcampSlice, padding, mode='constant', constant_values=0)

    correctedIM = np.roll(paddedIM, (-shifts[1],-shifts[0]), axis=(0,1))   

    annTiff = np.stack((paddedWGASlice, paddedGCaMPSlice, correctedIM), axis=0)
    '''

    gcampSlice = cv2.normalize(gcampSlice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    wgaSlice = cv2.normalize(wgaSlice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mIM = cv2.normalize(mIM, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    akaze = cv2.AKAZE_create()
    kpA, desA = akaze.detectAndCompute(gcampSlice, None)
    kpB, desB = akaze.detectAndCompute(mIM, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desA, desB)
    if len(matches) > threshold:
        matches = sorted(matches, key=lambda x: x.distance)
        ptsA = np.float32([kpA[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        ptsB = np.float32([kpB[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        matrix, mask = cv2.estimateAffinePartial2D(ptsA,ptsB)
        alignedgCaMPStack = cv2.warpAffine(gcampSlice, matrix, (mIM.shape[1], mIM.shape[0]))
        alignedgWGAStack = cv2.warpAffine(wgaSlice, matrix, (mIM.shape[1], mIM.shape[0]))
    #case 1, affine transformation
    else:
        print('Case 2 Alignment ... work in progress')
    #case 2 template matching
    annTiff = np.stack((alignedgWGAStack, alignedgCaMPStack, mIM), axis=0)
    tif.imwrite(annTifFN,annTiff)
    return annTiff
mIM = tif.imread('C:/temp/mIM.tif')
wgaStack = tif.imread('C:/temp/wgaStack.tif')
gCaMPStack = tif.imread('C:/temp/gCaMPStack.tif')
test = make_annotation_tif(mIM, gCaMPStack, wgaStack, 5, 'C:/temp/annTif.tif', wgaStack.shape)
# %%
