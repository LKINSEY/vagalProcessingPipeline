#%%
import tifffile as tif
import numpy as np
import pandas as pd
import os, glob, pickle, cv2
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.signal import fftconvolve
from extractResGalvoROIs import make_annotation_tif

#script for correcting annotation tiffs
expmtList = [
    'U:/expmtRecords/res_galvo/Lucas_250519_001',
    'U:/expmtRecords/res_galvo/Lucas_250521_001',
    'U:/expmtRecords/res_galvo/Lucas_250523_001'
]

def make_annotation_tif(mIM, gcampSlice, wgaSlice, threshold, annTifFN, resolution):
    
    #zstack will always be at 2x
    # gcampSlice = auto_contrast(gcampSlice)
    low, high = np.percentile(gcampSlice, [0.4, 99.6])
    gcampSlice = np.clip((gcampSlice - low) / (high - low), 0, 1) * 255

    if resolution[0] != mIM.shape[0]:
        print('flag!')
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
    print('wtf is going on')
    tif.imwrite('C:/temp/wild1.tif', mIM)
    tif.imwrite('C:/temp/wild2.tif', gCaMPSlice)
    tif.imwrite('C:/temp/wild3.tif',wgaSlice)
    print('do the thing')
    akaze = cv2.AKAZE_create()
    kpA, desA = akaze.detectAndCompute(gcampSlice, None)
    kpB, desB = akaze.detectAndCompute(mIM, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desA, desB)
    if len(matches) > threshold:
        #case 1, affine transformation
        matches = sorted(matches, key=lambda x: x.distance)
        ptsA = np.float32([kpA[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        ptsB = np.float32([kpB[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        matrix, mask = cv2.estimateAffinePartial2D(ptsA,ptsB)
        alignedgCaMPStack = cv2.warpAffine(gcampSlice, matrix, (mIM.shape[1], mIM.shape[0]))
        alignedgWGAStack = cv2.warpAffine(wgaSlice, matrix, (mIM.shape[1], mIM.shape[0]))
    else:
        print('Case 2 Alignment ... work in progress')
        #case 2 template matching
    annTiff = np.stack((alignedgWGAStack, alignedgCaMPStack, mIM), axis=0)
    tif.imwrite(annTifFN,annTiff)
    return annTiff


for expmt in expmtList:
    print('Experiment:', expmt)
    notesPath = glob.glob(expmt+'/expmtNotes*')[0]
    notes = pd.read_excel(notesPath)
    zStacks_red = glob.glob(expmt+'/ZSeries*/*Ch1*.tif')
    zStacks_green = glob.glob(expmt+'/ZSeries*/*Ch2*.tif')
    trialSliceMatch = notes['slice_label'].values
    trialPaths = glob.glob(expmt+'/TSeries*')
    for trialIDX in range(len(trialPaths)):
        print('trial ', trialIDX)
        trialTifPath = glob.glob(trialPaths[trialIDX]+f'/rT{trialIDX+1}_C1_*ch2*.tif')[0] #just takes first cycle if multiple cycles
        bestSlice = trialSliceMatch[trialIDX]
        print('loading...')
        trialTifLoaded = tif.imread(trialTifPath)
        wgaStackLoaded = tif.imread(zStacks_red)
        gCaMPStackLoaded = tif.imread(zStacks_green)
        mIM = np.nanmean(trialTifLoaded, axis=0)
        gCaMPSlice = gCaMPStackLoaded[bestSlice,:,:]
        print(gCaMPSlice.shape)
        wgaSlice = wgaStackLoaded[bestSlice,:,:]
        annTiffFN = expmt+f'/segmentations/WGA_manual/AVG_T{trialIDX}_C1_ch2_slice{bestSlice}.tif'
        print('making tiff')

        result = make_annotation_tif(mIM, gCaMPSlice, wgaSlice, 5, annTiffFN, wgaSlice.shape)

print('wtf')


# %%
