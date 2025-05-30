#%%
import tifffile as tif
import numpy as np
import pandas as pd
import os, glob, pickle, cv2
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.signal import fftconvolve
from extractResGalvoROIs import make_annotation_tif

expmtList = glob.glob('U:/expmtRecords/res_galvo/Lucas*')

def make_annotation_tif(mIM, gcampSlice, wgaSlice, threshold, annTifFN, resolution):
    
    #zstack will always be at 2x
    # gcampSlice = auto_contrast(gcampSlice)
    low, high = np.percentile(gcampSlice, [0.4, 99.6])
    gcampSlice = np.clip((gcampSlice - low) / (high - low), 0, 1) * 255

    if resolution[0] != mIM.shape[0]:
        print('flag!')
        mIM = resize(mIM, (resolution[0], resolution[1]), preserve_range=True, anti_aliasing=True)


    gcampSlice = cv2.normalize(gcampSlice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
 
    wgaSlice = cv2.normalize(wgaSlice, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mIM = cv2.normalize(mIM, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
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

def count_cells(expmtPath):
    notesPath = glob.glob(expmtPath+'/expmtNotes*')[0]
    notes = pd.read_excel(notesPath)
    # sliceLabels = notes['slice_label'].values
    segmentations = glob.glob(expmtPath+'/cellCountingTiffs/*seg.npy')
    sliceSetIDX = 1
    cellCounts = pd.DataFrame({
            'WGA+/GCaMP-':[],
            'WGA+/GCaMP+': [],
            'WGA-/GCaMP+': [],
            'label':[] ,
            'sliceNum': []
        })
    for annPath in segmentations:
        annotation = np.load(annPath, allow_pickle=True).item()
        masks = annotation['masks']
        wgaOnly_mask = masks[0,:,:]
        wgaOnly = len(np.unique(wgaOnly_mask))-1
        doublePositive_mask = masks[1,:,:]
        doublePositive = len(np.unique(doublePositive_mask))-1
        gcampOnly_mask = masks[2,:,:]
        gcampOnly = len(np.unique(gcampOnly_mask))-1
        data = {
            'WGA+/GCaMP-': wgaOnly,
            'WGA+/GCaMP+': doublePositive,
            'WGA-/GCaMP+': gcampOnly,
            'label': notes['lung_label'].values[0],
            'sliceNum': sliceSetIDX
        }
        cellCounts.loc[len(cellCounts)] = [wgaOnly,doublePositive, gcampOnly, notes['lung_label'].values[0],sliceSetIDX]
        # cellCounts = pd.concat([cellCounts, pd.DataFrame(data)], ignore_index=True)

        sliceSetIDX+=1
    print(
        expmtPath+'\n',
        'Counts: \t WGA+/GCaMP- \t WGA+/GCaMP- \t WGA-/GCaMP+  \n',
        f'\t {np.sum(cellCounts["WGA+/GCaMP-"].values)}',
        f' \t {np.sum(cellCounts["WGA+/GCaMP+"].values)}',
        f' \t {np.sum(cellCounts["WGA-/GCaMP+"].values)}'
    )


for expmt in expmtList:
    count_cells(expmt)


#%%
expmtList = glob.glob('U:/expmtRecords/res_galvo/Lucas*')
for expmt in expmtList:
    print('Experiment:', expmt)
    #for purposes of analysis
    if os.path.exists(expmt+'/cellCountingTiffs/'):
        print('Cells Counted!')
    else:
        notesPath = glob.glob(expmt+'/expmtNotes*')[0]
        notes = pd.read_excel(notesPath)
        label = notes['lung_label'].values[0]
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
            if label == 'WGATR':
                print('WGATR')
                WGAtrialTifPath = glob.glob(trialPaths[trialIDX]+f'/rT{trialIDX+1}_C1_*ch1*.tif')[0]
                wgaLoaded = tif.imread(WGAtrialTifPath)
                wgaSlice = np.nanmean(wgaLoaded, axis=0)
                mIM = np.nanmean(trialTifLoaded, axis=0)
                gCaMPSlice = mIM
            elif label=='WGA594':
                print('WGA594')
                wgaStackLoaded = tif.imread(zStacks_red)
                gCaMPStackLoaded = tif.imread(zStacks_green)
                mIM = np.nanmean(trialTifLoaded, axis=0)
                gCaMPSlice = gCaMPStackLoaded[bestSlice,:,:]
                print(gCaMPSlice.shape)
                wgaSlice = wgaStackLoaded[bestSlice,:,:]
            
            #uncommented only for cell counting
            if os.path.exists(expmt+f'/cellCountingTiffs/'):
                # annTiffFN = expmt+f'/segmentations/WGA_manual/AVG_T{trialIDX}_C1_ch2_slice{bestSlice}.tif'
                annTiffFN = expmt+f'/cellCountingTiffs/cellCounting_T{trialIDX}_slice{bestSlice}.tif'
            else:
                os.mkdir(expmt+f'/cellCountingTiffs/')
                # annTiffFN = expmt+f'/segmentations/WGA_manual/AVG_T{trialIDX}_C1_ch2_slice{bestSlice}.tif'
                annTiffFN = expmt+f'/cellCountingTiffs/cellCounting_T{trialIDX}_slice{bestSlice}.tif'
            print('making tiff')

            result = make_annotation_tif(mIM, gCaMPSlice, wgaSlice, 5, annTiffFN, wgaSlice.shape)




# %%
