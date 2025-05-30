#%%
import tifffile as tif
import numpy as np
import matplotlib.pyplot as plt
import os, glob, pickle, cv2
import jnormcorre
import jnormcorre.motion_correction
import jnormcorre.utils.registrationarrays as registrationarrays
from skimage.transform import resize
from scipy.signal import fftconvolve
import pandas as pd

#does not work with cellpose since cellpose needs python3.10, this is python 3.13
#run in dataAnalysis env

def register_tSeries(rawData, regParams, compareRaw=False):

    max_shifts = regParams['maxShifts']
    frames_per_split = regParams['frames_per_split']
    num_splits_to_process_rig = regParams['num_splits_to_process_rig']
    niter_rig = regParams['niter_rig']
    save_movie = regParams['save_movie']
    pw_rigid = regParams['pw_rigid']
    strides = regParams['strides']
    overlaps = regParams['overlaps']
    max_deviation_rigid = regParams['max_deviation_rigid']

    template = np.nanmean(rawData[:,:,:], axis=0) #use mean image to register

    #template used in motion corrector object, which
    corrector = jnormcorre.motion_correction.MotionCorrect(rawData, max_shifts=max_shifts, frames_per_split=frames_per_split,
                                                    num_splits_to_process_rig=num_splits_to_process_rig, strides=strides,
                                                        overlaps=overlaps, 
                                                        max_deviation_rigid = max_deviation_rigid, niter_rig=niter_rig,
                                                        pw_rigid = pw_rigid)
    frame_corrector, _ = corrector.motion_correct(template=template, save_movie=save_movie)
    
    frame_corrector.batching = regParams['frame_corrector_batching']
    motionCorrectedData = registrationarrays.RegistrationArray(frame_corrector, rawData, pw_rigid=False)

    #Flag for if user wants to compare raw data to registered data for visual inspection
    if compareRaw:
        from fastplotlib.widgets import ImageWidget
        iw = ImageWidget(data = [rawData, motionCorrectedData], 
                        names=["Raw", "Motion Corrected"], 
                        histogram_widget = True)
        iw.show()

    #Come up with statistic that describes how much motion correction occured (indirectly assess experiment quality)
    avgMoveRaw = np.nanmean(np.nanstd(rawData, axis=0))
    avgMoveReg = np.nanmean(np.nanstd(motionCorrectedData[:], axis=0))
    prepQuality = avgMoveRaw/avgMoveReg

    return motionCorrectedData, prepQuality

def generate_shifts(mIM, gcampSlice, trial):
    shiftTolerance = 100
    correlation = fftconvolve(gcampSlice, mIM[::-1, ::-1], mode='same')
    yShift, xShift = np.unravel_index(np.argmax(correlation), correlation.shape)
    centerY, centerX = np.array(correlation.shape) // 2
    dy = centerY - yShift
    dx = centerX - xShift
    if dx > shiftTolerance or dy > shiftTolerance:
        dx = 0
        dy = 0 
        plt.figure()
        plt.imshow(correlation)
        plt.show()
        plt.title(f'Trial {trial+1} exceeds shift tolerance! Review Manually!')
        plt.figure()
        plt.imshow(mIM)
        plt.title(f'Trial {trial+1} mIM')
        plt.show()
        plt.figure()
        plt.imshow(gcampSlice)
        plt.title(f'Trial {trial+1} slice')
        plt.show()
    return (dx,dy)

def make_annotation_tif(mIM, gcampSlice, wgaSlice, threshold, annTifFN, resolution):
    
    #zstack will always be at 2x
    low, high = np.percentile(gcampSlice, [0.4, 99.6]) #adjust contrast for best outcome
    gcampSlice = np.clip((gcampSlice - low) / (high - low), 0, 1) * 255

    if resolution[0] != mIM.shape[0]:
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

'''
TODO:
    -  make gate based on WGA conjugate
    -  make method for determining best slice
'''

def read_single_expmt_data(expmtPath, regParams, galvo='mechanical', dev=False):
    '''
        expmtPath: directory experiment data is in

    '''
    #reading experiment notes
    expmtSummary = {}; hasZstack = False
    bestSlices = []; summaryTiffs = []; trialShifts = []; amountCorrected = []
    trialCounter = 0
    trialList = glob.glob(expmtPath+'/TSeries*')
    expmtNotes = pd.read_excel(glob.glob(expmtPath+'/expmtNotes*.xlsx')[0])
    sliceLabels = expmtNotes['slice_label'].values
    labelUsed = expmtNotes['lung_label'].values[0]
    
    #checking for zstack
    if len(glob.glob(expmtPath+'/ZSeries*'))>0:
        #only have zstacks for WGA594 experiments currently
        hasZstack = True
        redStackPath = glob.glob(expmtPath+'/ZSeries*/*Ch1*.tif')[0]
        greenStackPath = glob.glob(expmtPath+'/ZSeries*/*Ch2*.tif')[0]
        wgaStack = tif.imread(redStackPath)
        gcampStack = tif.imread(greenStackPath)
        segPath = glob.glob(expmtPath+'/ZSeries*')[0]
        if os.path.exists(segPath + f'/segmentation/WGA_manual/'):
            segmentationsMade = glob.glob(segPath + f'/segmentation/WGA_manual/*.npy')
            print(f'{len(segmentationsMade)}/{len(trialList)} segmentations made')
        else:
            os.makedirs(segPath + '/segmentation/WGA_manual')

    #registering tif of each trial (motion correction)
    for trialPath in trialList:
        if dev:
            print('Running Test Registration')
            trialTiff = glob.glob(trialPath+'/*.tif')[-1] #for now just make sure we read 2nd channel
            loadedTiff = tif.imread(trialTiff)
            print(loadedTiff.shape)
            registeredTSeries, quality = register_tSeries(loadedTiff, regParams, compareRaw=False)
            registeredTSeries = np.where(registeredTSeries[:]>64000, 0, registeredTSeries[:])
            plt.imshow(np.nanmean(registeredTSeries[:], axis=0))
            plt.show()
            tif.imwrite(expmtPath+f'/prevRegistration/T{trialCounter+1}_reg.tif', registeredTSeries[:])
            if trialCounter==5:
                return
        else:
            #check if already registered with jnormcorre, if not register
            if os.path.exists(trialPath + f'/T{trialCounter+1}_registered.tif'):
                print(f'Trial {trialCounter+1} Already Registered!')
                registeredTSeries = tif.imread(trialPath + f'/T{trialCounter+1}_registered.tif')
                registeredTSeries_fixed = np.where(registeredTSeries[:]>60000, 0, registeredTSeries[:])
            else:
                print(f'Registering Trial {trialCounter+1}')
                #Load in trial data
                trialTiff = glob.glob(trialPath+'/*.tif')[-1] #for now just make sure we read 2nd channel
                loadedTiff = tif.imread(trialTiff)
                if hasZstack:
                    if wgaStack.shape[1] != loadedTiff.shape[1]:
                        #segmentations are on zstack, so match tseries resolution to zstack
                        loadedTiff_us = resize(loadedTiff[:], output_shape=(loadedTiff.shape[0], wgaStack.shape[1], wgaStack.shape[2]), preserve_range=True, anti_aliasing=True)
                    else:
                        loadedTiff_us = loadedTiff
                else:
                    #only need to register if no zstack taken
                    loadedTiff_us = loadedTiff
                registeredTSeries, quality = register_tSeries(loadedTiff_us, regParams, compareRaw=False)
                registeredTSeries_fixed = np.where(registeredTSeries[:]>60000, 0, registeredTSeries[:]) #necessary to patch some artifacts from motion correction
                amountCorrected.append(quality)
                print('Registered Data:', registeredTSeries_fixed.shape)
                tif.imwrite(trialPath + f'/T{trialCounter+1}_registered.tif', registeredTSeries_fixed)

        if hasZstack:
            bestSlice = sliceLabels[trialCounter] #TODO: make method for determining best slice

        #segmentation from stack only for WGA594
        if 'WGA594' in labelUsed:
            if bestSlice<0:
                bestSlices.append(np.nan) 
                summaryTiffs.append(np.nan)
                trialShifts.append(np.nan)
            else:
                if len(glob.glob(segPath + f'/segmentation/WGA_manual/t{trialCounter}*.tif'))>0:
                    segmentationFN = glob.glob(segPath+f'/segmentation/WGA_manual/t{trialCounter}_*seg.npy')
                    if len(segmentationFN) == 1:
                        print(f'Trial {trialCounter+1} segmented!')
                    else:
                        print(f'Trial {trialCounter+1} is ready to segment!')
                else:
                    print(f'Preparing Annotation Tif for:')
                    #pre-segmentation path
                    annTifFN = segPath + f'/segmentation/WGA_manual/t{trialCounter}_Ann_slice{bestSlice}.tif'
                    mIM = np.nanmean(registeredTSeries_fixed, axis=0)
                    print(mIM.shape)
                    gcampSlice = gcampStack[bestSlice,:,:]
                    shifts = generate_shifts(mIM, gcampSlice, trialCounter)
                    annTiff = make_annotation_tif(expmtPath, mIM, gcampSlice, wgaStack[bestSlice,:,:], shifts, bestSlice, trialCounter,annTifFN)
                    bestSlices.append(bestSlice)
                    summaryTiffs.append(mIM)
                    trialShifts.append(shifts)
        else:
            if dev:
                print('Not caring about ann tiffs')
            else:
                print('Not using zstack for annotation, annotTif is with TSeries')
                if os.path.exists(expmtPath + '/segmentations/manual/'):
                    segmentationsMade = glob.glob(expmtPath + '/segmentations/manual/*.npy')
                else:
                    os.makedirs(expmtPath + '/segmentations/manual/')
                annTiff = np.nanmean(registeredTSeries_fixed, axis=0)
                print('Writing Tif', annTiff.shape)
                tif.imwrite(expmtPath + f'/segmentations/manual/T{trialCounter+1}_annTiff.tif', annTiff)
                print(f'{len(segmentationsMade)}/{len(trialList)} segmentations made')
        trialCounter+=1
    
    #save experimental summaries from registration
    expmtSummary['bestSlice'] = bestSlices
    expmtSummary['mIM'] = summaryTiffs
    expmtSummary['shifsFromStack'] = trialShifts
    # expmtSummary['registrationQuality'] = amountCorrected

    return expmtSummary



def register_mech_galvo_trials(expmtPath, regParams):
    trialPaths = glob.glob(expmtPath+'/TSeries*')
    try:
        expmtNotes = pd.read_excel(glob.glob(expmtPath+'/expmtNotes*')[0])
    except IndexError:
        print('Need to create expmtNotes for experiment! exiting...')
        return
    zSeriesPathWGA = glob.glob(expmtPath+'/ZSeries*/*Ch1*.tif')[0]
    zSeriesPathGCaMP = glob.glob(expmtPath+'/ZSeries*/*Ch2*.tif')[0]
    slices = expmtNotes['slice_label'].values
    trialCounter = 1
    for trial in trialPaths:
        trialIDX = trialCounter-1
        trialCycles_ch1 = glob.glob(trial+'/TSeries*Ch1*.tif')
        trialCycles_ch2 = glob.glob(trial+'/TSeries*Ch2*.tif')
        registeredTrials = glob.glob(trial+'/rT*_C*_ch*.tif')
        if len(registeredTrials) <1:
            print(f'Reading Trial {trialCounter} Cycles...')
            for cycleIDX in range(len(trialCycles_ch1)):
                print('Cycle', cycleIDX, 'of', len(trialCycles_ch1))
                cycleTiff_ch1 = tif.imread(trialCycles_ch1[cycleIDX])
                cycleTiff_ch2 = tif.imread(trialCycles_ch2[cycleIDX])
                #make annotation tiffs here
                if expmtNotes['lung_label'].values[0] == 'WGA594':
                    registeredCycle_ch1, _ = register_tSeries(cycleTiff_ch1, regParams)
                    correctedRegisteredCycle_ch1 = np.where(registeredCycle_ch1[:]>60000, 0, registeredCycle_ch1[:])
                    registeredCycle_ch2, _ = register_tSeries(cycleTiff_ch2, regParams)
                    correctedRegisteredCycle_ch2 = np.where(registeredCycle_ch2[:]>60000, 0, registeredCycle_ch2[:])
                    mIM = np.nanmean(cycleTiff_ch2, axis=0)
                    wgaZStack = tif.imread(zSeriesPathWGA)
                    gcampZStack = tif.imread(zSeriesPathGCaMP)
                    trialSlice = slices[trialIDX]
                    wgaSlice = wgaZStack[trialSlice,:,:]
                    gcampSlice = gcampZStack[trialSlice,:,:]
                    resolution = gcampSlice.shape
                    # shifts = generate_shifts(mIM, gcampSlice, trial)
                    if os.path.exists(expmtPath+'/segmentations/WGA_manual/'):
                        annTiffFN = expmtPath+f'/segmentations/WGA_manual/AVG_rT{trialCounter}_C{cycleIDX+1}_ch2.tif'
                    else:
                        os.mkdir(expmtPath+'/segmentations/')
                        os.mkdir(expmtPath+'/segmentations/WGA_manual')
                        annTiffFN = expmtPath+f'/segmentations/WGA_manual/AVG_rT{trialCounter}_C{cycleIDX+1}_ch2.tif'
                    #resizing so segmentation masks match zstack resolution
                    correctedRegisteredCycle_ch1 = resize(correctedRegisteredCycle_ch1[:], output_shape=(correctedRegisteredCycle_ch1.shape[0], resolution[0], resolution[1]), preserve_range=True, anti_aliasing=True)
                    correctedRegisteredCycle_ch2 = resize(correctedRegisteredCycle_ch2[:], output_shape=(correctedRegisteredCycle_ch1.shape[0], resolution[0], resolution[1]), preserve_range=True, anti_aliasing=True)
                    if cycleIDX == 0:
                        _ = make_annotation_tif(mIM, gcampSlice, wgaSlice, 25, annTiffFN, resolution)
                elif expmtNotes['lung_label'].values[0] == 'WGATR':
                    registeredCycle_ch1, _ = register_tSeries(cycleTiff_ch1, regParams)
                    correctedRegisteredCycle_ch1 = np.where(registeredCycle_ch1[:]>60000, 0, registeredCycle_ch1[:])
                    registeredCycle_ch2, _ = register_tSeries(cycleTiff_ch2, regParams)
                    correctedRegisteredCycle_ch2 = np.where(registeredCycle_ch2[:]>60000, 0, registeredCycle_ch2[:])
                    if os.path.exists(expmtPath+'/segmentations/WGA_manual/'):
                        annTiffFN = expmtPath+f'/segmentations/WGA_manual/AVG_rT{trialCounter}_C{cycleIDX+1}_ch2.tif'
                    else:
                        os.mkdir(expmtPath+'/segmentations/')
                        os.mkdir(expmtPath+'/segmentations/WGA_manual')
                        annTiffFN = expmtPath+f'/segmentations/WGA_manual/AVG_rT{trialCounter}_C{cycleIDX+1}_ch2.tif'
                    tif.imwrite(annTiffFN, mIM)

                tif.imwrite(trial+f'/rT{trialCounter}_C{cycleIDX+1}_ch1.tif', correctedRegisteredCycle_ch1[:])
                tif.imwrite(trial+f'/rT{trialCounter}_C{cycleIDX+1}_ch2.tif', correctedRegisteredCycle_ch2[:])
        else:
            print(f'Trial {trialCounter} is registered!')
        trialCounter+=1
#%% Run Processing Pipeline
if __name__=='__main__':
    dataFrom = [
        'U:/expmtRecords/Lucas*',
        'C:/Analysis/april_data/Lucas*',
        'U:/expmtRecords/res_galvo/Lucas*'
        ]
    regParams = {
        'maxShifts': (25,25),
        'frames_per_split': 1000, 
        'num_splits_to_process_rig': 5,
        'niter_rig': 4,
        'save_movie': False,
        'pw_rigid': True,
        'strides': (64, 64),
        'overlaps': (32,32),
        'max_deviation_rigid': 25,
        'frame_corrector_batching': 100
        }
    expmtRecords = glob.glob(dataFrom[2])
    for expmt in expmtRecords:
        print('Working on', expmt)
        dataDict = register_mech_galvo_trials(expmt,regParams)
        with open(f'{expmt}/expmtSummary.pkl', 'wb') as f:
            pickle.dump(dataDict, f)
    print('Done')
