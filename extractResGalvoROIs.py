#%%
import tifffile as tif
import numpy as np
import pandas as pd
import os, glob, pickle, cv2
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy.signal import fftconvolve
#heavily relies on jnormcorre!
import jnormcorre
import jnormcorre.motion_correction
import jnormcorre.utils.registrationarrays as registrationarrays
# run in dataAnalysis env

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

'''
on hold...


# def generate_shifts(mIM, gcampSlice, trial):
#     shiftTolerance = 100
#     correlation = fftconvolve(gcampSlice, mIM[::-1, ::-1], mode='same')
#     yShift, xShift = np.unravel_index(np.argmax(correlation), correlation.shape)
#     centerY, centerX = np.array(correlation.shape) // 2
#     dy = centerY - yShift
#     dx = centerX - xShift
#     if dx > shiftTolerance or dy > shiftTolerance:
#         dx = 0
#         dy = 0 
#         plt.figure()
#         plt.imshow(correlation)
#         plt.show()
#         plt.title(f'Trial {trial+1} exceeds shift tolerance! Review Manually!')
#         plt.figure()
#         plt.imshow(mIM)
#         plt.title(f'Trial {trial+1} mIM')
#         plt.show()
#         plt.figure()
#         plt.imshow(gcampSlice)
#         plt.title(f'Trial {trial+1} slice')
#         plt.show()
#     return (dx,dy)
'''

def make_annotation_tif(mIM, gcampSlice, wgaSlice, threshold, annTifFN, resolution):
    
    #zstack will always be at 2x
    low, high = np.percentile(gcampSlice, [0.4, 99.6]) #adjust contrast for best outcome
    gcampSlice = np.clip((gcampSlice - low) / (high - low), 0, 1) * 255

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

def register_res_galvo_trials(expmtPath, regParams):
    trialPaths = glob.glob(expmtPath+'/TSeries*')
    zSeriesPathWGA = glob.glob(expmtPath+'/ZSeries*/*Ch1*.tif')[0]
    zSeriesPathGCaMP = glob.glob(expmtPath+'/ZSeries*/*Ch2*.tif')[0]
    try:
        expmtNotes = pd.read_excel(glob.glob(expmtPath+'/expmtNotes*')[0])
    except IndexError:
        print('Need to create expmtNotes for experiment! exiting...')
        return
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

def extract_res_roi_traces(expmtPath):
    print(expmt)
    pad = 25 #pad for roi windows
    #currently only for texas red labeling
    trialPaths = glob.glob(expmtPath+'/TSeries*')    
    try:
        expmtNotes = pd.read_excel(glob.glob(expmtPath+'/expmtNotes*')[0])
    except IndexError:
        print('Need to create expmtNotes for experiment! exiting...')
        return
    slicePerTrial = expmtNotes['slice_label'].values
    trialCounter = 0
    dataDict = {}
    numSegmentations = glob.glob(expmtPath+f'/segmentations/WGA_manual/*.npy')
    if os.path.exists(expmt+'/expmtTraces.pkl'):
        print('Traces Already Extracted')
        return None
    else:
        if len(numSegmentations)>0:
            for trial in trialPaths:
                registeredTiffs_ch1 = glob.glob(trial+'/rT*C*Ch1.tif')
                registeredTiffs_ch2 = glob.glob(trial+'/rT*C*Ch2.tif')
                segmentationUsed = slicePerTrial[trialCounter]
                masksNPY = glob.glob(expmtPath+f'/segmentations/WGA_manual/*slice{segmentationUsed}_seg.npy')
                print(masksNPY)
                segmentationLoaded = np.load(masksNPY[0], allow_pickle=True).item()
                masks = segmentationLoaded['masks']
                if len(masks.shape) == 3:
                    masks = masks[0,:,:]
                rgbIM = np.zeros((3, masks.shape[0], masks.shape[1]))
                rgbIM[1,:,:] = masks
                rois = np.unique(masks)[1:]
                cycleFeatures = {}
                for cycleIDX in range(len(registeredTiffs_ch1)):
                    greenCycle = tif.imread(registeredTiffs_ch2[cycleIDX])
                    if cycleIDX == 0:
                        redCycle = tif.imread(registeredTiffs_ch1[cycleIDX]) 
                        rgbIM[2,:,:] = resize(np.nanmean(greenCycle, axis=0), (masks.shape[0], masks.shape[1]), preserve_range=True, anti_aliasing=True)
                        roiFeatures = {}
                        meanRed = np.nanmean(redCycle, axis=0)
                    cycleTrace = []
                    for roi in rois:
                        if cycleIDX == 0:         #only do this once               
                            xROI, yROI = (masks==roi).nonzero()
                            maxX = np.max(xROI)+pad; minX = np.min(xROI)-pad ; xDiameter = maxX-minX
                            maxY = np.max(yROI)+pad; minY = np.min(yROI)-pad ; yDiameter = maxY - minY
                            roiWindow = rgbIM[:, minY:maxY,minX:maxX]
                            redCell = meanRed * (masks==roi)
                            redCellNAN = np.where(redCell==0, np.nan, redCell)
                            avgRed = np.nanmean(redCellNAN, axis=(0,1))
                            roiFeatures[f'roi{roi}_redAvg'] = avgRed
                            roiFeatures[f'roi{roi}_diameter'] = [xDiameter, yDiameter]
                            roiFeatures[f'roi{roi}_window'] = roiWindow
                        extractedROI = greenCycle*(masks==roi)
                        roiNAN = np.where(extractedROI==0, np.nan, extractedROI)
                        roiTrace = np.nanmean(roiNAN, axis=(1,2))
                        cycleTrace.append(roiTrace)
                    cycleFeatures[f'cycle{cycleIDX}_traces'] = cycleTrace
                cycleFeatures[f'T{trialCounter}_roiFeatures'] = roiFeatures
                dataDict[trialCounter] = cycleFeatures
                trialCounter+=1
        else:
            print('No ROIs segmented yet')
    return dataDict
    
#%%
if __name__=='__main__':
    dataFrom = [
        'U:/expmtRecords/Lucas*',
        'C:/Analysis/april_data/Lucas*',
        'U:/expmtRecords/res_galvo/Lucas*',
        ]
    expmtRecords = glob.glob(dataFrom[2])
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

    for expmt in expmtRecords:
        
        register_res_galvo_trials(expmt, regParams)
        #insert cellpose command here
        dataDict = extract_res_roi_traces(expmt)
        if dataDict:
            with open(expmt+'/expmtTraces.pkl', 'wb') as f:
                pickle.dump(dataDict, f)

# %%
