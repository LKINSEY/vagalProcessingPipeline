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



def make_annotation_tif(mIM, gcampSlice, wgaSlice, threshold, annTifFN, resolution):
    
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
        return
        #case 2 template matching
    annTiff = np.stack((alignedgWGAStack, alignedgCaMPStack, mIM), axis=0)
    tif.imwrite(annTifFN,annTiff)
    return annTiff

def register_trials(expmtPath, regParams):
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
                cycleTiff_ch2 = tif.imread(trialCycles_ch2[cycleIDX])
                registeredCycle_ch2, _ = register_tSeries(cycleTiff_ch2, regParams)
                correctedRegisteredCycle_ch2 = np.where(registeredCycle_ch2[:]>59000, np.nan, registeredCycle_ch2[:])
                mIM = np.nanmean(correctedRegisteredCycle_ch2, axis=0)    
                trialSlice = slices[trialIDX]                
                if trialSlice == -1:
                        print('passing due to zstack error')
                        break
                if expmtNotes['lung_label'].values[0] == 'WGA594': #uses stack to find WGA
                    wgaZStack = tif.imread(zSeriesPathWGA)
                    gcampZStack = tif.imread(zSeriesPathGCaMP)
                    wgaSlice = wgaZStack[trialSlice,:,:]
                    gcampSlice = gcampZStack[trialSlice,:,:]
                    resolution = gcampSlice.shape
                    correctedRegisteredCycle_ch2 = resize(correctedRegisteredCycle_ch2[:], output_shape=(correctedRegisteredCycle_ch2.shape[0], resolution[0], resolution[1]), preserve_range=True, anti_aliasing=True)
                elif expmtNotes['lung_label'].values[0] == 'WGATR': #uses mean trail ch1 to find WGA
                    cycleTiff_ch1 = tif.imread(trialCycles_ch1[cycleIDX])
                    registeredCycle_ch1, _ = register_tSeries(cycleTiff_ch1, regParams)
                    correctedRegisteredCycle_ch1 = np.where(registeredCycle_ch1[:]>59000,np.nan, registeredCycle_ch1[:])
                   
                    tif.imwrite(trial+f'/rT{trialCounter}_C{cycleIDX+1}_ch1.tif', correctedRegisteredCycle_ch1[:])
                    wgaSlice = np.nanmean(correctedRegisteredCycle_ch1, axis=0)
                    gcampSlice = mIM
                    resolution = mIM.shape
                if os.path.exists(expmtPath+'/cellCountingTiffs/'):
                    annTiffFN = expmt+f'/cellCountingTiffs/cellCounting_T{trialIDX}_slice{trialSlice}.tif'
                else:
                    os.mkdir(expmtPath+'/cellCountingTiffs/')
                    annTiffFN = expmt+f'/cellCountingTiffs/cellCounting_T{trialIDX}_slice{trialSlice}.tif'
                
                
                if cycleIDX == 0:
                    _ = make_annotation_tif(mIM, gcampSlice, wgaSlice, 5, annTiffFN, resolution)
                tif.imwrite(trial+f'/rT{trialCounter}_C{cycleIDX+1}_ch2.tif', correctedRegisteredCycle_ch2[:])
        else:
            print(f'Trial {trialCounter} is registered!')
        trialCounter+=1

def extract_roi_traces(expmtPath):
    print('Extracting:\n',expmt)
    pad = 25 
    trialPaths = glob.glob(expmtPath+'/TSeries*')    
    redZStack = glob.glob(expmtPath+'/ZSeries/ZSeries*Ch1*.tif')[0]
    trialCounter = 0
    dataDict = {}
    numSegmentations = glob.glob(expmtPath+f'/cellCountingTiffs/*.npy')

    try:
        expmtNotes = pd.read_excel(glob.glob(expmtPath+'/expmtNotes*')[0])
        
    except IndexError:
        print('Need to create expmtNotes for experiment! exiting...')
        return
    slicePerTrial = expmtNotes['slice_label'].values
    lungLabel = expmtNotes['lung_label'].values[0]
    if lungLabel == 'WGA594':
        wgaStack = tif.imread(redZStack)

    if os.path.exists(expmt+'/expmtTraces.pkl'):
        print('Traces Already Extracted')
        return None
    else:
        if len(numSegmentations)>0: #make sure segmentations exist
            for trial in trialPaths:

                #get our paths squared away
                registeredTiffs_ch1 = glob.glob(trial+'/rT*C*Ch1.tif')
                registeredTiffs_ch2 = glob.glob(trial+'/rT*C*Ch2.tif')
                segmentationUsed = slicePerTrial[trialCounter]
                masksNPY = glob.glob(expmtPath+f'/cellCountingTiffs/*slice{segmentationUsed}_seg.npy')
                print(masksNPY)

                #Load and Sort ROIs
                segmentationLoaded = np.load(masksNPY[0], allow_pickle=True).item()
                masks = segmentationLoaded['masks']
                outlines = segmentationLoaded['outlines']
                colabeledROIs = np.unique(masks[1,:,:])[1:]
                gCaMPOnly = np.unique(masks[2,:,:])
                masks = masks[1,:,:] + masks[2,:,:] #extracting all gCaMP+ cells
                resolution = masks.shape
                outlines = outlines[1,:,:] + outlines[2,:,:]
                
                #generate mean image for roi view
                masksIM = np.zeros((3, masks.shape[0], masks.shape[1]))
                masksIM[1,:,:] = masks
                

                #generate roi outlines over WGA image to view WGA+ cells
                outlineIM = np.zeros((3,masks.shape[0], masks.shape[1]))
                outlineIM[0,:,:] = outlines>0
                outlineIM[2,:,:] = outlines>0
                if lungLabel == 'WGA594':
                    rmIM =  wgaStack[segmentationUsed,:,:]
                else: #if trial_ch1 is the red image
                    rmIM = tif.imread(registeredTiffs_ch1[0]) #only the first cycle will be used since brightest
                    rmIM = np.nanmean(rmIM, axis=0)
                outlineIM[1,:,:] = np.power(   rmIM/np.max(rmIM-20) , .52)

                #some experiments are chopped up into cycles, others are not, this accounts for it
                rois = np.unique(masks)[1:] 
                cycleFeatures = {}
                for cycleIDX in range(len(registeredTiffs_ch1)):
                    greenCycle = tif.imread(registeredTiffs_ch2[cycleIDX])

                    if greenCycle.shape[1] != resolution[0]: #fit resolution of stack so rois match resolution of tif
                        print('Resizing cycle')
                        greenCycle = resize(greenCycle, (greenCycle.shape[0], resolution[0], resolution[1])) 

                    if cycleIDX == 0: #finish making the roi viewer
                        mIM = np.nanmean(greenCycle, axis=0)
                        masksIM[0,:,:] = np.power(   mIM/np.max(mIM-20) , .72)
                        masksIM[2,:,:] = np.power(   mIM/np.max(mIM-20) , .72)

                    #extract and save traces of every gCaMP+ ROI
                    cycleTrace = []
                    roiFeatures = {}
                    roiFeatures['gCaMP_only_rois'] = gCaMPOnly
                    roiFeatures['colabeled_rois'] = colabeledROIs
                    for roi in rois:
                        if cycleIDX == 0:         #only do this once               
                            xROI, yROI = (masks==roi).nonzero()
                            maxX = np.max(xROI)+pad; minX = np.min(xROI)-pad ; xDiameter = maxX-minX
                            maxY = np.max(yROI)+pad; minY = np.min(yROI)-pad ; yDiameter = maxY - minY
                            rroiWindow = rmIM[:, minY:maxY,minX:maxX]
                            groiWindow = mIM[:, minY:maxY,minX:maxX]
                            redCell = rmIM * (masks==roi)
                            avgRed = np.nanmean(redCell, axis=(0,1))
                            roiFeatures[f'roi{roi}_redAvg'] = avgRed
                            roiFeatures[f'roi{roi}_diameter'] = [xDiameter, yDiameter]
                            roiFeatures[f'roi{roi}_windowCh1'] = rroiWindow
                            roiFeatures[f'roi{roi}_windowCh2'] = groiWindow
                        extractedROI = greenCycle*(masks==roi)
                        roiNAN = np.where(extractedROI==0, np.nan, extractedROI)
                        roiTrace = np.nanmean(roiNAN, axis=(1,2))
                        cycleTrace.append(roiTrace)
                    cycleFeatures[f'cycle{cycleIDX}_traces'] = cycleTrace #raw unmodified traces
                cycleFeatures[f'T{trialCounter}_roiFeatures'] = roiFeatures
                dataDict[trialCounter] = cycleFeatures
                trialCounter += 1
        else:
            print('No ROIs segmented yet')
    return dataDict
    
#%
if __name__=='__main__':
    dataFrom = [
        'U:/expmtRecords/Lucas*',
        'C:/Analysis/april_data/Lucas*',
        'U:/expmtRecords/res_galvo/Lucas*',
        'U:/expmtRecords/mech_galvo/Lucas*',
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
        
        register_trials(expmt, regParams)
        #insert cellpose command here
        dataDict = extract_roi_traces(expmt)
        if dataDict:
            with open(expmt+'/expmtTraces.pkl', 'wb') as f:
                pickle.dump(dataDict, f)

# %%
