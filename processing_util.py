#%%
import tifffile as tif
import numpy as np
import pandas as pd
import os, glob, pickle, cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import jnormcorre
import jnormcorre.motion_correction
import jnormcorre.utils.registrationarrays as registrationarrays
from skimage.transform import resize

expmtFiles = glob.glob('U:/expmtRecords/gas_expmts/*')


def extract_roi_traces(expmtPath):
    print('\nExtracting:\n',expmt, flush=False)
    pad = 25 
    trialPaths = glob.glob(expmtPath+'/TSeries*')    
    redZStack = glob.glob(expmtPath+'/ZSeries*/ZSeries*Ch1*.tif')[0]
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
                if segmentationUsed == -1:
                    print('Skipping trial because zstack error')
                    trialCounter += 1
                    continue
                masksNPY = glob.glob(expmtPath+f'/cellCountingTiffs/*slice{segmentationUsed}_seg.npy')
                print(masksNPY)

                #Load and Sort ROIs
                segmentationLoaded = np.load(masksNPY[0], allow_pickle=True).item()
                masks = segmentationLoaded['masks']
                outlines = segmentationLoaded['outlines']
                colabeledROIs = np.unique(masks[1,:,:])[1:]
                gCaMPOnly = np.unique(masks[2,:,:])[1:]
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
                    annTiff = tif.imread(glob.glob(expmtPath+f'/cellCountingTiffs/*slice{segmentationUsed}.tif')[0])
                    rIM = annTiff[0,:,:]
                else: #if trial_ch1 is the red image
                    rmIM = tif.imread(registeredTiffs_ch1[0]) #only the first cycle will be used since brightest
                    rmIM = np.nanmean(rmIM, axis=0)
                outlineIM[1,:,:] = np.power(   rmIM/np.max(rmIM-20) , .52)

                #some experiments are chopped up into cycles, others are not, this accounts for it
                rois = np.unique(masks)[1:] 
                cycleFeatures = {}
                for cycleIDX in range(len(registeredTiffs_ch2)):
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
                            rroiWindow = rmIM[ minY:maxY,minX:maxX]
                            groiWindow = mIM[ minY:maxY,minX:maxX]
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
                dataDict[f'T{trialCounter}_masksIM'] = masksIM
                dataDict[f'T{trialCounter}_outlinesIM'] = outlineIM
                trialCounter += 1
        else:
            print('No ROIs segmented yet')
    return dataDict

def register_trials(expmtPath, regParams):
    print('Registering:\n', expmtPath)
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
                registeredCycle_ch2, _ = register_tSeries(cycleTiff_ch2, regParams, expmtPath)
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
                else: #uses mean trail ch1 to find WGA
                    cycleTiff_ch1 = tif.imread(trialCycles_ch1[cycleIDX])
                    registeredCycle_ch1, _ = register_tSeries(cycleTiff_ch1, regParams, expmtPath)
                    correctedRegisteredCycle_ch1 = np.where(registeredCycle_ch1[:]>59000,np.nan, registeredCycle_ch1[:])
                   
                    tif.imwrite(trial+f'/rT{trialCounter}_C{cycleIDX+1}_ch1.tif', correctedRegisteredCycle_ch1[:])
                    wgaSlice = np.nanmean(correctedRegisteredCycle_ch1, axis=0)
                    gcampSlice = mIM
                    resolution = mIM.shape
                if os.path.exists(expmtPath+'/cellCountingTiffs/'):
                    annTiffFN = expmtPath+f'/cellCountingTiffs/cellCounting_T{trialIDX}_slice{trialSlice}.tif'
                else:
                    os.mkdir(expmtPath+'/cellCountingTiffs/')
                    annTiffFN = expmtPath+f'/cellCountingTiffs/cellCounting_T{trialIDX}_slice{trialSlice}.tif'
                
                
                if cycleIDX == 0:
                    _ = make_annotation_tif(mIM, gcampSlice, wgaSlice, 5, annTiffFN, resolution)
                tif.imwrite(trial+f'/rT{trialCounter}_C{cycleIDX+1}_ch2.tif', correctedRegisteredCycle_ch2[:])
        else:
            print(f'\rTrials Registered!', end='', flush=True)
        trialCounter+=1

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
        #case 2 template matching
        print('Case 2 Alignment ... have not encountered this yet actually')
        return
        
    annTiff = np.stack((alignedgWGAStack, alignedgCaMPStack, mIM), axis=0)
    tif.imwrite(annTifFN,annTiff)
    return annTiff

def register_tSeries(rawData, regParams, expmtPath, compareRaw=False):
    max_shifts = regParams['maxShifts']
    frames_per_split = regParams['frames_per_split']
    num_splits_to_process_rig = regParams['num_splits_to_process_rig']
    niter_rig = regParams['niter_rig']
    save_movie = regParams['save_movie']
    pw_rigid = regParams['pw_rigid']
    strides = regParams['strides']
    overlaps = regParams['overlaps']
    max_deviation_rigid = regParams['max_deviation_rigid']

    if 'mech_galvo' in expmtPath:
        template = np.nanmean(rawData[:5, :,:], axis=0)
    else:
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

def extract_metaData(expmt):
    trials = glob.glob(expmt+'/TSeries*/')
    metaData = {}
    for tidx, tPath in enumerate(trials): #roughly 2-3s a trial for nFrames = 120
        metaDataFN = os.path.join(
            tPath, 
            [FN for FN in os.listdir(tPath) if 'VoltageRecording' not in FN and '.xml' in FN][0]
            )
        
        try:
            md= ET.parse(metaDataFN)
        except FileNotFoundError:
            print(f'No Meta Data For Trial {tidx}')
            return
        
        root = md.getroot()
        trialMeta = {}
        
        #Extracting PVStateShard for trial, state shard is nested in root[1] always
        for child in root[1]:
            if child.attrib.get('value'):
                trialMeta[child.attrib.get('key')] = child.attrib.get('value')
            else:
                if child.attrib.get('key') == 'laserPower':
                    laserDict = {}
                    for idx in range(len(child)):
                        laserDict[child[idx].attrib.get('description')] = child[idx].attrib.get('value')
                    trialMeta[child.attrib.get('key')] = laserDict
                elif child.attrib.get('key') == 'laserWavelength':
                    trialMeta[child.attrib.get('key')] = child[0].attrib.get('value')
                elif child.attrib.get('key') == 'micronsPerPixel':
                    micronPerPixelDict = {}
                    for idx in range(len(child)):
                        micronPerPixelDict[child[idx].attrib.get('index')] = child[idx].attrib.get('value')
                    trialMeta[child.attrib.get('key')] = micronPerPixelDict
                elif child.attrib.get('key') == 'pmtGain':
                    pmtGainDict = {}
                    for idx in range(len(child)):
                        pmtGainDict[child[idx].attrib.get('description')] = child[idx].attrib.get('value')
                    trialMeta[child.attrib.get('key')] = pmtGainDict
                elif child.attrib.get('key') == 'positionCurrent':
                    motorPosDict = {}
                    for idx in range(len(child)):
                        motorPosDict[child[idx].attrib.get('index')] = child[idx][0].attrib.get('value')
                    trialMeta['motorPos'] = motorPosDict

        #Extracting MetaData From Each Frame of Tiff, frame shards always nested in root[2]
        trialMeta['nFrames'] = len(root[2]) - 2 #subtracting out <PVStateShard /> and <VoltageRecording>
        trialMeta['framePeriod'] = root[2][3][3][0].attrib.get('value')
        trialMeta['fps'] = 1/float(root[2][3][3][0].attrib.get('value'))
        relTime = []
        absTime = []
        for child in root[2]:
            if child.tag == 'Frame':
                relTime.append(child.attrib.get('relativeTime'))
                absTime.append(child.attrib.get('absoluteTime'))
        trialMeta['relTime'] = np.array(relTime)
        trialMeta['absTime'] = np.array(absTime)
        metaData[tidx] = trialMeta
    return metaData
                

    