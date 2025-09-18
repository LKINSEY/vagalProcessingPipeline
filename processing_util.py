#%%
import tifffile as tif
import numpy as np
import pandas as pd
import os, glob, pickle, cv2, torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from pathlib import Path
import xml.etree.ElementTree as ET
import masknmf
from skimage.transform import resize
from scipy.ndimage import center_of_mass
from datetime import datetime

#registration and processing functions

def define_unique_fovs(metaData):
    fovNum = 0
    fovList = []
    fovID = ((0,0,0),0,(0,0))
    for tID in metaData['TSeries']:
        motorPositions = metaData['TSeries'][tID]['stateShard']['positionCurrent']
        xyzPos = (float(motorPositions['XAxis']), 
                           float(motorPositions['YAxis']), 
                           float(motorPositions['Z Focus']))
        zoom = int(metaData['TSeries'][tID]['stateShard']['opticalZoom'])
        resolution = (int(metaData['TSeries'][tID]['stateShard']['pixelsPerLine']),
                      int(metaData['TSeries'][tID]['stateShard']['linesPerFrame']))
        
        thisFOV = (xyzPos, zoom, resolution)
        if metaData['TSeries'][tID]['QC'] == 1:
            if set(thisFOV) == set(fovID):
                fovList.append(fovNum)
            else:
                fovNum += 1
                fovList.append(fovNum)
                fovID = thisFOV
        else:
            fovList.append(-1)
    return fovList

def extract_roi_traces(expmtPath, metaData):
    '''
    parameters:
        expmtPath (str): string path directing function to data directory
        metaData (dict): metadata dictionary of extracted metadata for each trial
    outputs:
        dataDict (dict): data for each trial in dictionary format
    '''
    print('\nExtracting:\n',expmtPath, flush=False)
    pad = 25 
    trialPaths = glob.glob(expmtPath+'/TSeries*')    
    trialCounter = 0
    dataDict = {}
    numSegmentations = glob.glob(expmtPath+f'/cellCountingTiffs/*.npy')

    try:
        expmtNotes = pd.read_excel(glob.glob(expmtPath+'/expmtNotes*')[0])
        
    except IndexError:
        print('Need to create expmtNotes for experiment! exiting...')
        return
    
    # slicePerTrial = expmtNotes['slice_label'].values
    slicePerTrial = define_unique_fovs(metaData)

    if os.path.exists(expmtPath+'/expmtTraces.pkl'):
        print('Traces Already Extracted')
        return None
    else:
        if len(numSegmentations)>0: #make sure segmentations exist
            for trial in trialPaths:
                registeredTiffs_ch1 = glob.glob(trial+'/rT*C*Ch1.tif')
                registeredTiffs_ch2 = glob.glob(trial+'/rT*C*Ch2.tif')
                segmentationUsed = slicePerTrial[trialCounter]
                if segmentationUsed == -1:
                    print('Skipping trial because Trial QCed!')
                    trialCounter += 1
                    continue
                masksNPY = glob.glob(expmtPath+f'/cellCountingTiffs/*slice{segmentationUsed}_seg.npy')
                print(f'Extracting traces from fov {segmentationUsed} for trial {trialCounter}')

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

                #some experiments are chopped up into cycles, others are not, this accounts for it
                rois = np.unique(masks)[1:] 
                cycleFeatures = {}
                for cycleIDX in range(len(registeredTiffs_ch2)):
                    greenCycle = tif.imread(registeredTiffs_ch2[cycleIDX])
                    #best way I can save this feature for later
                    if len(registeredTiffs_ch1)>0:
                        redCycle = tif.imread(registeredTiffs_ch1[cycleIDX])
                    else:
                        redCycle = greenCycle #when making plotting functions I will use this fact

                    if greenCycle.shape[1] != resolution[0]: #fit resolution of stack so rois match resolution of tif
                        print('Resizing cycle')
                        greenCycle = resize(greenCycle, (greenCycle.shape[0], resolution[0], resolution[1])) 
                        redCycle = resize(redCycle, (redCycle.shape[0], resolution[0], resolution[1])) 

                    if cycleIDX == 0: #finish making the roi viewer
                        mIM = np.nanmean(greenCycle, axis=0)
                        rmIM = np.nanmean(redCycle, axis=0)
                        masksIM[0,:,:] = np.power(   mIM/np.max(mIM-20) , .72)
                        masksIM[2,:,:] = np.power(   mIM/np.max(mIM-20) , .72)
                        outlineIM[1,:,:] = np.power(   rmIM/np.max(rmIM-20) , .52)

                    #extract and save traces of every gCaMP+ ROI
                    cycleTrace = []
                    cycleTrace_red = []
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
                        #get red trace
                        extractedROI_red = redCycle*(masks==roi)
                        roiNaN_red = np.where(extractedROI_red==0, np.nan, extractedROI_red)
                        roiTrace_red = np.nanmean(roiNaN_red, axis=(1,2))
                        cycleTrace_red.append(roiTrace_red)
                        #get green trace
                        extractedROI = greenCycle*(masks==roi)
                        roiNAN = np.where(extractedROI==0, np.nan, extractedROI)
                        roiTrace = np.nanmean(roiNAN, axis=(1,2))
                        cycleTrace.append(roiTrace)

                    cycleFeatures[f'cycle{cycleIDX}_traces'] = cycleTrace #raw unmodified traces
                    cycleFeatures[f'cycle{cycleIDX}_traces_red'] = cycleTrace_red
                    cycleFeatures[f'T{trialCounter}_roiFeatures'] = roiFeatures

                dataDict[trialCounter] = cycleFeatures
                dataDict[f'T{trialCounter}_masksIM'] = masksIM
                dataDict[f'T{trialCounter}_outlinesIM'] = outlineIM
                trialCounter += 1

        else:
            print('No ROIs segmented yet')
            return None
        
    return dataDict

def fft_rigid_cycle_moco_shifts(mIM, template):
    '''
    I dont really use this, but its nice to have for sanity checks

    this method is for finding how much movement occurs from one cycle to another. Sometimes there is movement
    if the prep is not solidified under the scope, sometimes mechanical stim moves things, ect.

    We take 2 images:
    mIM = (n,m) mean image of cycle
    template = (n,m) mean image of FIRST cycle (or cycle we are going to use as our template)

    Cross-Corr using FFT:
    first multiply the real fft of the template by the complex conjugate of the real fft of the mIM
    then take an inverse fft of this result
    this produces a corr plane, where the max value of this plane represents the indices to shift the mIM
    in order to register with the template

    Shifting based of Argmax of Result
    mIM needs to be padded first, then the shift occurs, then we trim the padding so that masks
    can still be accurately overlayed onto the mIM in later trace extraction steps

    To correct a cycle tiff, do:
    np.roll(tiffArr, shift=shifts, axis=(1,2)
    '''
    C = np.fft.irfft2(
        np.fft.rfft2(template) * np.conj(np.fft.rfft2(mIM))
    )
    # cShift = np.fft.fftshift(C,(0,1))
    pxlPosX, pxlPosY = np.unravel_index(np.argmax(C), C.shape)
    shifts = (pxlPosX , pxlPosY)
    return shifts

def register_2ch_trials(expmtPath, metaData, regParams):
    '''
    No Longer Supporting Registration using alternative WGA Conjugates
    MUST have 2 channels recording moving forward
    '''
    print('Registering:\n', expmtPath)
    trialPaths = np.array(glob.glob(expmtPath+'/TSeries*'))
    #omitting zstack loading for now since all experiments no longer use other WGA conjugates

    #expmt notes check
    try:
        expmtNotes = pd.read_excel(glob.glob(expmtPath+'/expmtNotes*')[0])
    except IndexError:
        print('Need to create expmtNotes for experiment! exiting...')
        return
    
    # fovPerTrial = expmtNotes['slice_label'].values
    # fovs = np.unique(fovPerTrial)

    fovPerTrial = define_unique_fovs(metaData)
    fovs = np.unique(fovPerTrial)


    trialCount = 1
    for fov in fovs:
        fovBool = fovPerTrial==fov
        trialSet = trialPaths[fovBool]
        trialInSetCount = 0
        for trial in trialSet:
            print(f'Registering Trial {trialCount} in FOV {fov}')
            ch1Tiffs = glob.glob(trial+'/TSeries*Ch1*.tif')
            ch2Tiffs = glob.glob(trial+'/TSeries*Ch2*.tif')
            registeredTrials = glob.glob(trial+'/rT*_C*_ch*.tif')
            if len(registeredTrials)==0:
                for cycleIDX in range(len(ch2Tiffs)):
                    print('Cycle', cycleIDX+1, 'of', len(ch2Tiffs))
                    loadedCh1Tiff = tif.imread(ch1Tiffs[cycleIDX])
                    loadedCh2Tiff = tif.imread(ch2Tiffs[cycleIDX])
                    if trialInSetCount == 0:
                        #for the first trial in a trial set
                        annTiffDr = Path(expmtPath)/'cellCountingTiffs'
                        annTiffDr.mkdir(parents=True, exist_ok=True)
                        annTiffFN = str(annTiffDr / f'cellCounting_T{trialCount-1}_slice{fov}.tif')
                        if cycleIDX == 0:
                            #first cycle of first trial in set -- everything will be aligned to first cycle of first trial in set, so only do this once
                            registeredCycle = register_tSeries(loadedCh2Tiff, regParams, template = None)
                            registeredRed = register_tSeries(loadedCh1Tiff, regParams, template = None)
                            mTemplate = np.nanmean(registeredCycle, axis=0)
                            mRedIM = np.nanmean(registeredRed, axis=0)  
                            _ = make_annotation_tif(mTemplate, mTemplate, mRedIM, 5, annTiffFN, mTemplate.shape)
                            tif.imwrite(trial+f'/rT{trialInSetCount}_C{cycleIDX+1}_ch2.tif', registeredCycle[:])
                            tif.imwrite(trial+f'/rT{trialInSetCount}_C{cycleIDX+1}_ch1.tif', registeredRed[:])
                        else:
                            #consecutive cycle of first trial in set (if it exists)
                            registeredCycle = register_tSeries(loadedCh2Tiff, regParams, template = mTemplate) #register all to initial registration
                            registeredRed = register_tSeries(loadedCh1Tiff, regParams, template = mRedIM)#register all to initial registration
                            mCycle = np.nanmean(registeredCycle, axis=0)
                            cycleShifted = np.roll(registeredCycle, shift=fft_rigid_cycle_moco_shifts(mCycle, mTemplate), axis=(1,2))
                            redCycleShifted = np.roll(registeredRed, shift=fft_rigid_cycle_moco_shifts(mCycle, mTemplate), axis=(1,2))
                            tif.imwrite(trial+f'/rT{trialInSetCount}_C{cycleIDX+1}_ch2.tif', cycleShifted[:])
                            tif.imwrite(trial+f'/rT{trialInSetCount}_C{cycleIDX+1}_ch1.tif', redCycleShifted[:])
                    else: 
                        # for all cycles of consecutive trials in trial set 
                        loadedCh2Tiff = tif.imread(ch2Tiffs[cycleIDX])
                        registeredCycle = register_tSeries(loadedCh2Tiff, regParams, template = mTemplate) #register all to initial registration
                        registeredRed = register_tSeries(loadedCh1Tiff, regParams, template = mRedIM) #register all to initial registration
                        mCycle = np.nanmean(registeredCycle, axis=0)
                        tif.imwrite(trial+f'/rT{trialInSetCount}_C{cycleIDX+1}_ch2.tif', registeredCycle[:])
                        tif.imwrite(trial+f'/rT{trialInSetCount}_C{cycleIDX+1}_ch1.tif', registeredRed[:])
                trialInSetCount += 1
            else:
                print(f'Trial {trialCount} Already Registered')
            trialCount+=1

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
        annTiff = np.stack((alignedgWGAStack, alignedgCaMPStack, mIM), axis=0)
    else:
        #case 2 template matching
        print('Case 2 Alignment ... have not encountered this yet actually')
        annTiff = np.stack((wgaSlice, gcampSlice, mIM), axis=0)
        # return
        
    
    tif.imwrite(annTifFN,annTiff)
    return annTiff

def register_tSeries(rawData, regParams, template = None):
    max_shifts = regParams['maxShifts']
    frames_per_split = regParams['frames_per_split']
    device = regParams['device']
    niter_rig = regParams['niter_rig']
    overlaps = regParams['overlaps']
    max_deviation_rigid = regParams['max_deviation_rigid']
    num_blocks = regParams['num_blocks']

    #code copied from repo's demo
    if template is None:
        rigid_strategy = masknmf.RigidMotionCorrection(max_shifts = max_shifts)
        pwrigid_strategy = masknmf.PiecewiseRigidMotionCorrection(num_blocks = num_blocks,
                                                                overlaps = overlaps,
                                                                max_rigid_shifts = max_shifts,
                                                                max_deviation_rigid = max_deviation_rigid)
        pwrigid_strategy = masknmf.motion_correction.compute_template(rawData,
                                                                    rigid_strategy,
                                                                    num_iterations_piecewise_rigid = niter_rig,
                                                                    pwrigid_strategy = pwrigid_strategy,
                                                                    device = device,
                                                                    batch_size = frames_per_split)
        moco_results = masknmf.RegistrationArray(rawData, pwrigid_strategy, device = device)
    else:
        rigid_strategy = masknmf.RigidMotionCorrection(max_shifts = max_shifts)
        pwrigid_strategy = masknmf.PiecewiseRigidMotionCorrection(num_blocks = num_blocks,
                                                                overlaps = overlaps,
                                                                max_rigid_shifts = max_shifts,
                                                                max_deviation_rigid = max_deviation_rigid,
                                                                template=torch.tensor(template))

        moco_results = masknmf.RegistrationArray(rawData, pwrigid_strategy, device = device)

    return moco_results[:]





#Plotting and summary functions

def extract_metadata(expmt):
    metaData = {}
    trials = glob.glob(expmt+'/TSeries*/')
    zstackXML = glob.glob(expmt+'/ZSeries*/*001.xml')[0]

    #read expmtNotes in metaData extraction
    expmtNotes = pd.read_excel(glob.glob(expmt+'\expmtNotes*.xlsx')[0])
    #new style of notes moving forward will detail better what type of stimulation given
    '''
    new columns will be:
    - duration
        examples: 3s, 5s, 7s for exsp or insp stims, blank for baseline, and length of gas exposure for gas trials
    - type
        examples: insp, exsp, baseline-V, baseline-LF, hyperVent, hypoVent, hypoxia, hypercapnia, ect...
    - magnitude
        examples: 30, 20, 10, 5, 5
    - magnitude units
        examples: cmH2O, seconds, minutes
    '''


    try:
        md= ET.parse(zstackXML)
    except FileNotFoundError:
        print(f'No Meta Data For ZStack')
        # return
    rootStack = md.getroot()
    stackMeta = {}
    for child in rootStack:
        if len(child.attrib) == 0: #its our PVShard
            stateShardMeta = {}
            for state in child:
                if len(state.keys()) == 2:
                    stateShardMeta[state.get('key')] = state.get('value')
                else:
                    indexMeta = {}
                    for indexedValue in state:
                        if len(indexedValue)>0:
                            for subindexedValue in indexedValue:
                                if 'description' in subindexedValue.keys():
                                    indexMeta[subindexedValue.get('description')] = subindexedValue.get('value')
                                else:
                                    indexMeta[indexedValue.get('index')] = subindexedValue.get('value')

                        else:
                            if 'description' in indexedValue.keys():
                                indexMeta[indexedValue.get('description')] = indexedValue.get('value')
                            else:
                                indexMeta[indexedValue.get('index')] = indexedValue.get('value')
                    stateShardMeta[state.get('key')] = indexMeta
        if child.attrib.get('type') ==  'ZSeries':
            stackMeta['start_time'] = child.get('time')
            nSlices = len(child)-1 #first is just a shard
            stackMeta['nSlices'] = nSlices
            sliceRelTimes = {}
            sliceAbsTimes = {}
            frameMeta = {}
            sliceIDX = 0
            for frame in child[1:]:
                sliceRelTimes[sliceIDX] = frame.attrib.get('relativeTime')
                sliceAbsTimes[sliceIDX] = frame.attrib.get('absoluteTime')
                positionMeta = {}
                for shard in frame:
                    if len(shard)>0:
                        for axis in shard[0]:
                            positionMeta[axis.get('index')] = axis[0].get('value')
                frameMeta[sliceIDX] = positionMeta
                sliceIDX+=1
            stackMeta['scanTimes_rel'] = sliceRelTimes
            stackMeta['scanTimes_abs'] = sliceAbsTimes
            stackMeta['frameMetas'] = frameMeta
    ZSeriesMeta = {}
    ZSeriesMeta['Frames'] = stackMeta
    ZSeriesMeta['stateShard'] = stateShardMeta
    metaData['ZSeries'] = ZSeriesMeta


    tSeriesMeta = {}

    for tidx, tPath in enumerate(trials): #roughly 2-3s a trial for nFrames = 120
        trialMeta = {}
        trialMeta['type'] = expmtNotes['type'].values[tidx]
        trialMeta['duration'] = expmtNotes['duration'].values[tidx]
        trialMeta['magnitude'] = expmtNotes['magnitude'].values[tidx]
        trialMeta['units'] = expmtNotes['units'].values[tidx]
        trialMeta['QC'] = expmtNotes['QC'].values[tidx]
        
        
        metaDataFN = os.path.join(
            tPath, 
            [FN for FN in os.listdir(tPath) if 'VoltageRecording' not in FN and '.xml' in FN][0]
            )
        try:
            md= ET.parse(metaDataFN)
        except FileNotFoundError:
            print(f'No Meta Data For Trial {tidx}')
            # return
        root = md.getroot()
        
        cycleCount = 0
        for child in root:
            if len(child.attrib) == 0: #its our PVShard
                stateShardMeta = {}
                for state in child:
                    if len(state.keys()) == 2:
                        stateShardMeta[state.get('key')] = state.get('value')
                    else:
                        indexMeta = {}
                        for indexedValue in state:
                            if len(indexedValue)>0:
                                for subindexedValue in indexedValue:
                                    if 'description' in subindexedValue.keys():
                                        indexMeta[subindexedValue.get('description')] = subindexedValue.get('value')
                                    else:
                                        indexMeta[indexedValue.get('index')] = subindexedValue.get('value')

                            else:
                                if 'description' in indexedValue.keys():
                                    indexMeta[indexedValue.get('description')] = indexedValue.get('value')
                                else:
                                    indexMeta[indexedValue.get('index')] = indexedValue.get('value')
                        stateShardMeta[state.get('key')] = indexMeta
                trialMeta['stateShard'] = stateShardMeta
            #for future reference we may need to consider xyGrid?

            elif child.get('type') == 'TSeries Timed Element':
                trialMeta[f'cycle_{cycleCount}_time'] = child.get('time')
                frameMeta = {}

                #subtracting the <PVShard/> and <VoltageRecording> children if voltage is there otherwise just <PVShard/> child
                if 'VoltageRecording' in child[1].get('configurationFile'):
                    frameTime_rel = np.zeros((len(child)-2,)) 
                    frameTime_abs = np.zeros((len(child)-2,))
                    for fIDX, frame in enumerate(child[2:]):
                        frameTime_rel[fIDX] = frame.get('relativeTime')
                        frameTime_abs[fIDX] = frame.get('absoluteTime')
                else:
                    frameTime_rel = np.zeros((len(child)-1,)) 
                    frameTime_abs = np.zeros((len(child)-1,))
                    for fIDX, frame in enumerate(child[1:]):
                        frameTime_rel[fIDX] = frame.get('relativeTime')
                        frameTime_abs[fIDX] = frame.get('absoluteTime')
                #take framerate from last frame
                frameMeta['fs'] = frame[3][0].get('value') #the only hardcoded child in this entire thing... Im just lazy...
                frameMeta['frameTime_rel'] = frameTime_rel
                frameMeta['frameTime_abs'] = frameTime_abs
                trialMeta[f'cycle_{cycleCount}_Framemeta'] = frameMeta
                cycleCount+=1
        trialMeta['nCycles'] = cycleCount
        tSeriesMeta[tidx] = trialMeta

    metaData['TSeries'] = tSeriesMeta
    return metaData
                
def trialize_physiology(physDict, metaDataDict):
    trig = physDict['Trial_Trigger_raw']
    fs_physio = float(physDict['Trial_Trigger_fs'])
    highs = np.where(trig>2, 1, 0)
    edges = (np.diff(highs)>=1).astype(int)
    risingEdges = np.where(edges)[0]
    edgeDistances = np.diff(risingEdges)

    relT = np.array(list(metaDataDict['ZSeries']['Frames']['scanTimes_rel'].values()), dtype=np.float64)
    deltaT = np.diff(relT)
    deltaTicks_expected = (deltaT*fs_physio).astype(int) #fs of 
    nSlices = len(deltaTicks_expected)

    zStackStart = 0
    errs = []
    for i in range(5): #highly improbable to have >5 scans before a zstack in a given physiological recording
        shiftedEdges = np.roll(edgeDistances, shift = -zStackStart)
        firstNEdges = shiftedEdges[:nSlices]
        errs.append(np.sum(abs(firstNEdges - deltaTicks_expected)))
        zStackStart+= 1

    startingEdge = np.argmin(errs)
    startingTick = risingEdges[startingEdge]

    highsAdjusted = highs[startingTick:]

    # ticksSubtracted = risingEdges - startingTick
    # scanTicksAfterzStack = ticksSubtracted[startingEdge:] #in ticks recorded at fs_physio rate

    startTime = metaDataDict['ZSeries']['Frames']['start_time'] #our t=0
    sT = datetime.strptime(startTime[:15], "%H:%M:%S.%f")
    secondStart = sT.hour*3600 + sT.minute*60 + sT.second + sT.microsecond/1e6
    tickStart = int(secondStart*fs_physio)

    trialTicksFromStart = {}
    lastTick = {}
    for tIDX in metaDataDict['TSeries'].keys():
        trialDict = metaDataDict['TSeries'][tIDX]
        cycleStartTicks = np.zeros((metaDataDict['TSeries'][tIDX]['nCycles'],))
        cycleEndTicks = np.zeros((metaDataDict['TSeries'][tIDX]['nCycles'],))
        for cIDX in range(metaDataDict['TSeries'][tIDX]['nCycles']):

            cycleStart = metaDataDict['TSeries'][tIDX][f'cycle_{cIDX}_time']
            csT = datetime.strptime(cycleStart[:15], "%H:%M:%S.%f")
            cycleSecondStart = csT.hour*3600 + csT.minute*60 + csT.second + csT.microsecond/1e6
            cycleTickStart = int(cycleSecondStart*fs_physio)
            cycleStartTicks[cIDX] = cycleTickStart - tickStart

            lastFrameInCycle = metaDataDict['TSeries'][tIDX][f'cycle_{cIDX}_Framemeta']['frameTime_rel'][-1]
            cycleTickEnd = int(lastFrameInCycle*fs_physio)
            cycleEndTicks[cIDX] = cycleTickEnd + cycleTickStart - tickStart

        trialTicksFromStart[tIDX] = cycleStartTicks
        lastTick[tIDX] = cycleEndTicks

    
    trializedData = {}

    for t in trialTicksFromStart.keys():
        startTick = int(trialTicksFromStart[t][0] - fs_physio) #added padding to show trial bounderies better
        stopTick = int(lastTick[t][-1] + fs_physio)

        ###
        #c'est stupide, je sais. J'ameliorerai le code a l'avenir.
        measurements = [ 'Spirometer_raw',
                        'ECG_raw',
                        'ECG_Rate_raw',
                        'Air_Flow_Filter_(20Hz)_raw',
                        'TV_raw',
                        'Breath_Rate_raw']
        ###


        trializedMeasurements = {}
        for dataType in measurements:
            adjustedData = physDict[dataType][startingTick:]
            trializedTrace = adjustedData[startTick:stopTick]
            trializedMeasurements[dataType] = trializedTrace
        trializedMeasurements['Trial_Trig'] = highsAdjusted[startTick:stopTick]
        trializedData[t] = trializedMeasurements
    return trializedData

def find_stim_tick_physio(duration, trialBreathingRate, fs_physio):
    endTick = np.where(np.diff(trialBreathingRate)== np.nanmax(np.diff(trialBreathingRate)))[0] - fs_physio
    startTick = int( endTick - (duration*fs_physio))
    return startTick

def sync_physiology(physioDict, dataDict, metaData):
    '''
    inputs:
        physioDict (dict): raw data extracted from labcharts
        dataDict (dict): traces extracted from segmentations of registered tiffs
        metaData (dict): data extracted from XML files from Bruker microscope recordings
    
    outputs:
        plottingDict (dict): 
            plottingDict[trialID].keys()=
                physio (dict): the physiological data for trialID
                Fraw (ndarray): the raw trace array for trialID, size--> (nFrames,nROIs)
                dFF (ndarray): the dFF of trace array for trialID, size --> (nFrames, nROIs) baseline used 3 seconds before stim
                stimIDX (int): index of frame stimulation occurs for 2p data (use to index Fraw or dFF)
                traceX (ndarray): synchronized x labels for each frame in Fraw or dFF
                physioX (ndarray): synchronized x labels for each frame in data from physio


    '''
    fs_physio = physioDict['Spirometer_fs']
    plottingDict = {}
    trializedPhysio = trialize_physiology(physioDict, metaData)
    registeredTrials = [t for t in dataDict.keys() if type(t) is int]
    trialIDX = 0
    for tID in registeredTrials:
        trialDict = {}
        physioX = np.arange(start=-fs_physio, stop = len(trializedPhysio[tID]['Trial_Trig'])-fs_physio, step=1)
        trialDict['physioX'] = physioX
        traceX = []
        traceY = []
        if metaData['TSeries'][tID]['nCycles']>1:
            # xCorrMatrices = [] #commenting out because not sure if I want this
            for cIDX in range(metaData['TSeries'][tID]['nCycles']):
                frameTimeStamps = metaData['TSeries'][tID][f'cycle_{cIDX}_Framemeta']['frameTime_abs'] # might need to change to relative just in case?
                frameTicks = (frameTimeStamps*fs_physio).astype(int)
                frameTicks = np.concatenate([frameTicks, np.array([frameTicks[-1]+1])], axis=0, dtype=int)
                # roiTraces_xCorr = pd.DataFrame(np.array(dataDict[tID][f'cycle{cIDX}_traces']).T).corr()
                # xCorrMatrices.append(roiTraces_xCorr)
                roiTraces = np.pad(np.array(dataDict[tID][f'cycle{cIDX}_traces']).T, ((0,1),(0,0)), mode='constant', constant_values=np.nan)
                traceX.append(frameTicks)
                traceY.append(roiTraces)
            # roiXCorr = np.nanmean(np.array(xCorrMatrices), axis=0) #not sure if this makes sense... assumes if 1 cycle is corr the next will be similar corr...
        else:
            frameTimeStamps = metaData['TSeries'][tID]['cycle_0_Framemeta']['frameTime_abs']
            frameTicks = (frameTimeStamps*fs_physio).astype(int)
            traceX.append(frameTicks)
            traceX = np.array(traceX)
            traceY.append(np.array(dataDict[tID]['cycle0_traces']).T)
            traceY = np.array(traceY)

        traceX = np.concatenate(traceX, axis=0)
        Fraw = np.concatenate(traceY, axis=0)
        trialDict['Fraw'] = Fraw
        trialDict['traceX'] = traceX
        trialType = metaData['TSeries'][tID]['type']
        if 'baseline' in trialType:
            fps = (1/float(metaData['TSeries'][tID]['cycle_0_Framemeta']['fs']))
            baselinePeriod = round(fps * 3) #hardcoded ~3 seconds before stim is baseline
            f0 = np.nanmean(Fraw[:baselinePeriod, :], axis=0) #baseline is first 3 seconds of recording on first epoch
            trialDict['stimIDX'] = np.nan

        #will include 'gas' criteria here soon
        #elif 'gas' in trialType

        else:
            duration = metaData['TSeries'][tID]['duration']
            stimTick = find_stim_tick_physio(duration, trializedPhysio[tID]['Breath_Rate_raw'], fs_physio)
            stimIDX = np.argmin(abs(traceX - stimTick))
            fps = (1/float(metaData['TSeries'][tID]['cycle_0_Framemeta']['fs']))
            baselinePeriod = round(fps * 3) #hardcoded ~3 seconds before stim is baseline
            f0 = np.nanmean(Fraw[(stimIDX-baselinePeriod):stimIDX, :], axis=0)
            trialDict['stimIDX'] = stimIDX
        
        dFF = (traceY - f0) / f0
        trialDict['dFF'] = dFF
        trialDict['physio'] = trializedPhysio[tID]
        trialIDX+=1
        plottingDict[tID] = trialDict
    return plottingDict