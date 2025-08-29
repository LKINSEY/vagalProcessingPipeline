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

#registration and processing functions

def extract_roi_traces(expmtPath):
    print('\nExtracting:\n',expmtPath, flush=False)
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

    if os.path.exists(expmtPath+'/expmtTraces.pkl'):
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
                #no longer using WGA594 - uncomment if processing older files
                # if lungLabel == 'WGA594':
                #     annTiff = tif.imread(glob.glob(expmtPath+f'/cellCountingTiffs/*slice{segmentationUsed}.tif')[0])
                #     rIM = annTiff[0,:,:]
                # else: #if trial_ch1 is the red image
                #     rmIM = tif.imread(registeredTiffs_ch1[0]) #only the first cycle will be used since brightest
                #     rmIM = np.nanmean(rmIM, axis=0)
                outlineIM[1,:,:] = np.power(   rmIM/np.max(rmIM-20) , .52)

                #some experiments are chopped up into cycles, others are not, this accounts for it
                rois = np.unique(masks)[1:] 
                cycleFeatures = {}
                for cycleIDX in range(len(registeredTiffs_ch2)):
                    greenCycle = tif.imread(registeredTiffs_ch2[cycleIDX])
                    redCycle = tif.imread(registeredTiffs_ch1[cycleIDX])

                    if greenCycle.shape[1] != resolution[0]: #fit resolution of stack so rois match resolution of tif
                        print('Resizing cycle')
                        greenCycle = resize(greenCycle, (greenCycle.shape[0], resolution[0], resolution[1])) 
                        redCycle = resize(greenCycle, (greenCycle.shape[0], resolution[0], resolution[1])) 

                    if cycleIDX == 0: #finish making the roi viewer
                        mIM = np.nanmean(greenCycle, axis=0)
                        rmIM = np.nanmean(redCycle, axis=0)
                        masksIM[0,:,:] = np.power(   mIM/np.max(mIM-20) , .72)
                        masksIM[2,:,:] = np.power(   mIM/np.max(mIM-20) , .72)

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
                        extractedROI = greenCycle*(masks==roi)
                        extractedROI_red = redCycle*(mask==roi)
                        roiNAN = np.where(extractedROI==0, np.nan, extractedROI)
                        roiNaN_red = np.where(extractedROI_red==0, np.nan, extractedROI)
                        roiTrace_red = np.nanmean(roiNaN_red, axis=(1,2))
                        roiTrace = np.nanmean(roiNAN, axis=(1,2))
                        cycleTrace.append(roiTrace)
                        cycleTrace_red.append(roiTrace_red)
                    cycleFeatures[f'cycle{cycleIDX}_traces'] = cycleTrace #raw unmodified traces
                    cycleFeatures[f'cycle{cycleIDX}_traces_red'] = cycleTrace_red
                    cycleFeatures[f'T{trialCounter}_roiFeatures'] = roiFeatures
                dataDict[trialCounter] = cycleFeatures
                dataDict[f'T{trialCounter}_masksIM'] = masksIM
                dataDict[f'T{trialCounter}_outlinesIM'] = outlineIM
                trialCounter += 1
        else:
            print('No ROIs segmented yet')
    return dataDict

def fft_rigid_cycle_moco_shifts(mIM, template):
    '''
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

def register_2ch_trials(expmtPath, regParams):
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
    
    fovPerTrial = expmtNotes['slice_label'].values
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

def sync_traces(expmtPath, dataDict):
    '''
    trials:
    dataDict.keys() = dict_keys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 'T0_masksIM', 'T0_outlinesIM', ect...])
    traces or features:
    dataDict[0].keys() = dict_keys(['cycle0_traces', 'cycle1_traces', ...ect...  'T0_roiFeatures'])
    traces == raw data
    features of rois
    dataDict[0]['T0_roiFeatures'].keys() = dict_keys(['roi1_redAvg', 'roi1_diameter', 'roi1_window', ... ect...
    
    returns traces dict
    traces.keys() = dict_keys([0, 'T0_roiOrder', ect...])
    shape of traces[trialNum]
    traces[0] --> (frame, roiNumber) roiNumber is from 0:len(rois), T0_roiOrder is the identity of each roi
    
    '''

    #establish notes and output dict
    expmtNotes = pd.read_excel(glob.glob(expmtPath+'/expmtNotes*.xlsx')[0])
    trialPaths = glob.glob(expmtPath+'/TSeries*')
    stimFrames = expmtNotes['stim_frame'].values
    fpsPerTrial = expmtNotes['frame_rate'].values
    trialPaths = glob.glob(expmtPath+'/TSeries*')
    traceDict = {}
    for trial in range(len(trialPaths)):
        if trial not in dataDict.keys():
            continue
        print(f'\rSyncing traces for trial {trial+1}/{len(trialPaths)}', end='',flush=True)
        try:
            stimFrame = stimFrames[trial]
            trialPath = trialPaths[trial]
            nCycles = len(glob.glob(trialPath+'/rT*_C*_ch2.tif'))
            gCaMPOnly = dataDict[trial][f'T{trial}_roiFeatures']['gCaMP_only_rois'] 
            colabeledROIs = dataDict[trial][f'T{trial}_roiFeatures']['colabeled_rois']
            if 0 in gCaMPOnly:
                gCaMPOnly = gCaMPOnly[1:]
            if 0 in colabeledROIs:
                colabeledROIs = colabeledROIs[1:]
            rois = np.unique(np.concat([colabeledROIs, gCaMPOnly]))
        except:
            print('error detected')
            continue
        
        #iterate through registered cycle traces
        trialTraceArray = []
        for cycleIDX in range(nCycles):
            intercycleinterval = 25
            rawROIs = np.array(dataDict[trial][f'cycle{cycleIDX}_traces'])
            #concatenating ventialtor is synced to microscope we concatenat trials with vent signal as last trace
            if stimFrame == 'voltage':
                fps = fpsPerTrial[trial]
                ventilatorSamplingRate = 10000 #will read xml in future to retrieve this, but this usually is pretty consistent
                downSampleVent = round(ventilatorSamplingRate/fps) #so for a 29.94 Hz recording this will sample vent trace every 334th frame
                voltageSignals = glob.glob(trialPath+'/TSeries*VoltageRecording*.csv')
                if 'baseline' in expmtNotes['stim_type'].values[trial]:
                    trialTraceArray.append(rawROIs)
                elif 'gas' in expmtNotes['stim_type'].values[trial]:
                    trialTraceArray.append(rawROIs)
                else:
                    ventilatorTrace = (((pd.read_csv(voltageSignals[cycleIDX]).iloc[:,3]>3.).astype(float)-2)/4)[::downSampleVent]
                    if ventilatorTrace.shape[0] == rawROIs.shape[1]:
                        rawTracesPadded = np.pad(rawROIs, ((0,0),(0,intercycleinterval)), mode='constant', constant_values=np.nan)
                        voltageTracePadded = np.pad(np.array(ventilatorTrace), (0,intercycleinterval), mode='constant', constant_values=np.nan)
                        addToTraces = np.vstack([rawTracesPadded, voltageTracePadded])
                        trialTraceArray.append(addToTraces)
                    else:
                        try:
                            rawTracesPadded = np.pad(rawROIs, ((0,0),(0,intercycleinterval)), mode='constant', constant_values=np.nan)
                            voltageTracePadded = np.pad(np.array(ventilatorTrace), (0,intercycleinterval), mode='constant', constant_values=np.nan)[1:]
                            print(f'\nTrial {trial+1} Sampling is off by 1\n')
                            addToTraces = np.vstack([rawTracesPadded, voltageTracePadded])
                            trialTraceArray.append(addToTraces)                            
                        except ValueError:
                            print('\n error with cycle - ommitting')

            else: #if manually recorded stim frame (it is not split into trials if it is this case)
                trialTraceArray.append(rawROIs)
        trialTrace = np.hstack(trialTraceArray).T
        traceDict[trial] = trialTrace
        traceDict[f'T{trial}_roiOrder'] = rois
    print(f'\r{expmtPath} synced!\n', end='', flush=True)
    return traceDict 

def compare_all_ROIs(conditionStr, trial, traces, notes, expmt):
    rawF = traces[trial].T
    xlabel = notes['frame_rate'][trial]

    if 'baseline' in conditionStr:
        
        f0 = np.nanmean(rawF[:40])
        dFF = (rawF - f0)/f0

        roiLabels = traces[f'T{trial}_roiOrder']
        fig, ax = plt.subplots()
        fig.suptitle(f'Trial {trial}\n{conditionStr}')
        ax.imshow(dFF, aspect='auto',  interpolation='none', cmap='Greens')
        ax.set_yticks(np.arange(len(roiLabels)))
        ax.set_yticklabels(roiLabels)
        ax.get_xaxis().set_visible(False)
        ax.set_xlabel(xlabel)
        fig.tight_layout()
    
    elif 'gas' in conditionStr:
        #gas will plot mean of each cycle
        # np.where()
        # print(np.where(np.diff(((np.isnan(rawF[1,:]).astype(int))*-1)+1)==1)[0][0])
        #
        #need to find a way to extract frames per cycle to better get this value
        #for now when this bugs out just hardcode this
        # jump = np.where(np.diff(((np.isnan(rawF[1,:]).astype(int))*-1)+1)==1)[0][0] #only works for 120 frame cycles
        jump = 120
        cycle = 0
        cycleAvgs = []
        for idx in range(jump, rawF.shape[1], jump):
            cycleTrace = rawF[:,cycle:cycle+jump]
            cycleAvgs.append(np.nanmean(cycleTrace, axis=1))
            cycle+=jump
        rawAvgs = np.array(cycleAvgs).T
        xAxis = np.arange(rawAvgs.shape[1])
        normalizedDFF = (rawAvgs - np.nanmean(rawAvgs, axis=0)) / np.nanstd(rawAvgs, axis=0)
        roiLabels = traces[f'T{trial}_roiOrder']
        fig, ax = plt.subplots()
        fig.suptitle(f'Trial {trial}\n{conditionStr} (Avg Cycle)')
        ax.imshow(normalizedDFF, aspect='auto',  interpolation='none', cmap='Greens')
        ax.set_yticks(np.arange(0, len(roiLabels), 5))
        ax.set_yticklabels(roiLabels[::5])
        
        ax.set_xlabel('Cycle Number')
        ax.set_xticklabels(xAxis)
        ax.set_xticks(xAxis)

        ax.axvline(2-.5, label='Delivery', color='blue')
        ax.axvline(4-.5, label='Basal', color='black')
        ax.legend(bbox_to_anchor=(1.25,1.05))
        fig.tight_layout()  


        return fig


    else: #assuming all other types of trials are mechanical stim trials
        if notes['stim_frame'].values[trial]=='voltage':
            stimFrame = find_stim_frame(traces[trial][:,-1], conditionStr)
            sync = True
        else:
            stimFrame = notes['stim_frame'][trial]
            sync = False
        #TODO: will need to generalize this for any type of 2p experiment...
        
        if 'mech_galvo' in expmt:
            vLine = 20
            stepping = 5
            if stimFrame>=21:
                beggining = 20
            else:
                beggining = 0
            if stimFrame+30 > rawF.shape[1]:
                end = rawF.shape[1] - stimFrame
            else:
                end = 30
        else:
            stepping = 50
            refTrace = rawF[-1,:]
            if stimFrame >=150:
                beggining = 150
            else:
                beggining = 0
            
            if stimFrame + 300 > len(refTrace):
                end = len(refTrace) - stimFrame
            else:
                end = 300
        if sync:
            if beggining == 0:
                f0 = np.nanmean(rawF[:-1,beggining:stimFrame], axis=1) 
                f0 = np.reshape(f0, (f0.shape[0],1))
                plottingF = rawF[:-1,beggining:stimFrame+end]
                ventTrace = ((rawF[-1,beggining:stimFrame+end])+.5)*4
                xAxis = np.arange(-stimFrame, end,stepping)
                
            else:
                f0 = np.nanmean(rawF[:-1,stimFrame -beggining:stimFrame], axis=1)
                f0 = np.reshape(f0, (f0.shape[0],1))
                plottingF = rawF[:-1,stimFrame - beggining:stimFrame+end]
                ventTrace = ((rawF[-1,stimFrame - beggining:stimFrame+end])+.5)*4
                xAxis = np.arange(-beggining, end,stepping)
                vLine = beggining
            dFF = (plottingF - f0)/f0
            if dFF.shape[0] <= 2:
                normalizedDFF = dFF*5
                normalizedDFF = np.vstack([normalizedDFF,ventTrace])
            else:
                normalizedDFF = (dFF - np.nanmean(dFF, axis=0))/ (np.nanstd(dFF, axis=0))
                normalizedDFF = np.vstack([normalizedDFF,ventTrace])
        else:
            f0 = np.nanmean(rawF[:,beggining:stimFrame], axis=1) ##
            f0 = np.reshape(f0, (len(f0),1))
            plottingF = rawF[:,stimFrame - beggining:stimFrame+end]
            dFF = (plottingF - f0)/f0
            normalizedDFF = (dFF - np.nanmean(dFF, axis=0))/ (np.nanstd(dFF, axis=0))
            xAxis = np.arange(-beggining, end,stepping)
            
        roiLabels = traces[f'T{trial}_roiOrder']
        roiSteps = round(len(roiLabels)*.1)
        if roiSteps == 0:
            roiSteps = 1
        fig, ax = plt.subplots()
        fig.suptitle(f'Trial {trial}\n{conditionStr}')
        im = ax.imshow(normalizedDFF, aspect='auto',  interpolation='none', cmap='Greens')
        vLine = beggining
        ax.axvline(vLine, color='black')

        ax.set_yticks(np.arange(0,len(roiLabels), roiSteps))
        ax.set_yticklabels(roiLabels[::roiSteps])

        ax.set_xticklabels(xAxis)
        ax.set_xticks(np.arange(0,normalizedDFF.shape[1], stepping))
        ax.get_xaxis().set_visible(True)
        fig.colorbar(im, ax=ax)
        fig.supxlabel(f'Frames (fps={xlabel})')
        fig.supylabel('ROI Num')
        fig.tight_layout()

    return fig

def response_distribution(conditionStr, trial, traces, notes):
    '''
    Only call function if mechanical stimulation occurs, this function will
    group cells based off their response characteristics
    '''
    rawF = traces[trial].T
    rawF = rawF[:-1]#drop vent trace
    fps = notes['frame_rate'][trial]
    if notes['stim_frame'].values[trial]=='voltage':
        stimFrame = find_stim_frame(traces[trial][:,-1], conditionStr)
    else:
        stimFrame = notes['stim_frame'][trial]
    
    
    if fps <= 4:
        beggining = 20 if stimFrame >= 21 else 0
        end = (len(rawF) - stimFrame) if (stimFrame+30>len(rawF)) else 30
    else:
        beggining = 150 if stimFrame >=150 else 0
        end = (len(rawF)-stimFrame) if (stimFrame+300 > len(rawF)) else 300
    
    f0 = np.nanmean(rawF[beggining:stimFrame])
    if beggining == 0:
        psthWindowAVG_pre = np.nanmean(rawF[beggining:stimFrame], axis=1)
        psthWindowAVG_post = np.nanmean(rawF[stimFrame:stimFrame+end], axis=1)
    else:
        psthWindowAVG_pre = np.nanmean(rawF[stimFrame - beggining:stimFrame], axis=1)
        psthWindowAVG_post = np.nanmean(rawF[stimFrame:stimFrame+end], axis=1)
    
    changeDiff = psthWindowAVG_post - psthWindowAVG_pre
    print(changeDiff.shape)
    plt.bar(changeDiff)
    rois = traces[f'T{trial}_roiOrder']
    plt.xticks(rois)
    plt.xlabel('ROI ID')

def summerize_experiment(expmtPath, dataDict):
    '''
    traces.keys() = 0, 'T0_roiOrder', 1, 'T1_roiOrder', ect...
    traces[0] = np.array([
        [roi1],
        [roi2],
        [roi3],
        ect...
    ])
    traces['T0_roiOrder'] =  np.array([
        2,
        14, 
        15,
        ect...
    ])
    '''

    #preparing data to be in a trialized format
    traces = sync_traces(expmtPath, dataDict)

    #read manually entered metadata
    expmtNotes = pd.read_excel(glob.glob(expmtPath+'/expmtNotes*')[0])
    slicePerTrial = expmtNotes['slice_label'].values
    trialSets = np.unique(slicePerTrial)
    conditions = expmtNotes['stim_type'].values
    if -1 in trialSets:
        trialSets = trialSets[trialSets != -1]
    trialPaths = np.array(glob.glob(expmtPath+'/TSeries*'))
    nTrials = len(trialPaths)
    trialIndices = np.arange(nTrials)

    print('Plotting...\n')
    #iterate through FOVs that are defined as slices in manually curated metadata notes
    for trialSet in trialSets:
        
        try:
            #Load Segmentations - dependent on method set up as of 6-5-25
            segFN = glob.glob(expmtPath+f'/cellCountingTiffs/*slice{trialSet}_seg.npy')[0]
            masksLoaded = np.load(segFN, allow_pickle=True).item()
            masks = masksLoaded['masks']
            gcampROIs = np.unique(masks[2,:,:])[1:]
            colabeledROIs = np.unique(masks[1,:,:])[1:]
            gcampCenters = [center_of_mass(masks[2,:,:]==roi) for roi in gcampROIs]
            colabeledCenters = [center_of_mass(masks[1,:,:]==roi) for roi in colabeledROIs]
            firstTrialInSet = trialIndices[slicePerTrial==trialSet][0]
            maskIM = dataDict[f'T{firstTrialInSet}_masksIM']
            if maskIM.shape[0] == 3:
                maskIM = np.permute_dims(maskIM, (1,2,0))
            outlinesIM = dataDict[f'T{firstTrialInSet}_outlinesIM']
            outlinesIM = outlinesIM*2.2
            if outlinesIM.shape[0] == 3:
                outlinesIM = np.permute_dims(outlinesIM, (1,2,0))
        except IndexError:
            print('Ignoring Placeholder Sets')
            pdfSummary.close()
            return
        
        #Set Up Save Dir
        figureDR = Path(expmtPath)/'figures'
        figureDR.mkdir(parents=True, exist_ok=True)
        saveFN = str(figureDR / f'slice{trialSet}_summary.pdf')
        pdfSummary = PdfPages(saveFN)


        try:

            #Set Up ROI IMSHOW
            fig, ax = plt.subplots()
            ax.imshow(maskIM)
            for roi in range(len(gcampROIs)):
                ax.text(gcampCenters[roi][1], gcampCenters[roi][0], f'{gcampROIs[roi]}', color='black', size=5, label='WGA-')
            for roi in range(len(colabeledROIs)):
                ax.text(colabeledCenters[roi][1], colabeledCenters[roi][0], f'{colabeledROIs[roi]}', color='#FF4500', size=5)
            
            textColors = [
                Patch(facecolor='#FF4500', edgecolor='#FF4500', label='WGA+'),
                Patch(facecolor='black', edgecolor='black', label='WGA-')
            ]
        
            ax.set_title('Mask Overlayed on GCaMP+ Cells')
            ax.legend(handles=textColors, loc='upper right', title='ROI Colors')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            fig.suptitle(f'Slice {trialSet} Segmentations')
            plt.savefig(pdfSummary, format='pdf')

            #Set up ROI Outlines around WGA Channel
            fig, ax = plt.subplots()
            ax.imshow(outlinesIM)
            ax.axis='off'
            ax.set_title('ROI Outlines Overlayed on WGA Channel')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            fig.suptitle(f'Slice {trialSet} Segmentations')
            plt.savefig(pdfSummary, format='pdf')
            
            #Plot ROIs together according to condition
            for trialIDX in trialIndices[slicePerTrial==trialSet]:
                fig = compare_all_ROIs(conditions[trialIDX],trialIDX,traces, expmtNotes, expmtPath)
                plt.savefig(pdfSummary, format='pdf')


            
            #Plot individual ROIs according to condition
            trialsBool = slicePerTrial==trialSet
            plotsInFOV =len(np.where(trialsBool.astype(int))[0])
            fovBool = trialsBool
            rois = np.arange(len(traces[f'T{firstTrialInSet}_roiOrder']))
            for roi in rois:
                if plotsInFOV>6:
                    trialsUsing = np.where(fovBool.astype(int))[0]
                    blocks = np.where(fovBool.astype(int))[0][::6]
                    for ts in range(len(blocks)):
                        trialsBool = np.zeros((len(slicePerTrial),))
                        if blocks[ts] == blocks[-1]:
                            trialsBool[trialsUsing[np.where(trialsUsing==blocks[ts])[0][0]:]] = 1
                            fig = analyze_roi_across_conditions( trialsBool.astype(bool), roi, traces, expmtNotes, gcampROIs)
                            plt.savefig(pdfSummary, format='pdf')
                        else:
                            loci = np.where(trialsUsing==blocks[ts+1])[0][0]
                            trialsBool[trialsUsing[ts:loci]]=1
                            fig = analyze_roi_across_conditions( trialsBool.astype(bool), roi, traces, expmtNotes, gcampROIs)
                            plt.savefig(pdfSummary, format='pdf')
                else:
                    fig = analyze_roi_across_conditions( trialsBool, roi, traces, expmtNotes, gcampROIs)
                    plt.savefig(pdfSummary, format='pdf')
                
                
        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()
            pdfSummary.close() 
            print('Plotting Error')

        pdfSummary.close() 
            
def find_stim_frame(ventilatorTrace, condition):
    edges = ventilatorTrace - np.roll(ventilatorTrace, -1)
    risingIDX = np.where(edges<0)[0]
    fallingIDX = np.where(edges>0)[0]
    if '5e' in condition:
        exspLengths = fallingIDX - np.roll(fallingIDX,1)
        longestExsp = np.unique(exspLengths)[-1]
        stimFrame = fallingIDX[np.where(exspLengths == longestExsp)[0]-1]
    elif 'baseline' in condition:
        stimFrame = None
        return
    else:
        inspLengths = risingIDX - np.roll(risingIDX, 1)
        longestInsp = np.unique(inspLengths)[-1]
        stimFrame = risingIDX[np.where(inspLengths == longestInsp)[0]-1]
    return stimFrame[0]

def analyze_roi_across_conditions(trialsBool, roiChoice, traces, notes, gcampROIs):
    '''
    trialsBool = [0,0,0,0,1,1,1,1,0,0,0,0], where len(trialsBool) = number of trials
    roiChoice = int(21)
    traces = dictionary{
        int(trial): np.array([t, nROIS])
    }
    '''
    uL = 0.75 #setting to set default upper limit
    lL = -0.25 #default for setting lower limit
    trialsBool = trialsBool.astype(bool)
    #Extracting MetaData from Only Relevant Trials
    nTrials = len(trialsBool)
    trialIndices = np.arange(nTrials)[trialsBool]
    nConditions = len(np.where(trialsBool==True)[0])
    conditions = notes['stim_type'].values[trialsBool]
    frameRate = notes['frame_rate'].values[trialsBool]



    #plotting roi according to condition
    if nConditions>1:
        fig, ax = plt.subplots(nConditions, 1, sharex=False, constrained_layout=True)
        

        for conditionIDX in range(nConditions):
            fps = frameRate[conditionIDX]
            conditionStr = conditions[conditionIDX]
            trial = trialIndices[conditionIDX]
            trialROIs = traces[f'T{trial}_roiOrder']
            roi = trialROIs[roiChoice]
            fig.suptitle(f'ROI {roi}')
            rawF = traces[trial][:,roiChoice].T
            if roi in gcampROIs:
                plottingColor = '#8A2BE2'
            else:
                plottingColor = '#32CD32'
            if 'baseline' in conditionStr:
                f0 = np.nanmean(rawF[:round(len(rawF)/4)])
                dFF = (rawF - f0)/f0
                upperLimit = max(uL, max(dFF))
                lowerLimit = min(lL, min(dFF))
                ax[conditionIDX].plot(dFF, color=plottingColor)
                ax[conditionIDX].axhline(0, color='black', alpha=0.5)
                ax[conditionIDX].set_ylabel(f'{conditionStr}\n({fps} fps)', fontsize=8)
                ax[conditionIDX].set_ylim([lowerLimit, upperLimit])
            
            elif 'gas' in conditionStr:
                f0 = np.nanmean(rawF[:120]) #baseline is the basal condition cycle
                dFF = (rawF - f0)/f0
                upperLimit = max(uL, max(dFF))
                lowerLimit = min(lL, min(dFF))
                ax[conditionIDX].plot(dFF, color=plottingColor)
                ax[conditionIDX].axhline(0, color='black', alpha=0.5)
                ax[conditionIDX].set_ylabel(f'{conditionStr}\n({fps} fps)', fontsize=8)
                ax[conditionIDX].set_ylim([lowerLimit, upperLimit])

            #currently assuming all else is mechanical stim trials
            else: 
                if notes['stim_frame'].values[trial]=='voltage':
                    stimFrame = find_stim_frame(traces[trial][:,-1], conditionStr)
                else:
                    stimFrame = notes['stim_frame'][trial]
                #currently logic is based off manually curated metadata
                if fps <= 4:
                    beggining = 20 if stimFrame>= 21 else 0
                    end = (len(rawF) - stimFrame) if (stimFrame+30>len(rawF)) else 30
                else:
                    beggining = 150 if stimFrame >=150 else 0
                    end = (len(rawF)-stimFrame) if (stimFrame+300 > len(rawF)) else 300
                f0 = np.nanmean(rawF[beggining:stimFrame])
                if beggining == 0:
                    plottingF = rawF[beggining:stimFrame+end]
                    xAxis = np.arange(-stimFrame, end)
                    ax[conditionIDX].set_xlim([-stimFrame, end])
                else:
                    plottingF = rawF[stimFrame - beggining:stimFrame+end]
                    xAxis = np.arange(-beggining, end)
                    ax[conditionIDX].set_xlim([-beggining, end])

                dFF = (plottingF - f0)/f0
                upperLimit = max(uL, max(dFF))
                lowerLimit = min(lL, min(dFF))
                ax[conditionIDX].axvline(0, color='black', alpha=0.2)
                ax[conditionIDX].plot(xAxis, dFF, color=plottingColor)
                ax[conditionIDX].set_ylim([lowerLimit, upperLimit])
                ax[conditionIDX].axhline(0, color='black', alpha=0.2)
                ax[conditionIDX].set_ylabel(f'{conditionStr}\n({fps} fps)', fontsize=8)

    else: #assuming single condition trial sets are NOT mechanical stimulation trials... because whats the point then?
        fig, ax = plt.subplots()
        trial = trialIndices[0]
        trialROIs = traces[f'T{trial}_roiOrder']
        roi = trialROIs[roiChoice]
        fig.suptitle(f'ROI {roi}')
        if roi in gcampROIs:
            plottingColor = '#8A2BE2'
        else:
            plottingColor = '#32CD32'
        trial = trialIndices[0]
        fps = frameRate[0]
        conditionStr = conditions[0]
        trial = trialIndices[0]
        rawF = traces[trial][:,roiChoice].T
        if 'baseline' in conditionStr:
                f0 = np.nanmean(rawF[:round(len(rawF)/4)])
                dFF = (rawF - f0)/f0
                upperLimit = max(uL, max(dFF))
                lowerLimit = min(lL, min(dFF))
                ax.plot(dFF, color=plottingColor)
                ax.axhline(0, color='black', alpha=0.5)
                ax.set_ylabel(f'{conditionStr}\n({fps} fps)', fontsize=8)
                ax.set_ylim([lowerLimit, upperLimit])
        elif 'gas' in conditionStr:
            f0 = np.nanmean(rawF[:120]) #baseline is the basal condition cycle
            dFF = (rawF - f0)/f0
            upperLimit = max(uL, max(dFF))
            lowerLimit = min(lL, min(dFF))
            ax.plot(dFF, color=plottingColor)
            ax.axhline(0, color='black', alpha=0.5)
            ax.set_ylabel(f'{conditionStr}\n({fps} fps)', fontsize=8)
            ax.set_ylim([lowerLimit, upperLimit])

    return fig

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
        trialMeta['datetime'] = root.attrib.get('date')
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

        #Clearing just to be confident it is cleared
        del root
        del child

        #extract zStack/ZSeries metadata 
        zstackMeta = {}
        stackPath = glob.glob(expmt+'/ZSeries*')[0]
        metaDataFN = os.path.join(
            stackPath, 
            [FN for FN in os.listdir(stackPath) if 'VoltageRecording' not in FN and '.xml' in FN][0]
            )
        
        try:
            md= ET.parse(metaDataFN)
        except FileNotFoundError:
            print(f'Error Extracting ZStack Metadata')
            metaData['zstack'] = {}
            return metaData
        root = md.getroot()
        
        #Extracting PVStateShard for ZSeries, state shard is nested in root[1] always
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

        #Extracting motor positions of each slice, nested in root[2] always
        zstackMeta['nSlices'] = len(root[2])-1 #subtracting the first tage <PVShard />
        
        sliceNum = 0
        for child in root[2]:
            if child.tag =='Frame':
                sliceMotorPos = {}
                for idx in range(len(child[3][0])):
                    sliceMotorPos[child[3][0][idx].attrib.get('index')] = child[3][0][idx][0].attrib.get('value')
                zstackMeta[sliceNum] = sliceMotorPos
            sliceNum+=1

        metaData['zstack'] = zstackMeta
    return metaData
                
