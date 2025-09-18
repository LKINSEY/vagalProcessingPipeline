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
        trialTraceArray_red = []
        for cycleIDX in range(nCycles):
            intercycleinterval = 25
            rawROIs = np.array(dataDict[trial][f'cycle{cycleIDX}_traces'])
            # rawROIs_red = np.array(dataDict[trial][f'cycle{cycleIDX}_traces_red'])
            #concatenating ventialtor is synced to microscope we concatenat trials with vent signal as last trace
            if stimFrame == 'voltage':
                fps = fpsPerTrial[trial]
                ventilatorSamplingRate = 10000 #will read xml in future to retrieve this, but this usually is pretty consistent
                downSampleVent = round(ventilatorSamplingRate/fps) #so for a 29.94 Hz recording this will sample vent trace every 334th frame
                voltageSignals = glob.glob(trialPath+'/TSeries*VoltageRecording*.csv')
                if 'baseline' in expmtNotes['stim_type'].values[trial]:
                    trialTraceArray.append(rawROIs)
                    # trialTraceArray_red.append(rawROIs_red)
                elif 'gas' in expmtNotes['stim_type'].values[trial]:
                    trialTraceArray.append(np.pad(rawROIs, ((0,0),(0,intercycleinterval)), mode='constant', constant_values=np.nan)) #pad to show time seperation 
                    # trialTraceArray_red.append(np.pad(rawROIs_red, ((0,0),(0,intercycleinterval)), mode='constant', constant_values=np.nan))
                else:
                    ventilatorTrace = (((pd.read_csv(voltageSignals[cycleIDX]).iloc[:,3]>3.).astype(float)-2)/4)[::downSampleVent]
                    if ventilatorTrace.shape[0] == rawROIs.shape[1]:
                        rawTracesPadded = np.pad(rawROIs, ((0,0),(0,intercycleinterval)), mode='constant', constant_values=np.nan)
                        # rawTracesPadded_red = np.pad(rawROIs_red, ((0,0),(0,intercycleinterval)), mode='constant', constant_values=np.nan)
                        voltageTracePadded = np.pad(np.array(ventilatorTrace), (0,intercycleinterval), mode='constant', constant_values=np.nan)
                        addToTraces = np.vstack([rawTracesPadded, voltageTracePadded])
                        # addToTraces_red = np.vstack([rawTracesPadded_red, voltageTracePadded])
                        trialTraceArray.append(addToTraces)
                        # trialTraceArray_red.append(addToTraces_red)
                    else:
                        try:
                            rawTracesPadded = np.pad(rawROIs, ((0,0),(0,intercycleinterval)), mode='constant', constant_values=np.nan)
                            # rawTracesPadded_red = np.pad(rawROIs_red, ((0,0),(0,intercycleinterval)), mode='constant', constant_values=np.nan)
                            voltageTracePadded = np.pad(np.array(ventilatorTrace), (0,intercycleinterval), mode='constant', constant_values=np.nan)[1:]
                            print(f'\nTrial {trial+1} Sampling is off by 1\n')
                            addToTraces = np.vstack([rawTracesPadded, voltageTracePadded])
                            # addToTraces_red = np.vstack([rawTracesPadded_red, voltageTracePadded])
                            trialTraceArray.append(addToTraces)    
                            # trialTraceArray_red.append(addToTraces_red)                     
                        except ValueError:
                            print('\n error with cycle - ommitting')

            else: #if manually recorded stim frame (it is not split into trials if it is this case)
                trialTraceArray.append(rawROIs)
                # trialTraceArray_red.append(rawROIs_red)
        trialTrace = np.hstack(trialTraceArray).T
        # trialTrace_red = np.hstack(trialTraceArray_red).T
        traceDict[trial] = trialTrace
        # traceDict[f'{trial}_iso'] = trialTrace_red
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
            # isoBestic = traces[f'{trial}_iso'][:,roiChoice].T
            if roi in gcampROIs:
                plottingColor = '#8A2BE2'
            else:
                plottingColor = '#32CD32'
            # withoutISOCorrection = "#747070"
            if 'baseline' in conditionStr:


                f0 = np.nanmean(rawF[:round(len(rawF)/4)])
                # f0_iso = np.nanmean(rawF[:round(len(isoBestic)/4)])
                dFF = (rawF - f0)/f0
                # dFF_iso = (isoBestic - f0_iso)/f0
                # isoCorrection = 
                upperLimit = max(uL, max(dFF))
                lowerLimit = min(lL, min(dFF))
                ax[conditionIDX].plot(dFF, color=plottingColor, label='Without Iso Correction') #keep for now... 
                ax[conditionIDX].plot(dFF, color=plottingColor, label='Without Iso Correction')
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