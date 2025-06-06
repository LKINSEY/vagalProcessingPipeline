#%%
import tifffile as tif
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import center_of_mass
import matplotlib
matplotlib.use('WebAgg') #display in web browser (requires a plt.show() and plt.close('all') to reset view)
# %matplotlib inline #go back to displaying inline
import pandas as pd
import os,glob, pickle
from matplotlib.patches import Patch

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
                ventilatorTrace = (((pd.read_csv(voltageSignals[cycleIDX]).iloc[:,3]>3.).astype(float)-2)/4)[::downSampleVent]
                if ventilatorTrace.shape[0] == rawROIs.shape[1]:
                    rawTracesPadded = np.pad(rawROIs, ((0,0),(0,intercycleinterval)), mode='constant', constant_values=np.nan)
                    voltageTracePadded = np.pad(np.array(ventilatorTrace), (0,intercycleinterval), mode='constant', constant_values=np.nan)
                else:
                    print('Sampling is off by 1')
                    rawTracesPadded = np.pad(rawROIs, ((0,0),(0,intercycleinterval)), mode='constant', constant_values=np.nan)
                    voltageTracePadded = np.pad(np.array(ventilatorTrace), (0,intercycleinterval), mode='constant', constant_values=np.nan)[1:]
                addToTraces = np.vstack([rawTracesPadded, voltageTracePadded])
                trialTraceArray.append(addToTraces)
            else: #if manually recorded stim frame (it is not split into trials if it is this case)
                trialTraceArray.append(rawROIs)
        trialTrace = np.hstack(trialTraceArray).T
        traceDict[trial] = trialTrace
        traceDict[f'T{trial}_roiOrder'] = rois
    print(f'\r{expmtPath} synced!\n', end='', flush=True)
    return traceDict 

def compare_all_ROIs(conditionStr, trial, traces, notes, expmt):
    xlabel = notes['frame_rate'][trial]
    if conditionStr == 'baseline':
        rawF = traces[trial].T
        f0 = np.nanmean(rawF[:40])
        dFF = (rawF - f0)/f0
        normalizedDFF = (dFF - np.nanmean(dFF, axis=0)) / np.nanstd(dFF, axis=0)

        roiLabels = traces[f'T{trial}_roiOrder']
        fig, ax = plt.subplots()
        fig.suptitle(f'Trial {trial}\n{conditionStr}')
        ax.imshow(normalizedDFF, aspect='auto',  interpolation='none', cmap='Greens')
        ax.set_yticks(np.arange(len(roiLabels)))
        ax.set_yticklabels(roiLabels)
        ax.get_xaxis().set_visible(False)
        ax.set_xlabel(xlabel)
        fig.tight_layout()
    else:
        if notes['stim_frame'].values[0]=='voltage':
            stimFrame = find_stim_frame(traces[trial][:,-1], conditionStr)
        else:
            stimFrame = notes['stim_frame'][trial]
        rawF = traces[trial].T
        #TODO: will need to generalize this for any type of 2p experiment...
        if 'mech_galvo' in expmt:
            stepping = 5
            if stimFrame>=21:
                beggining = 20
            else:
                beggining = len(rawF[:stimFrame])
            if stimFrame+30 > rawF.shape[1]:
                end = rawF.shape[1] - stimFrame
            else:
                end = 30
        else:
            stepping = 50
            voltageTrace = rawF[-1,:]
            if stimFrame >=150:
                beggining = 150
            else:
                beggining = len(voltageTrace[:stimFrame])
            
            if stimFrame + 300 > len(voltageTrace):
                end = len(voltageTrace) - stimFrame
            else:
                end = 300

        f0 = np.nanmean(rawF[:,beggining:stimFrame], axis=1) ##
        f0 = np.reshape(f0, (len(f0),1))
        plottingF = rawF[:,stimFrame - beggining:stimFrame+end]
        print('beggining', beggining, 'end', end)
        dFF = (plottingF - f0)/f0
        normalizedDFF = (dFF - np.nanmean(dFF, axis=0))/ (np.nanstd(dFF, axis=0))
        roiLabels = traces[f'T{trial}_roiOrder']
        roiSteps = round(len(roiLabels)*.1)

        fig, ax = plt.subplots()
        fig.suptitle(f'Trial {trial}\n{conditionStr}')
        im = ax.imshow(normalizedDFF, aspect='auto',  interpolation='none', cmap='Greens')
        ax.axvline(beggining, color='black')

        ax.set_yticks(np.arange(0,len(roiLabels), roiSteps))
        ax.set_yticklabels(roiLabels[::roiSteps])

        xAxis = np.arange(-beggining, end,stepping)
        ax.set_xticklabels(xAxis)
        ax.set_xticks(np.arange(0,normalizedDFF.shape[1], stepping))
        ax.get_xaxis().set_visible(True)
        fig.colorbar(im, ax=ax)
        fig.supxlabel(f'Frames (fps={xlabel})')
        fig.supylabel('ROI Num')
        fig.tight_layout()

    return fig

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
    traces = sync_traces(expmtPath, dataDict)
    expmtNotes = pd.read_excel(glob.glob(expmtPath+'/expmtNotes*')[0])
    slicePerTrial = expmtNotes['slice_label'].values
    trialSets = np.unique(slicePerTrial)
    conditions = expmtNotes['stim_type'].values
    if -1 in trialSets:
        trialSets = trialSets[1:]
    trialPaths = np.array(glob.glob(expmtPath+'/TSeries*'))
    nTrials = len(trialPaths)
    trialIndices = np.arange(nTrials)
    # setCounter = 1
    print('Plotting...\n')
    for trialSet in trialSets:
        trialsInSet = trialPaths[slicePerTrial==trialSet]
        segFN = glob.glob(expmtPath+f'/cellCountingTiffs/*slice{trialSet}_seg.npy')[0]
        masksLoaded = np.load(segFN, allow_pickle=True).item()
        masks = masksLoaded['masks']
        #dependent on method set up as of 6-5-25
        gcampROIs = np.unique(masks[2,:,:])[1:]
        colabeledROIs = np.unique(masks[1,:,:])[1:]
        gcampCenters = [center_of_mass(masks[2,:,:]==roi) for roi in gcampROIs]
        colabeledCenters = [center_of_mass(masks[1,:,:]==roi) for roi in colabeledROIs]
        print('/n ', trialIndices[slicePerTrial==trialSet])
        firstTrialInSet = trialIndices[slicePerTrial==trialSet][0]
        if os.path.exists(expmt+'/figures/'):
            pdfSummary = PdfPages(expmtPath+f'/figures/slice{trialSet}_summary.pdf')
        else:
            os.mkdir(expmt+'/figures/')
            pdfSummary = PdfPages(expmtPath+f'/figures/slice{trialSet}_summary.pdf')
        maskIM = dataDict[f'T{firstTrialInSet}_masksIM']
        if maskIM.shape[0] == 3:
            maskIM = np.permute_dims(maskIM, (1,2,0))
        outlinesIM = dataDict[f'T{firstTrialInSet}_outlinesIM']
        outlinesIM = outlinesIM*2.2
        if outlinesIM.shape[0] == 3:
            outlinesIM = np.permute_dims(outlinesIM, (1,2,0))

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


        fig, ax = plt.subplots()
        ax.imshow(outlinesIM)
        ax.axis='off'
        ax.set_title('ROI Outlines Overlayed on WGA Channel')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.suptitle(f'Slice {trialSet} Segmentations')
        plt.savefig(pdfSummary, format='pdf')
        
        
        for trialIDX in trialIndices[slicePerTrial==trialSet]:
            fig = compare_all_ROIs(conditions[trialIDX],trialIDX,traces, expmtNotes, expmt)
            plt.savefig(pdfSummary, format='pdf')
        
        trialsBool = slicePerTrial==trialSet
        rois = np.arange(traces[firstTrialInSet].shape[1])
        for roi in rois:
            fig = analyze_roi_across_conditions(expmtPath, trialsBool, roi, traces, expmtNotes, gcampROIs, colabeledROIs)
            plt.savefig(pdfSummary, format='pdf')
        
        pdfSummary.close()
            
def find_stim_frame(ventilatorTrace, condition):
    edges = ventilatorTrace - np.roll(ventilatorTrace, -1)
    risingIDX = np.where(edges<0)[0]
    fallingIDX = np.where(edges>0)[0]
    if condition == '5e':
        exspLengths = fallingIDX - np.roll(fallingIDX,1)
        longestExsp = np.unique(exspLengths)[-1]
        stimFrame = fallingIDX[np.where(exspLengths == longestExsp)[0]-1]
    elif condition=='baseline':
        stimFrame = None
        return
    else:
        inspLengths = risingIDX - np.roll(risingIDX, 1)
        longestInsp = np.unique(inspLengths)[-1]
        stimFrame = risingIDX[np.where(inspLengths == longestInsp)[0]-1]
    return stimFrame[0]


def analyze_roi_across_conditions(expmtPath, trialsBool, roi, traces, notes, gcampROIs, colabeledROIs):
    nTrials = len(trialsBool)
    trialIndices = np.arange(nTrials)[trialsBool]
    nConditions = len(np.where(trialsBool==True)[0])
    conditions = notes['stim_type'].values[trialsBool]
    frameRate = notes['frame_rate'].values[trialsBool]
    fig, ax = plt.subplots(nConditions, 1, sharex=False, constrained_layout=True)
    if roi in gcampROIs:
        wgaPositive = False
    else:
        wgaPositive = True
    for condition in range(nConditions):
        fps = frameRate[condition]
        conditionStr = conditions[condition]
        trial = trialIndices[condition]
        # trialROIs = traces[f'T{trial}_roiOrder']
        rawF = traces[trial][:,roi].T #1D
        if conditionStr == 'baseline':
            f0 = np.nanmean(rawF[:round(len(rawF)/4)])
            dFF = (rawF - f0)/f0
            # normalizedDFF = (dFF - np.nanmean(dFF)) / np.nanstd(dFF)
            fig.suptitle(f'Trial {trial}\n{conditionStr}')
            if wgaPositive:
                ax[condition].plot(dFF, color='#32CD32')
            else:
                ax[condition].plot(dFF, color='#8A2BE2')
            
            ax[condition].axhline(0, color='black')
            ax[condition].set_ylabel(f'{conditionStr}\n{fps}')
        else:
            if notes['stim_frame'].values[0]=='voltage':
                stimFrame = find_stim_frame(traces[trial][:,-1], conditionStr)
            else:
                stimFrame = notes['stim_frame'][trial]
            #TODO: will need to generalize this for any type of 2p experiment...
            if 'mech_galvo' in expmt:
                if stimFrame>=21:
                    beggining = 20
                else:
                    beggining = len(rawF[:stimFrame])
                if stimFrame+30 > len(rawF):
                    end = len(rawF) - stimFrame
                else:
                    end = 30
            else:
                voltageTrace = rawF[-1,:]
                if stimFrame >=150:
                    beggining = 150
                else:
                    beggining = len(voltageTrace[:stimFrame])
                
                if stimFrame + 300 > len(voltageTrace):
                    end = len(voltageTrace) - stimFrame
                else:
                    end = 300
            xAxis = np.arange(-beggining, end)
            f0 = np.nanmean(rawF[beggining:stimFrame])
            plottingF = rawF[stimFrame - beggining:stimFrame+end]
            dFF = (plottingF - f0)/f0
            # normalizedDFF = (dFF - np.nanmean(dFF, axis=0))/ (np.nanstd(dFF, axis=0))
            if wgaPositive:
                ax[condition].plot(xAxis, dFF, color='#32CD32')
            else:
                ax[condition].plot(xAxis, dFF, color='#8A2BE2')
            ax[condition].axhline(0, color='black', alpha=0.2)
            ax[condition].axvline(0, color='black', alpha=0.2)
            ax[condition].set_ylabel(f'Condition: {conditionStr}\n Frame Rate: {fps}')
            ax[condition].set_xlim([-beggining, end])





#%

if __name__=='__main__':
    dataFrom = [
        'U:/expmtRecords/Lucas*',
        'C:/Analysis/april_data/Lucas*',
        'U:/expmtRecords/res_galvo/Lucas*',
        'U:/expmtRecords/mech_galvo/Lucas*',
        ]
    expmtRecords = glob.glob(dataFrom[3])
    plt.close('all')
    for expmt in expmtRecords:
        print('Loading Traces for \n', expmt)
        if os.path.exists(expmt+'/expmtTraces.pkl'):
            if os.path.exists(expmt+'/figures/'):
                print('Figures already generated for this experiment')
            else:
                with open(expmt+'/expmtTraces.pkl', 'rb') as f:
                    dataDict = pickle.load(f)
                plt.close('all')
                summerize_experiment(expmt, dataDict)


                print('Saved Plots...')
        else:
            print('Experiment not processed yet')
        print('******************* NEXT *******************')

# %%
