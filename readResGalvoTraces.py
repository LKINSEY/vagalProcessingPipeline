#%%
import tifffile as tif
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import center_of_mass
# import matplotlib
# matplotlib.use('WebAgg') #display in web browser (requires a plt.show() and plt.close('all') to reset view)
# %matplotlib inline #go back to displaying inline
import pandas as pd
import os,glob, pickle

def compress_traces(expmtPath, dataDict):
    '''
    trials:
    dataDict.keys() = dict_keys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    traces or features:
    dataDict[0].keys() = dict_keys(['cycle0_traces', 'cycle1_traces', ...ect...  'T0_roiFeatures'])
    features of rois
    dataDict[0]['T0_roiFeatures'].keys() = dict_keys(['roi1_redAvg', 'roi1_diameter', 'roi1_window', ... ect...
    '''
    expmtNotes = pd.read_excel(glob.glob(expmtPath+'/expmtNotes*.xlsx')[0])
    slicesForTrial = expmtNotes['slice_label'].values
    fpsPerTrial = expmtNotes['frame_rate'].values
    trialPaths = glob.glob(expmtPath+'/TSeries*')
    traceDict = {}
    for trial in dataDict.keys():
        fps = fpsPerTrial[trial-1]
        ventilatorSamplingRate = 10000
        downSampleVent = round(ventilatorSamplingRate/fps) #so for a 29.94 Hz recording this will sample vent trace every 334th frame
        print(f'Extracting Traces for trial {trial}')
        trialPath = trialPaths[trial]
        nCycles = len(glob.glob(trialPath+'/rT*_C*_ch2.tif'))
        roiRedness = [] #avg red value of each roi
        numROIs = int(len(dataDict[trial][f'T{trial}_roiFeatures'])/3)
        rednessOfROIs = np.zeros((numROIs,1))
        roiFeatureDict = dataDict[trial][f'T{trial}_roiFeatures']
        for roi in range(numROIs):
            rednessOfROIs[roi] = roiFeatureDict[f'roi{roi+1}_redAvg']
        print(f'Red Acquired for {len(rednessOfROIs)} ROIs')
        roiRedness.append(rednessOfROIs)
        if nCycles>1:
            trialTrace = []
            voltageSignals = glob.glob(trialPath+'/TSeries*VoltageRecording*.csv')
            for cycleIDX in range(nCycles):
                ventilatorTrace = (((pd.read_csv(voltageSignals[cycleIDX]).iloc[:,3]>3.).astype(float)-2)/4)[::downSampleVent] #downsampling bc fps = 30, but voltage sampled 10000 hz
                rawTrace = np.array(dataDict[trial][f'cycle{cycleIDX}_traces'])
                frameLoss = 25
                rawTracesPadded = np.pad(rawTrace, ((0,0),(0,frameLoss)), mode='constant', constant_values=np.nan)
                voltageTracePadded = np.pad(np.array(ventilatorTrace), (0,frameLoss), mode='constant', constant_values=np.nan)[1:]
                cycleTrace =rawTracesPadded            
                try:
                    addToTraces = np.vstack([cycleTrace, voltageTracePadded])
                    trialTrace.append(addToTraces)
                except:
                    #if cycle number is different from other trials break loop
                    print('Trial with unexpected number of cycles detected, breaking analysis')
                    break
            trialTrace = np.hstack(trialTrace).T
            traceDict[trial] = trialTrace
        else:
            trialTrace = np.hstack([
                np.array(dataDict[trial]['cycle0_traces']),
                (((pd.read_csv(voltageSignals[0]).iloc[:,3]>3.).astype(float)-2)/4)[::downSampleVent]
                ])
            traceDict[trial] =  trialTrace
            traceDict[f'T{trial}_roiRedness'] = roiRedness
    return traceDict

def compare_all_ROIs(conditionIDX, trial, conditions, traces):
    trial = trial-1 #because it goes in as +1 from init script
    actualTrial = trial+conditionIDX
    trialTraces = traces[trial+conditionIDX]
    voltageTrace = trialTraces[:,-1]
    conditionString = conditions[actualTrial]#expmtNotes['stim_type'].values[actualTrial]
    stimFrame = find_stim_frame(voltageTrace, conditionString)
    if stimFrame is None:
        return
    else:
        if stimFrame >=150:
            beggining = 150
        else:
            beggining = len(voltageTrace[:stimFrame])
        
        if stimFrame + 300 > len(voltageTrace):
            end = len(voltageTrace) - stimFrame
        else:
            end = 300
        fig, ax  = plt.subplots()
        f0 = np.nanmean(trialTraces[stimFrame-beggining:stimFrame,:]) #mean is the average frames before the mechanical challenge
        plottingTrace = trialTraces[stimFrame-beggining: stimFrame+end,:]
        dFF = (plottingTrace - f0)/f0

        normalizedDFF = (dFF - np.nanmean(dFF, axis=0)) / np.nanstd(dFF, axis=0)
        ax.imshow(normalizedDFF.T, aspect = 'auto', interpolation='none', cmap='Greens')
        ax.get_xaxis().set_visible(False)
        ax.set_title(f'{conditionString}')
        ax.axvline(beggining, color='black')
        fig.tight_layout()
        fig.supylabel('ROI Number')
        ax.set_xlabel('Time (30 fps)')
        fig.tight_layout()
    return fig

def summerize_experiment(expmtPath, trial, traces):
    trialPath = glob.glob(expmt+f'/TSeries*/rT{trial}*Ch2*.tif')[0]
    loadedTif = tif.imread(trialPath)
    mIM = np.nanmean(loadedTif, axis=0)
    expmtNotes = pd.read_excel(glob.glob(expmtPath+'/expmtNotes*.xlsx')[0])
    labeling = expmtNotes['lung_label'][0]
    if labeling == 'WGATR':
        roiRedValue = traces[f'T{trial}_roiRedness'][0]
        print(f'Labeling is {labeling}')
    else:
        print(f'Labeling is {labeling} - no red values to plot')
    slicePerTrial = expmtNotes['slice_label'].values
    segmentationUsed = slicePerTrial[trial-1]
    masksNPY = glob.glob(expmtPath+f'/segmentations/WGA_manual/*rT*C*Ch*slice{segmentationUsed}_seg.npy')
    masksLoaded = np.load(masksNPY[0], allow_pickle=True).item()

    roiMask = masksLoaded['masks']
    roiOutlines = masksLoaded['outlines']
    rois = np.unique(roiMask)[1:]
    if os.path.exists(expmt+'/figures/'):
        pdfSummary = PdfPages(expmtPath+f'/figures/slice{slicePerTrial[trial]}_dev_summary.pdf')
    else:
        os.mkdir(expmt+'/figures/')
        pdfSummary = PdfPages(expmtPath+f'/figures/slice{slicePerTrial[trial]}_dev_summary.pdf')
    roiCenters = []
    
    for roi in rois:
        roiCenters.append(center_of_mass(roiMask == roi))
    
    if labeling == 'WGATR':
        redTrialPath = glob.glob(expmt+f'/TSeries*/rT{trial}*Ch1*.tif')[0]
        redLoadedTif = tif.imread(redTrialPath)
        rmIM = np.nanmean(redLoadedTif, axis=0)
        
        gCAMPIM = np.zeros((roiMask.shape[0], roiMask.shape[1],3))
        wgaIM = np.zeros((roiMask.shape[0], roiMask.shape[1],3))
        gCAMPIM[:,:,1]  = roiMask
        gCAMPIM[:,:,0] = np.power(   mIM/np.max(mIM-20) , .72)
        gCAMPIM[:,:,2] = np.power(   mIM/np.max(mIM-20) , .72)
        
        wgaIM[:,:,1]  = roiOutlines
        wgaIM[:,:,0] = np.power(   rmIM/np.max(rmIM-20) , .52)
        wgaIM[:,:,2] = np.power(   rmIM/np.max(rmIM-20) , .52)

        fig, ax = plt.subplots( 1,2)
        gCAMPIM = np.clip(gCAMPIM, 0,1)
        ax[0].imshow(gCAMPIM)
        ax[0].axis('off')
        ax[0].set_title('GCaMP IM')
        wgaIM = np.clip(wgaIM, 0,1)
        ax[1].imshow(wgaIM)
        ax[1].set_title('WGA IM')
        ax[1].axis('off')
        for roi in rois:
            ax[0].text(roiCenters[roi-1][1]-2,roiCenters[roi-1][0], f'{roi}', color='black', size=2)
        fig.suptitle(f'ROI {roi} Overlay')
        plt.savefig(pdfSummary, format='pdf')
       
        expmtNotes = pd.read_excel(glob.glob(expmtPath+'/expmtNotes*.xlsx')[0])
        expmtConditions = np.unique(expmtNotes['stim_type'].values)
        nTrials = len(glob.glob(expmtPath+'/TSeries*'))
        nConditions = len(expmtConditions)

        for conditionIDX in range(nConditions):
            allROIsFig = compare_all_ROIs(conditionIDX, trial, expmtNotes['stim_type'].values, traces)
            plt.savefig(pdfSummary, format='pdf')
    
        for roi in rois:
            dFFFigure = plot_individual_cell(expmtPath, trial, roi, traces)
            plt.savefig(pdfSummary, format='pdf')
        
        print('Saving...')
        pdfSummary.close()
    else:
        print('WGA-594 res galvo pipeline still a work in progress')

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
    else:
        inspLengths = risingIDX - np.roll(risingIDX, 1)
        longestInsp = np.unique(inspLengths)[-1]
        stimFrame = risingIDX[np.where(inspLengths == longestInsp)[0]-1]
    return stimFrame[0]

def plot_individual_cell(expmtPath, trial, roi, traces):
    expmtNotes = pd.read_excel(glob.glob(expmtPath+'/expmtNotes*.xlsx')[0])
    expmtConditions = np.unique(expmtNotes['stim_type'].values)
    nConditions = len(expmtConditions)
    nTrials = len(glob.glob(expmt+'/TSeries*'))
    for trialSet in range(0,nTrials,nConditions):
        if trial>trialSet and trial in range(trial, trialSet+nConditions):
            fig,ax = plt.subplots(nConditions,1, sharex=True)
            for condition in range(nConditions):
                actualTrial = trialSet+condition
                trialTraces = traces[trialSet+condition]
                voltageTrace = trialTraces[:,-1]
                conditionString = expmtNotes['stim_type'].values[actualTrial]
                stimFrame = find_stim_frame(voltageTrace, conditionString)
                if stimFrame is None:
                    break
                else:
                    if stimFrame >=150:
                        beggining = 150
                    else:
                        beggining = len(voltageTrace[:stimFrame])
                    
                    if stimFrame + 300 > len(voltageTrace):
                        end = len(voltageTrace) - stimFrame
                    else:
                        end = 300
                    
                    roiTrace = trialTraces[:,roi-1]
                    f0 = np.nanmean(roiTrace[stimFrame-beggining:stimFrame]) #mean is the average frames before the mechanical challenge
                    plottingTrace = roiTrace[stimFrame-beggining: stimFrame+end]
                    dFF = (plottingTrace - f0)/f0
                    if condition==0:
                        ax[condition].plot(np.arange(-beggining, end), dFF, label='Raw Trace', color='green')
                        ax[condition].set_ylabel(f'{conditionString}')
                    else:
                        ax[condition].plot(np.arange(-beggining, end), dFF, color='green')
                        ax[condition].set_ylabel(f'{conditionString}')
                    ax[condition].axvline(0, color='red')
                    
                    ax[condition].set_ylim([-0.5, 0.75])
                    fig.suptitle(f'ROI Traces of ROI={roi} at depth = {expmtNotes['z_pos'].values[actualTrial]}')
                    fig.supxlabel('Time (30 fps)')

            fig.legend()
    return fig

#%%

if __name__=='__main__':
    dataFrom = [
        'U:/expmtRecords/Lucas*',
        'C:/Analysis/april_data/Lucas*',
        'U:/expmtRecords/res_galvo/Lucas*',
        ]
    expmtRecords = glob.glob(dataFrom[2])
    plt.close('all')
    for expmt in expmtRecords:
        if os.path.exists(expmt+'/expmtTraces.pkl'):
            if os.path.exists(expmt+'/dev/'):
                print('Figures already generated for this experiment')
            else:
                print('Loading Traces for \n', expmt)
                with open(expmt+'/expmtTraces.pkl', 'rb') as f:
                    dataDict = pickle.load(f)
                expmtNotes = pd.read_excel(glob.glob(expmt+'/expmtNotes*.xlsx')[0])
                expmtConditions = expmtNotes['stim_type']
                nTrials = len(expmtConditions)
                nConditions = len(np.unique(expmtConditions))
                traces = compress_traces(expmt, dataDict)
                print('Traces extracted, now generating plots...')
                for trialSlice in range(0, nTrials, nConditions):
                    print(f'Generating Plots for trials {trialSlice} through {trialSlice + nConditions}')
                    summerize_experiment(expmt, trialSlice+1, traces)
                print('Saved Plots...')
                break #only work with 5-7 for now...
        else:
            print('Experiment not processed yet')
        print('******************* NEXT *******************')

# %%
