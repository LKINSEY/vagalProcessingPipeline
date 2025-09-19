#%%
import glob, pickle
from datetime import datetime
from processing_util import *

if __name__=='__main__':
    
    dataFrom = [
        'U:/expmtRecords/Lucas*',
        'U:/expmtRecords/september2025/*'
        ]
    
    expmtRecords = glob.glob(dataFrom[4])
    
    regParams = {
        'maxShifts': [25,25],
        'frames_per_split': 1000, 
        'niter_rig': 4,
        'overlaps': [5,5],
        'max_deviation_rigid': [25,25],
        'frame_corrector_batching': 100,
        'device': 'cpu',
        'num_blocks': [3,3]
    }

    for expmt in expmtRecords:
    


        
        metaData = extract_metadata(expmt)
        metaData['regParams'] = regParams
        
        register_2ch_trials(expmt, metaData, regParams)
        #insert cellpose command here
        dataDict = extract_roi_traces(expmt, metaData)
        
        
        if dataDict:
            if not dataDict[1]:
                print('Dictionary is Currupt - aborting')
            else:
                with open(expmt+'/expmtTraces.pkl', 'wb') as f:
                    pickle.dump(dataDict, f)
                with open(expmt+'/metaData.pkl', 'wb') as f:
                    pickle.dump(metaData, f)
            try:
                with open(expmt+'/expmtPhysiology.pkl', 'rb') as f:
                    physioDict = pickle.load(f)
                synchronizedTraces = sync_physiology(physioDict, dataDict, metaData)
                with open(expmt+f'/expmtSummary_{datetime.now().strftime("%Y-%m-%d_%H:%M")}.pkl', 'wb') as f:
                    pickle.dump(synchronizedTraces, f)
            except:
                print('No Physiological Pickle Detected ')




# %%
