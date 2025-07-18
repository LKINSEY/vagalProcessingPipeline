#%%
import glob, pickle
from processing_util import *

if __name__=='__main__':
    dataFrom = [
        'U:/expmtRecords/Lucas*',
        'C:/Analysis/april_data/Lucas*',
        'U:/expmtRecords/res_galvo/Lucas*',
        'U:/expmtRecords/mech_galvo/Lucas*',
        ]
    expmtRecords = glob.glob(dataFrom[3])
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
    
        metaData = extract_metaData(expmt)
        metaData['regParams'] = regParams
        
        register_2ch_trials(expmt, regParams)
        #insert cellpose command here
        dataDict = extract_roi_traces(expmt)
        if dataDict:
            if not dataDict[1]:
                print('Processing Error - not writing pickle')
            else:
                with open(expmt+'/expmtTraces.pkl', 'wb') as f:
                    pickle.dump(dataDict, f)

# %%
