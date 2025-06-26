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
            if not dataDict[1]:
                print('Processing Error - not writing pickle')
            else:
                with open(expmt+'/expmtTraces.pkl', 'wb') as f:
                    pickle.dump(dataDict, f)

# %%
