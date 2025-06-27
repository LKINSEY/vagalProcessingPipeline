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
from pathlib import Path
from processing_util import *


if __name__=='__main__':
    dataFrom = [
        'U:/expmtRecords/Lucas*',
        'C:/Analysis/april_data/Lucas*',
        'U:/expmtRecords/res_galvo/Lucas*',
        'U:/expmtRecords/mech_galvo/Lucas*',
        'U:/expmtRecords/complex_expmts/*',
        'U:/expmtRecords/gas_expmts/*001*',
        ]
    expmtRecords = glob.glob(dataFrom[5])
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
