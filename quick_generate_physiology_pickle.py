#open appropriate file beforehand
#%%
import labchart
import win32com.client
import matplotlib.pyplot as plt
import pickle
import matplotlib
from processing_util import include_comments

matplotlib.use('WebAgg') 
#%%
expmtPathToSave = 'U:/expmtRecords/september2025/Lucas_250912_001/'
adichtFN = '092925-recording.adicht'
app = labchart.Application()
doc = app.active_document
blockNum = 2
dataDict = {}


plt.close('all')
for chanName in doc.channel_names:

    keyName = chanName.replace(' ', '_')
    result = doc.get_channel_data(chanName, blockNum)

    dataDict[keyName+'_raw'] = result['y']
    dataDict[keyName+'_start'] = result['start_time']
    dataDict[keyName+'_end'] = result['end_time']
    dataDict[keyName+'_fs'] = result['fs']
    # dataDict[keyName+'_nTicks'] = int(result['fs']*result['end_time']) #just use the shape of raw data to get this
    #troubleshooting/sanity checking - USE THIS TO MAKE SURE YOU ARE SAVING THE WRITE DATA!!!!
#     plt.figure()
#     plt.plot(result['y'])
#     plt.title(chanName)
# plt.show()

dataDict['comments'] = include_comments(
    expmtPathToSave+adichtFN,
    dataDict,
    1
)

#be sure to specify expmt or baseline
with open(expmtPathToSave+'baselinePhysiology.pkl', 'wb') as f:
    pickle.dump(dataDict, f)
