#%%
import tifffile as tif
import numpy as np
import pandas as pd
import os, glob, pickle, cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

expmtFiles = glob.glob('U:/expmtRecords/gas_expmts/*')

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
    return metaData
                

    