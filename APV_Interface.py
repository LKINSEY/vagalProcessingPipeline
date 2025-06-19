from PyQt6.QtWidgets import (
    QListWidget, 
    QPushButton, 
    QComboBox,  
    QHBoxLayout, 
    QLabel, 
    QErrorMessage, 
    QApplication, 
    QMainWindow, 
    QTextEdit, 
    QVBoxLayout, 
    QWidget, 
    QGroupBox
)
from PyQt6.QtCore import (
    Qt, 
    QThreadPool
)
import os, json
from expmt_util import *

class APV_Interface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Alicat-Prairie View Interface')
        self.geometry()
        with open(os.getcwd() + '\\conditions.json', 'r') as f:
                conditionsDict = json.load(f)
        self.initUI()
        self.co2 = None
        self.o2 = None
        self.n2 = None
        self.pl = None
        self.alicatsOn = False
        self.pvConnection = False
       
    def initUI(self):
        background = QWidget()
        self.setCentralWidget(background)
        lefRightColumns = QHBoxLayout()
        background.setLayout(lefRightColumns)
        leftColumn = QVBoxLayout()
        rightColumn = QVBoxLayout()
        lefRightColumns.addLayout(leftColumn)
        lefRightColumns.addLayout(rightColumn)
        
        self.connectAlicats_button = QPushButton('Connect to Alicats')
        self.connectAlicats_button.clicked.connect(self.connect_to_alicats)
        self.alicatConnect_text = QLabel('Not Connected')

        self.connectPV_button = QPushButton('Connect To Prairie View')
        self.connectPV_button.clicked.connect(self.connect_to_pv)
        self.PVConnection_text = QLabel('Not Connected')

        self.runTrial_button = QPushButton('Run Single')
        self.runTrial_button.clicked.connect(self.send_trial_run_noParam)

        leftColumn.addWidget(self.connectAlicats_button)
        leftColumn.addWidget(self.alicatConnect_text)
        leftColumn.addWidget(self.connectPV_button)
        leftColumn.addWidget(self.PVConnection_text)
        leftColumn.addWidget(self.runTrial_button)

        self.conditionsList = QListWidget()
        for condition in self.conditionsDict.keys():
            self.conditionsList.addItem(condition)

        #TODO
        #alternative to 2 buttons, 1 button that just says "connect", then have 2 dots below it (Alicats and PV)
        #that turn green if connected, with text next to them saying their status with the QLabels)


    def connect_to_alicats(self):
        if not self.alicatsOn:
            self.co2, self.o2, self.n2 = connect_alicats('COM5')
            self.alicatsOn = True
            #change text to say "Disconnect" in ui
            #If successful connection, say: 'connected'
            #if unsuccessful connection, say: 'unable to connect after attempt {attempt number}, check com port
        else:
             close_alicats(self.co2, self.o2, self.n2)
             self.alicatsOn = False
             #change text to say "Connect" in ui
             #if successful, change qlabel to say 'Not Connected'
             #if unsuccessful, change qlabel to say 'Unable to disconnect alicats'

    def connect_to_pv(self):
        if not self.pvConnection:
            address = who_am_i()    
            self.pl = connect_to_prairie_view(address)
            if self.pl:
                print('success')
                self.pvConnection = True
        else:   
            disconnect_from_prairie_view(self.pl)
            self.pvConnection = False
        #TODO
        #still need to add the "if successful connect vs unsuccessful connection dialog"



    def send_trial_run_noParam(self):
        if self.pl:
            run_single_trial(self.pl)
        else:
             #update qlabel to say "you need to connect to pv or alicats" or something
             #for now jsut print because its end of day today
             print('Not Connected to prairie view')





'''
the big TODO list:
1.) make feature to save what condition is used every time you click "run trial"
2.) make sure you log the timing the conditions change
3.) make the saving of these logs to be saved to where pv is saving trials to
4.) do more...
'''