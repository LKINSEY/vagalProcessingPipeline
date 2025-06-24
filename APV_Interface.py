from PyQt6.QtWidgets import (
    QListWidget, 
    QPushButton, 
    QHBoxLayout, 
    QLabel, 
    QErrorMessage, 
    QApplication, 
    QMainWindow, 
    QLineEdit, 
    QVBoxLayout, 
    QWidget, 
    QGroupBox,
    QFileDialog,
)
from PyQt6.QtGui import QDoubleValidator, QAction
from PyQt6.QtCore import QTimer
import os, json, sys, asyncio
from datetime import datetime
from expmt_util import *


'''
This GUI is meant to serve as an additional window to compliment the Prairie View Screen.
Operation involves simply opening this window (with accompanying Batch File) when prairie view is already open
'''

class APV_Interface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Alicat-Prairie View Interface')
        self.setGeometry(200,200,750,200)
        self.show()

        conditionsJSONPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conditions.json')
        with open(conditionsJSONPath, 'r') as f:
                self.conditionsDict = json.load(f)
        self.co2 = None
        self.o2 = None
        self.n2 = None
        self.pl = None
        self.alicatsOn = False
        self.pvConnection = False
        self.currentcondition = self.conditionsDict['basal']
        self.trialCounter = 0
        self.savePath = 'C:/'
        self.gasChangeSetTime = int(120*1000)
        self.eventLog = {}
        
        self.initUI()
       
    def initUI(self):
        configMenu = self.menuBar()

        #for later...
        # configMenu.addMenu('Add Gas Condition')
        # configMenu.addMenu('Remove Gas Condition')
        # configMenu.addMenu('Exit')

        saveDirLocation = QAction('Choose Save Directory', self) #defaults to C:/
        configMenu.addAction(saveDirLocation)
        saveDirLocation.triggered.connect(self.update_save_dir)
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)

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

        self.runTrial_button = QPushButton('Run T-Series')
        self.runTrial_button.clicked.connect(self.send_trial_run_noParam)

        leftColumn.addWidget(self.connectAlicats_button)
        leftColumn.addWidget(self.alicatConnect_text)
        leftColumn.addWidget(self.connectPV_button)
        leftColumn.addWidget(self.PVConnection_text)
        leftColumn.addWidget(self.runTrial_button)

        self.conditionsList = QListWidget()
        for condition in self.conditionsDict.keys():
            self.conditionsList.addItem(condition)
        self.conditionsList.setCurrentRow(0)
        

        self.currentSettingsLabel = QGroupBox('Trial Settings')
        self.currentSettingsLabel.setFixedHeight(120)

        self.current_TitlesLayout = QHBoxLayout()
        self.current_o2Title = QLabel('O2 %')
        self.current_TitlesLayout.addWidget(self.current_o2Title)
        self.current_co2Title = QLabel('CO2 %')
        self.current_TitlesLayout.addWidget(self.current_co2Title)
        self.current_n2Title = QLabel('N2 % ')
        self.current_TitlesLayout.addWidget(self.current_n2Title)
        self.current_maxVolTitle = QLabel('Max Volume (SLPM)')
        self.current_TitlesLayout.addWidget(self.current_maxVolTitle)

        self.current_textEditsLayout = QHBoxLayout()
        self.current_o2Edit = QLineEdit(str(self.currentcondition['O2']))
        self.current_o2Edit.setValidator(validator)
        self.current_o2Edit.textChanged.connect(lambda text: self.__if_text_changed(text, self.current_o2Edit))
        self.current_textEditsLayout.addWidget(self.current_o2Edit)
        self.current_co2Edit = QLineEdit(str(self.currentcondition['CO2']))
        self.current_co2Edit.setValidator(validator)
        self.current_co2Edit.textChanged.connect(lambda text: self.__if_text_changed(text, self.current_co2Edit))
        self.current_textEditsLayout.addWidget(self.current_co2Edit)
        self.current_n2Edit = QLineEdit(str(self.currentcondition['N2']))
        self.current_n2Edit.setValidator(validator)
        self.current_n2Edit.textChanged.connect(lambda text: self.__if_text_changed(text, self.current_n2Edit))
        self.current_textEditsLayout.addWidget(self.current_n2Edit)
        self.current_maxVolEdit = QLineEdit(str(0.1))
        self.current_maxVolEdit.setValidator(validator)
        self.current_textEditsLayout.addWidget(self.current_maxVolEdit)

        self.zeroAlicats = QPushButton('Zero Alicats')
        self.zeroAlicats.clicked.connect(self.zero_alicats)
        self.instantAlicatChange = QPushButton('Update Gas Settings')
        self.instantAlicatChange.clicked.connect(self.instant_gas_change)

        self.currentSettingsLayout = QVBoxLayout()
        self.currentSettingsLayout.addLayout(self.current_TitlesLayout)
        self.currentSettingsLayout.addLayout(self.current_textEditsLayout)
        self.buttonsLayout = QHBoxLayout()
        self.buttonsLayout.addWidget(self.zeroAlicats)
        self.buttonsLayout.addWidget(self.instantAlicatChange)
        self.currentSettingsLayout.addLayout(self.buttonsLayout)
        self.currentSettingsLabel.setLayout(self.currentSettingsLayout)

        #TODO add a new condition add a file manager in top left that opens a condition adder - this shouldn't be a regular thing so its hidden, but need a dedicated way to customize conditions

        rightColumn.addWidget(self.conditionsList)
        rightColumn.addWidget(self.currentSettingsLabel)

        
        self.conditionsList.itemSelectionChanged.connect(
            lambda: self.update_current_settings(
                self.current_o2Edit, 
                self.current_co2Edit, 
                self.current_n2Edit))

    def update_save_dir(self):
        newSaveDIR = QFileDialog.getExistingDirectory(self, 'Select Save Save Dir Specified in Prairie View')
        if newSaveDIR:
            self.Path = newSaveDIR

    def __if_text_changed(self, text, lineEdit):
        print('__if_text_changed')
        lineEdit.setStyleSheet("background-color: lightblue;")

    def return_to_basal(self):
        print('return to basal')
        self.currentcondition = self.conditionsDict['basal']
        getMaxFlow = self.current_maxVolEdit.text()
        co2Print, o2Print, n2Print = asyncio.run(get_alicat_info(self.co2, self.o2, self.n2))
        asyncio.run(set_gas_flow_composition(self.co2, self.o2, self.n2, self.currentcondition, getMaxFlow))
        gasChangeDict = {
            'time': datetime.now().time().isoformat(),
            'gas': {
                'co2':co2Print,
                'o2':o2Print,
                'n2':n2Print
            },
            'conditions': 'basal'
        }
        self.eventLog['return_to_basal'] = gasChangeDict
        self.current_o2Edit.setText(str(self.currentcondition['O2']))
        self.current_co2Edit.setText(str(self.currentcondition['CO2']))
        self.current_n2Edit.setText(str(self.currentcondition['N2']))



    def instant_gas_change(self):
        print('instant_gas_change')
        changeDetails = {}
        getMaxFlow = self.current_maxVolEdit.text()
        co2Print, o2Print, n2Print = asyncio.run(get_alicat_info(self.co2, self.o2, self.n2))
        changeDetails['pre'] = {
            'time': datetime.now().time().isoformat(),
            'gas': {
                'co2':co2Print,
                'o2':o2Print,
                'n2':n2Print
            },
            'conditions': self.conditionsList.selectedItems()[0].text()
        }
        
        asyncio.run(set_gas_flow_composition(self.co2, self.o2, self.n2, self.currentcondition, getMaxFlow))
        co2Print, o2Print, n2Print = asyncio.run(get_alicat_info(self.co2, self.o2, self.n2))

        changeDetails['post'] = {
            'time': datetime.now().time().isoformat(),
            'gas': {
                'co2':co2Print,
                'o2':o2Print,
                'n2':n2Print
            },
            'conditions': self.conditionsList.selectedItems()[0].text()
        }
        self.eventLog['inst_change'] = changeDetails
        self.current_n2Edit.setStyleSheet("QTextEdit { background-color: none; }")
        self.current_co2Edit.setStyleSheet("QTextEdit { background-color: none; }")
        self.current_o2Edit.setStyleSheet("QTextEdit { background-color: none; }")

    def connect_to_alicats(self):
        print('connect_to_alicats')
        if not self.alicatsOn:
            try:
                self.co2, self.o2, self.n2 = asyncio.run(connect_alicats('COM5'))
                self.alicatsOn = True
                self.connectAlicats_button.setText('Disconnect Alicats')
                self.alicatConnect_text.setText('Alicats Connected')
            except:
                self.notConnected = QErrorMessage(self)
                self.notConnected.showMessage('Not Connected to Prairie View')
        else:
            asyncio.run(close_alicats(self.co2, self.o2, self.n2))
            self.alicatsOn = False
            self.connectAlicats_button.setText('Connect to Alicats')
            self.alicatConnect_text.setText('Not Connected')

    def connect_to_pv(self):
        print('connect_to_pv')
        if not self.pvConnection:
            address = who_am_i()    
            self.pl = connect_to_prairie_view(address)
            print(self.pl)
            if self.pl:
                self.connectPV_button.setText('Disconnect from PV')
                self.pvConnection = True
                self.PVConnection_text.setText('Connected to Prairie Link')
            else:
                self.PVConnection_text.setText('Did not connect to prairie view - try opening prairie view')

        else:   
            try:
                self.pl.Disconnect()
            finally:
                self.pvConnection = False
                self.PVConnection_text.setText('Not Connected')

    def send_trial_run_noParam(self):
        print('send_trial_run_noParam')
        co2Print, o2Print, n2Print = asyncio.run(get_alicat_info(self.co2, self.o2, self.n2))
        if self.pl:
            self.eventLog = {
                'ts': datetime.now().time().isoformat(),
                'initial_gas':{
                    'co2':co2Print,
                    'o2':o2Print,
                    'n2':n2Print
                }
            }
            run_single_trial(self.pl)

            #GAS CHANGE STARTS 120s AFTER TRIAL STARTS, but when it reverts thats customizable
            QTimer.singleShot(self.gasChangeSetTime, self.set_gases)
            
            QTimer.singleShot(self.gasChangeSetTime, self.return_to_basal)

            with open(os.path.join(self.savePath, f'gas_trial_{self.trialCounter}.json'), 'w') as f:
                json.dump(self.eventLog, f, indent=4)
            self.eventLog = {}
            self.current_n2Edit.setStyleSheet("QTextEdit { background-color: none; }")
            self.current_co2Edit.setStyleSheet("QTextEdit { background-color: none; }")
            self.current_o2Edit.setStyleSheet("QTextEdit { background-color: none; }")
            

        else:
             self.notConnected = QErrorMessage(self)
             self.notConnected.showMessage('Not Connected to Prairie View')
    
    def zero_alicats(self):
        print('zero_alicats')
        asyncio.run(set_gas_flow_composition(self.co2, self.o2, self.n2, self.conditionsDict['zero']))        
        self.currentcondition = self.conditionsDict['zero']
        for i in range(self.conditionsList.count()):
            if self.conditionsList.item(i).text() == 'zero':
                self.conditionsList.setCurrentRow(i)
        self.current_o2Edit.setText(str(self.currentcondition['O2']))
        self.current_o2Edit.setStyleSheet("background-color: none;")
        self.current_co2Edit.setText(str(self.currentcondition['CO2']))
        self.current_co2Edit.setStyleSheet("background-color: none;")
        self.current_n2Edit.setText(str(self.currentcondition['N2']))
        self.current_n2Edit.setStyleSheet("background-color: none;")  
        self.current_maxVolEdit.setText(str(0.1))
        self.current_maxVolEdit.setStyleSheet("QTextEdit { background-color: none; }")             


    def update_current_settings(self, current_o2Edit, current_co2Edit, current_n2Edit):
        print('update_current_settings')
        newCondition = self.conditionsList.selectedItems()[0].text()
        self.currentcondition = self.conditionsDict[newCondition]
        current_o2Edit.setText(str(self.currentcondition['O2']))
        current_co2Edit.setText(str(self.currentcondition['CO2']))
        current_n2Edit.setText(str(self.currentcondition['N2']))
        
    
    def set_gases(self):
        print('set_gases')
        gasTime = datetime.now().time().isoformat()
        getMaxFlow = self.current_maxVolEdit.text()
        asyncio.run(set_gas_flow_composition(self.co2, self.o2, self.n2, self.currentcondition,getMaxFlow))
        self.current_o2Edit.setStyleSheet("QTextEdit { background-color: none; }")
        self.current_co2Edit.setStyleSheet("QTextEdit { background-color: none; }")
        self.current_n2Edit.setStyleSheet("QTextEdit { background-color: none; }")
        self.current_maxVolEdit.setStyleSheet("QTextEdit { background-color: none; }")
        co2Print, o2Print, n2Print = asyncio.run(get_alicat_info(self.co2, self.o2, self.n2))
        gasChangeDict = {
            'time': gasTime,
            'gas': {
                'co2':co2Print,
                'o2':o2Print,
                'n2':n2Print
            },
            'conditions': self.conditionsList.selectedItems()[0].text()
        }
        self.eventLog['gas_change'] = gasChangeDict
        print(type(self.eventLog))
        
        
        self.trialCounter+=1

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainApp = APV_Interface()
    sys.exit(app.exec())


'''
the big TODO list:


3.) make the saving of these logs to be saved to where pv is saving trials to #kinda
4.) make feature to change gas composition without running trial (involves also differentiating between pending gas change and current gas)
5.) include text edit for max volume delivery
6.) text edit WHEN to deliver gas change
'''