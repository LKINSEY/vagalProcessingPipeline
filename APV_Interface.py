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
import os, json, sys
from expmt_util import *


'''
This GUI is meant to serve as an additional window to compliment the Prairie View Screen.
Operation involves simply opening this window (with accompanying Batch File) when prairie view is already open
'''

class APV_Interface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Alicat-Prairie View Interface')
        self.setGeometry(200,200,400,200)

        self.show()
        with open(os.getcwd() + '\\conditions.json', 'r') as f:
                self.conditionsDict = json.load(f)
        self.co2 = None
        self.o2 = None
        self.n2 = None
        self.pl = None
        self.alicatsOn = False
        self.pvConnection = False
        self.currentcondition = self.conditionsDict['basal']

        
        self.initUI()
       
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

        self.currentSettingsLabel = QGroupBox('Current Settings')
        self.currentSettingsLabel.setFixedHeight(120)

        self.current_TitlesLayout = QHBoxLayout()
        self.current_o2Title = QLabel('Current O2')
        self.current_TitlesLayout.addWidget(self.current_o2Title)
        self.current_co2Title = QLabel('Current CO2')
        self.current_TitlesLayout.addWidget(self.current_co2Title)
        self.current_n2Title = QLabel('Current N2')
        self.current_TitlesLayout.addWidget(self.current_n2Title)

        self.current_textEditsLayout = QHBoxLayout()
        self.current_o2Edit = QTextEdit(str(self.currentcondition['O2']))
        self.current_textEditsLayout.addWidget(self.current_o2Edit)
        self.current_co2Edit = QTextEdit(str(self.currentcondition['CO2']))
        self.current_textEditsLayout.addWidget(self.current_co2Edit)
        self.current_n2Edit = QTextEdit(str(self.currentcondition['N2']))
        self.current_textEditsLayout.addWidget(self.current_n2Edit)

        self.updateChangedSettings = QPushButton('Update Settings')

        self.currentSettingsLayout = QVBoxLayout()
        self.currentSettingsLayout.addLayout(self.current_TitlesLayout)
        self.currentSettingsLayout.addLayout(self.current_textEditsLayout)
        self.currentSettingsLayout.addWidget(self.updateChangedSettings)
        self.currentSettingsLabel.setLayout(self.currentSettingsLayout)

        #TODO add a new condition add a file manager in top left that opens a condition adder - this shouldn't be a regular thing so its hidden, but need a dedicated way to customize conditions

        rightColumn.addWidget(self.conditionsList)
        rightColumn.addWidget(self.currentSettingsLabel)




    def connect_to_alicats(self):
        if not self.alicatsOn:
            self.co2, self.o2, self.n2 = connect_alicats('COM5')
            self.alicatsOn = True
            self.connectAlicats_button.setText('Disconnect Alicats')
            self.alicatConnect_text.setText('Alicats Connected')
        else:
            close_alicats(self.co2, self.o2, self.n2)
            self.alicatsOn = False
            self.alicatConnect_text.setText('Not Connected')

    def connect_to_pv(self):
        if not self.pvConnection:
            address = who_am_i()    
            self.pl = connect_to_prairie_view(address)
            if self.pl:
                self.connectPV_button.setText('Disconnect from PV')
                self.pvConnection = True
                self.alicatConnect_text.setText('Connected to Prairie Link')
            else:
                self.alicatConnect_text.setText('Unable to Establish Prairie-Link')

        else:   
            disconnect_from_prairie_view(self.pl)
            self.pvConnection = False
            self.alicatConnect_text.setText('Not Connected')



    def send_trial_run_noParam(self):
        if self.pl:
            run_single_trial(self.pl)
        else:
             self.notConnected = QErrorMessage(self)
             self.notConnected.showMessage('Not Connected to Prairie View')
             #update qlabel to say "you need to connect to pv or alicats" or something
             #for now jsut print because its end of day today
             print('Not Connected to prairie view')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainApp = APV_Interface()
    sys.exit(app.exec())


'''
the big TODO list:
1.) make feature to save what condition is used every time you click "run trial"
2.) make sure you log the timing the conditions change
3.) make the saving of these logs to be saved to where pv is saving trials to
4.) do more...
'''