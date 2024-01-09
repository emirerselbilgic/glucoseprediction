#Emir Ersel Bilgiç
#28.12.2023
#6 input - 30 min prediction, Online learning experiment
import threading
import os
import time
import keras
from datetime import datetime
import numpy as np
import firebase_admin
from firebase_admin import db, credentials, firestore
import xlsxwriter as xlsxwriter
import openpyxl
from openpyxl import Workbook
from typing import Any
from train_model import train_model




########################################################################################################################
# initialization

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
seedNumber      =   34                  # seed number will be changed till 'seedRange'
epochRunning    =   1               # epoch number for running
featureNumber   =   1                   # 1 --> CGM only, 2 --> CGM + Basal Insulin, 3 --> CGM + Basal Insulin + CHO
layerNumber     =   1                    # number of model layers
modelType       =   0                  # 0 --> RNN, 1 --> LSTM, 2 --> GRU, 3 --> BiRNN, 4 --> BiLSTM, 5 --> BiGRU,
                                         # 6 --> ConvRNN, 7 --> ConvLSTM, 8 --> ConvGRU
patientFlag     =   6                    # 0 --> 540, 1 --> 544, 2 --> 552, 3 --> 559, 4 --> 563, 5 --> 567,
                                         # 6 --> 570, 7 --> 575, 8 --> 584, 9 --> 588, 10 --> 591, 11 --> 596
newPatientFlag  =   2   
testFlag        =   1                    # if test flag is 1, test code will run. If it is 0, it will not run.
plotFlag        =   1                    # if plot flag is 1, plots will appear. If it is 1, plots will not appear.
patientNumber   =   12                  # total number of patients
start           =   time.time()          # record start time
rows_to_transfer = 6                     # Number of rows to transfer in each iteration


modelList   = [ "RNN", "LSTM", "GRU", "BiRNN", "BiLSTM", "BiGRU", "ConvRNN",
              "ConvLSTM", "ConvGRU"   ]
wsList      = ['patient540', 'patient544', 'patient552', 'patient559', 'patient563', 'patient567', 'patient570',
               'patient575', 'patient584', 'patient588', 'patient591', 'patient596']

####################################################################################################################


while True:
    prediction = train_model(seedNumber, epochRunning, modelType, testFlag, plotFlag, patientFlag, newPatientFlag, featureNumber)
    print("Training starts...")


    

