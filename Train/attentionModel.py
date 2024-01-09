import random
from keras import initializers, regularizers, constraints
from pandas import read_csv
import numpy as np
from keras import Model
from keras.layers import Layer, Lambda, MaxPool2D
import keras.backend as K
from keras.layers import Input, Dense, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.metrics import mean_squared_error
import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from pandas import DataFrame
from pandas import concat
from math import sqrt
from numpy import concatenate
import tensorflow as tf
import seaborn as sns
from keras.losses import mean_squared_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os
import math
import datetime as dt
from keras.layers import Dense, Activation, BatchNormalization, LSTM, Bidirectional, TimeDistributed, Conv1D, \
    MaxPooling1D, Flatten, ConvLSTM2D, Conv2D, MaxPooling2D, Dropout, AveragePooling3D, RepeatVector, GRU, SimpleRNN
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras import optimizers, Model, Input
from keras import backend as K
import xlsxwriter as xlsxwriter
import time

from dataProcess import dataProcess
from attention import Attention

def attentionModel(layerNumber,algorithm, convFlag, biFlag,train_X0):
    model_input = Input(shape=(train_X0.shape[1], train_X0.shape[2]))
    if layerNumber == 2:
        if convFlag == 0 and biFlag == 0:
            x = algorithm(512, return_sequences=True)(model_input)
            x = algorithm(512, return_sequences=True)(x)
        elif convFlag == 1 and biFlag == 0:
            x= Conv1D(filters=32, kernel_size=1, activation='relu')(model_input)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
        elif convFlag == 0 and biFlag == 1:
            x = Bidirectional((algorithm(512, return_sequences=True)))(model_input)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
    elif layerNumber == 4:
        if convFlag == 0 and biFlag == 0:
            x = algorithm(512, return_sequences=True)(model_input)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
        elif convFlag == 1 and biFlag == 0:
            x= Conv1D(filters=32, kernel_size=1, activation='relu')(model_input)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
        elif convFlag == 0 and biFlag == 1:
            x = Bidirectional((algorithm(512, return_sequences=True)))(model_input)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
    elif layerNumber == 6:
        if convFlag == 0 and biFlag == 0:
            x = algorithm(512, return_sequences=True)(model_input)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
        elif convFlag == 1 and biFlag == 0:
            x= Conv1D(filters=32, kernel_size=1, activation='relu')(model_input)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
        elif convFlag == 0 and biFlag == 1:
            x = Bidirectional((algorithm(512, return_sequences=True)))(model_input)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
    elif layerNumber == 8:
        if convFlag == 0 and biFlag == 0:
            x = algorithm(512, return_sequences=True)(model_input)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
        elif convFlag == 1 and biFlag == 0:
            x = Conv1D(filters=32, kernel_size=1, activation='relu')(model_input)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
        elif convFlag == 0 and biFlag == 1:
            x = Bidirectional((algorithm(512, return_sequences=True)))(model_input)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
    elif layerNumber == 10:
        if convFlag == 0 and biFlag == 0:
            x = algorithm(512, return_sequences=True)(model_input)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
        elif convFlag == 1 and biFlag == 0:
            x = Conv1D(filters=32, kernel_size=1, activation='relu')(model_input)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
        elif convFlag == 0 and biFlag == 1:
            x = Bidirectional((algorithm(512, return_sequences=True)))(model_input)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
    elif layerNumber == 12:
        if convFlag == 0 and biFlag == 0:
            x = algorithm(512, return_sequences=True)(model_input)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
        elif convFlag == 1 and biFlag == 0:
            x = Conv1D(filters=32, kernel_size=1, activation='relu')(model_input)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
            x = algorithm(512, return_sequences=True)(x)
        elif convFlag == 0 and biFlag == 1:
            x = Bidirectional((algorithm(512, return_sequences=True)))(model_input)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)
            x = Bidirectional(algorithm(512, return_sequences=True))(x)

    x = Attention(units=512)(x)
    x = Dense(6)(x)
    x = Activation("linear")(x)
    model = Model(model_input, x)

    return model