import pandas as pd
import numpy as np
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import IsolationForest

def fix_anomalies(data, method='median'):
    fixed_data = data.copy()
    anomalies = fixed_data['Anomaly'] == 1

    if method == 'median':
        replacement_value = fixed_data[~anomalies]['CGM'].median()
    elif method == 'mean':
        replacement_value = fixed_data[~anomalies]['CGM'].mean()
    elif method == 'interpolate':
        fixed_data.loc[anomalies, 'CGM'] = np.nan
        fixed_data['CGM'] = fixed_data['CGM'].interpolate()
    elif method == 'remove':
        fixed_data = fixed_data[~anomalies]
    else:
        raise ValueError("Invalid method selected.")

    if method in ['median', 'mean']:
        fixed_data.loc[anomalies, 'CGM'] = replacement_value

    return fixed_data

first_run = True  # Flag to track the first run

def dataProcess(featureNumber,patientFlag, newPatientFlag, plotFlag):

    patientTrainList=['Used Dataset/all_data/540training.csv', 'Used Dataset/all_data/544training.csv', 'Used Dataset/all_data/552training.csv',
                        'Used Dataset/all_data/559training.csv', 'Used Dataset/all_data/563training.csv', 'Used Dataset/all_data/567training.csv',
                        'Used Dataset/all_data/570training.csv', 'Used Dataset/all_data/575training.csv', 'Used Dataset/all_data/584training.csv',
                        'Used Dataset/all_data/588training.csv', 'Used Dataset/all_data/591training.csv', 'Used Dataset/all_data/596training.csv']

    patientTestList = ['Used Dataset/all_data/540testing.csv', 'Used Dataset/all_data/544testing.csv', 'Used Dataset/all_data/552testing.csv',
                        'Used Dataset/all_data/559testing.csv', 'Used Dataset/all_data/563testing.csv', 'Used Dataset/all_data/567testing.csv',
                        'Used Dataset/all_data/570testing.csv', 'Used Dataset/all_data/575testing.csv', 'Used Dataset/all_data/584testing.csv',
                        'Used Dataset/all_data/588testing.csv', 'Used Dataset/all_data/591testing.csv', 'Used Dataset/all_data/596testing.csv' ]

    ####################################################################################################################

    dataset = pd.read_csv(patientTrainList[patientFlag], header=0, index_col=0, usecols=[i for i in range(featureNumber + 1)])
    test_dataset = pd.read_csv(patientTestList[patientFlag], header=0, index_col=0, usecols=[i for i in range(featureNumber + 1)])
    new_dataset = pd.read_csv(patientTrainList[newPatientFlag], header=0, index_col=0, usecols=[i for i in range(featureNumber + 1)])
    # dataset = dataset.fillna(dataset['CGM'].median())
    # dataset = dataset.to_frame()
    n_steps = 24

    def apply_isolation_forest(dataset, n_estimators=100, contamination=0.01, random_state=42):
        # Isolation Forest for anomaly detection on CGM data
        iso_forest = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=random_state)
        anomalies = iso_forest.fit_predict(dataset.values.reshape(-1, 1))

        # Adding a column for anomalies in the dataset
        dataset = dataset.to_frame()
        dataset['Anomaly'] = anomalies
        dataset['Anomaly'] = dataset['Anomaly'].map({1: 0, -1: 1})  # Mapping 1 to anomalies, 0 to normal

        return dataset

    # dataset = apply_isolation_forest(dataset)
    # new_dataset = apply_isolation_forest(new_dataset)
    

    # fixed_dataset_median=fix_anomalies(dataset, method='interpolate')
    # fixed_dataset_median=fix_anomalies(new_dataset, method='interpolate')

    #######################################################
    
    first_row_new_dataset = None
    model_filename = "ConvGRU_120ph_patientflag_0.h5" 
    global first_run
    if os.path.exists(model_filename):
        new_dataset = new_dataset.dropna()
        if first_run:
            first_datas = new_dataset.iloc[0:24] #train_x
            new_dataset = new_dataset[24:]
            print("first_datas", first_datas)
            dataset = np.vstack([dataset, first_datas])
            first_row_new_dataset = first_datas.iloc[-1:] #train_x
            print("bbbbb", first_row_new_dataset)
            first_run = False
        else:
            first_row_new_dataset = new_dataset.iloc[[0]] #train_x
            new_dataset = new_dataset[1:]
            print("first_row_new_dataset",first_row_new_dataset)
            dataset = np.vstack([dataset, first_row_new_dataset])
    DF_1 = pd.DataFrame(dataset)
    DF_1.to_csv(patientTrainList[patientFlag])
    DF_2 = pd.DataFrame(new_dataset)
    DF_2.to_csv(patientTrainList[newPatientFlag])

    #########################################################



    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in - 1, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out + 1):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    
    print(type(dataset))
    
    df5 = series_to_supervised(dataset, 24, 24)


    # ensure all data is float
    valuesTrain = df5.values
    valuesTrain = valuesTrain.astype('float32')


    # # normalize features
    # scaler = StandardScaler()
    # scaled = scaler.fit_transform(valuesTrain)
    # test_scaled = scaler.fit_transform(valuesTest)

    # train=scaled
    # test=test_scaled

    split_v= round(len(valuesTrain)*1)

    # split into input and outputs
    train_X0, train_y0 = valuesTrain[:split_v, :-24], valuesTrain[:split_v, -24:]



    # reshape input to be 3D [samples, timesteps, features]
    train_X0 = train_X0.reshape((train_X0.shape[0], 1, train_X0.shape[1]))

    input = train_X0[-1]
    target = train_y0[-1]


    train_X0_reshaped = input.reshape((input.shape[0], 1, input.shape[1]))
    train_y0_reshaped = target.reshape((1, 1, 24))

    return  train_X0, train_y0, train_X0_reshaped, train_y0_reshaped, first_row_new_dataset