from keras.models import Sequential 
from keras.layers import Dense, Activation, LSTM, Bidirectional, Conv1D, GRU, SimpleRNN
import keras
from keras import optimizers, Model, Input
import tensorflow as tf
import os
import numpy as np
from numpy import concatenate
from math import sqrt
from keras.metrics import mean_squared_error
import matplotlib.pyplot as plt
import random
from result import result
from predict import app_predict
from attention import Attention
from dataProcess import dataProcess
# np.set_printoptions(threshold=np.inf)


# tensorboard_callback = keras.callbacks.TensorBoard(
#     log_dir="./Logs",
#     histogram_freq=1,
#     write_graph=True,
#     write_images=False,
#     write_steps_per_second=False,
#     update_freq="batch",
#     profile_batch=0,
#     embeddings_freq=0,
#     embeddings_metadata=None,
# )

earlyStopping = keras.callbacks.EarlyStopping(monitor="loss", patience=4)


def train_model(seedNumber, epoch, modelType, testFlag, plotFlag, patientFlag, newPatientFlag, featureNumber):

    os.environ['PYTHONHASHSEED'] = str(seedNumber)

    random.seed(seedNumber)
    np.random.seed(seedNumber)
    tf.random.set_seed(seedNumber)
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                            inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)


    model_filename = "ConvGRU_120ph_patientflag_0.h5" 

    train_X0, train_y0, train_X0_reshaped, train_y0_reshaped, first_row_new_dataset = dataProcess(featureNumber, patientFlag, newPatientFlag, plotFlag)

    if os.path.exists(model_filename):
        model = keras.models.load_model(model_filename, custom_objects = {'Attention': Attention})
        print("Existing model loaded.")
        
        history = model.fit(train_X0_reshaped, train_y0_reshaped, epochs=50, batch_size=32, verbose=0, callbacks=[earlyStopping])
        model.save("ConvGRU_120ph_patientflag_0.h5")

        prediction = app_predict(train_y0_reshaped, model_filename, first_row_new_dataset)

        return prediction

    else:
        # model = Sequential()
        # model.add(Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=(train_X0.shape[1], train_X0.shape[2])))
        # model.add(GRU(128, return_sequences=False, input_shape=(train_X0.shape[1], train_X0.shape[2])))
        # model.add(Dense(128, activation='relu'))
        # model.add(Dense(64, activation='relu'))
        # model.add(Dense(24, activation='linear'))
        
        model = Sequential()
        model_input = Input(shape=(train_X0.shape[1], train_X0.shape[2]))
        print("model_input", model_input)
        print("train_X0", train_X0)
        x = Conv1D(filters=32, kernel_size=1, activation='relu')(model_input)
        x = GRU(512, return_sequences=True)(x)
        x = GRU(512, return_sequences=True)(x)
        x = Attention(units=512)(x)
        x = Dense(24)(x)
        x = Activation("linear")(x)
        model = Model(model_input, x)
        rmsprop = optimizers.RMSprop(learning_rate=0.0001, rho=0.9, epsilon=1e-08)

        model.compile(loss='mse',
                    optimizer=rmsprop, metrics=['accuracy'])
        model.fit(train_X0, train_y0, epochs=epoch, batch_size=32, verbose=1, callbacks=[])
        model.save("ConvGRU_120ph_patientflag_0.h5")
        print("New model created.")

        # predict = model.predict(train_X0)
        # initial_rmse = sqrt(mean_squared_error(predict, train_y0)) #################################
        # print("initial_rmse", initial_rmse)

    