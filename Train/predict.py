import keras
import tensorflow as tf
import numpy as np
from keras.metrics import mean_squared_error
from result import result
from attention import Attention

prediction_matrix = np.zeros((24, 24))

def app_predict(train_X0_reshaped, model_filename, first_row_new_dataset):
    
    model = keras.models.load_model(model_filename, custom_objects = {'Attention': Attention})
    prediction = model.predict(train_X0_reshaped)
    prediction = prediction.tolist()
    prediction = prediction[0]
    print("prediction\n********\n", prediction)

    # Function to update the matrix with a new prediction
    def update_matrix(matrix, new_prediction):
        # Shift all rows up by one position
        matrix[:-1, :] = matrix[1:, :]
        
        # Add the new prediction as the last row
        matrix[-1, :] = new_prediction

    
    print("\nUpdated Matrix after New Prediction:")
    print("prediction_matrix",prediction_matrix)
    diagonal_elements = np.fliplr(prediction_matrix).diagonal()
    diagonal_elements = np.flip(diagonal_elements)
    print("diagonal_elements",diagonal_elements)

    rmse_values = []
    for i in range(len(diagonal_elements)):
        rmse = np.sqrt(mean_squared_error([first_row_new_dataset][0], [diagonal_elements[i]]))
        rmse_values.append(rmse)


    print("RMSE values for each pair:")
    print(rmse_values)
    result(rmse_values)
    update_matrix(prediction_matrix, prediction)

    return prediction