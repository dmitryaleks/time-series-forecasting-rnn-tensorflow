import sys
import json
import build_model
import data_helper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from keras.callbacks import ModelCheckpoint

def root_mean_squared_error(y_true, y_pred):
    return math.sqrt(sum(map(lambda a_b: pow(a_b[0] - a_b[1], 2) , zip(y_pred, y_true)))/y_true.__len__())

def train_predict():
    """Train and predict time series data"""

    # Load command line arguments 
    train_file = sys.argv[1]
    parameter_file = sys.argv[2]

    # Load training parameters
    params = json.loads(open(parameter_file).read())

    # Load time series dataset, and split it into train and test
    x_train, y_train, x_test, y_test, x_test_raw, y_test_raw,\
            last_window_raw, last_window = data_helper.load_timeseries(train_file, params)

    # Build RNN (LSTM) model
    lstm_layer = [1, params['window_size'], params['hidden_unit'], 1]
    model = build_model.rnn_lstm(lstm_layer, params)
    print(f'DEBUG: len(x_train): {len(x_train)}')
    print(f'DEBUG: len(y_train): {len(y_train)}')
    print(f'DEBUG: len(x_test): {len(x_test)}')
    print(f'DEBUG: len(y_test): {len(y_test)}')

    weight_file = 'model_best_weights.h5'
    checkpoint = ModelCheckpoint(weight_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)

    # Train RNN (LSTM) model with train set
    history = model.fit(
            x_train,
            y_train,
            batch_size=params['batch_size'],
            epochs=params['epochs'],
            validation_split=params['validation_split'],
            verbose=True,
            callbacks=[checkpoint])

    model.load_weights(weight_file)

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

    # Check the model against test set
    predicted = build_model.predict_next_timestamp(model, x_test)        
    predicted_raw = []
    for i in range(len(x_test_raw)):
            predicted_raw.append(predicted[i])

    # Debug info
    print("DEBUG: Predicted test data:")
    for e in predicted:
        print(e)
    print("DEBUG: Predicted (raw) test data:")
    for e in predicted_raw:
        print(e)
    print("DEBUG: Actual test data:")
    for e in y_test_raw:
        print(e)

    rmse = root_mean_squared_error(predicted_raw, y_test_raw)
    print("RMSE: %f" % rmse)

    # Plot graph: predicted VS actual
    plt.subplot(111)
    plt.plot(predicted_raw, label='Predicted', marker='o')
    plt.plot(y_test_raw, label='Actual', marker='o')
    #plt.ylim(ymin=0)
    plt.legend()
    plt.show()

    # Predict next time stamp 
    next_timestamp = build_model.predict_next_timestamp(model, last_window)
    next_timestamp_raw = (next_timestamp[0] + 1) * last_window_raw[0][0]
    print('The next time stamp forecasting is: {}'.format(next_timestamp_raw))

if __name__ == '__main__':
    # python3 train_predict.py ./data/sales.csv ./training_config.json_
    train_predict()
