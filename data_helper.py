import numpy as np
import pandas as pd

def load_timeseries(filename, params):
    """Load time series dataset"""

    series = pd.read_csv(filename, sep=',', header=0, index_col=0, squeeze=True)
    data = series.values

    # Note: we should be calculating normalisation parameters only based on test data
    test_data_portion = data[round(params['train_test_split'] * len(data)):]
    mean_v = np.mean(test_data_portion)
    print(f'Mean: {mean_v}')
    data =  [(x - mean_v) for x in data]

    std_v = np.std(test_data_portion)
    print(f'Std: {std_v}')
    data =  [(x/std_v) for x in data]

    print(f'DEBUG: len(data): {len(data)}')
    #input('Press a key to continue...')

    adjusted_window = params['window_size']+ 1
    print(f'DEBUG: adjusted_window: {adjusted_window}')
    #input('Press a key to continue...')


    # Split data into windows
    raw = []
    for index in range(len(data) - adjusted_window):
            raw.append(data[index: index + adjusted_window])

    print(f'DEBUG: len(raw): {len(raw)}')
    #input('Press a key to continue...')

    # Normalize data
    result = normalize_windows(raw)

    print(f'DEBUG: len(result): {len(result)}')
    #input('Press a key to continue...')

    raw = np.array(raw)
    result = np.array(result)

    # Split the input dataset into train and test
    split_ratio = round(params['train_test_split'] * result.shape[0])

    print(f'DEBUG: split_ratio: {split_ratio}')
    #input('Press a key to continue...')

    train = result[:int(split_ratio), :]
    print(train)
    np.random.shuffle(train)

    # x_train and y_train, for training
    x_train = train[:, :-1]
    y_train = train[:, -1]

    # x_test and y_test, for testing
    x_test = result[int(split_ratio):, :-1]
    y_test = result[int(split_ratio):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))  

    x_test_raw = raw[int(split_ratio):, :-1]
    y_test_raw = raw[int(split_ratio):, -1]

    # Last window, for next time stamp prediction
    last_raw = [data[-params['window_size']:]]
    last = normalize_windows(last_raw)
    last = np.array(last)
    last = np.reshape(last, (last.shape[0], last.shape[1], 1))

    return [x_train, y_train, x_test, y_test, x_test_raw, y_test_raw, last_raw, last]

def normalize_windows(window_data):
    """Normalize data"""

    normalized_data = []
    
    for window in window_data:

        norm_factor  = float(window[0])
        if norm_factor == 0.0:
            print('DEBUG first element is zero')
            for x in window:
                if x != 0.0:
                    norm_factor = x
                    break

        #normalized_window = [((float(p) / norm_factor) - 1) for p in window]
        normalized_window = window
        normalized_data.append(normalized_window)

    return normalized_data
