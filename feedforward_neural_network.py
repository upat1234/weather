# Author: Patrick Underwood (upatrick@vt.edu)
# Version: 2024-12-10

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics
import numpy as np
import os
import csv
import io
import time

# Extracts the necessarry features and label from the csv data.
def extract_features(raw):
    data = []
    labels = []
    # Label is based on the 'RSH' values of 'FRSHTT' from GSOD data.
    for r in raw:
        if r['FRSHTT'][1] == '1' or r['FRSHTT'][2] == '1' or r['FRSHTT'][3] == '1':
            label = 1.0
        else:
            label = 0.0
        labels.append(label)

        # Features numbers are adjusted based on unit of measurement.
        features = []
        features.append(1.0)
        try:
            year = float(r['DATE'][:4])
            features.append((year - 2013.0) / 12.0)
        except ValueError:
            features.append(0.0)
        try:
            month_day = (float(r['DATE'][5:7]) - 1) * 30.0 + float(r['DATE'][8:10])
            features.append(month_day / 365.0)
        except ValueError:
            features.append(0.0)
        try:
            elevation = r['ELEVATION']
            if elevation == '' or elevation == 'NULL':
                features.append(0.0)
            else:
                features.append(float(elevation) / 4000.0)
        except ValueError:
            features.append(0.0)
        try:
            temp = r['TEMP']
            if temp == '9999.9' or temp == '':
                features.append(0.0)
            else:
                features.append(((float(temp) - 32.0) * 5.0 / 9.0 + 125.15) / 200.0)
        except ValueError:
            features.append(0.0)
        try:
            dewp = r['DEWP']
            if dewp == '9999.9' or dewp == '':
                features.append(0.0)
            else:
                features.append(((float(dewp) - 32.0) * 5.0 / 9.0 + 125.15) / 200.0)
        except ValueError:
            features.append(0.0)
        data.append(features)
    data = np.nan_to_num(data)
    return np.array(data), np.array(labels)

# Loads csv data from a certain file.
def csv_load(filename):
    lines = []
    count = 0
    with open(filename, 'rb') as csvfile:
        content = csvfile.read()
        content = content.replace(b'\0', b'')
        content = content.decode('utf-8', errors='ignore')
        csvfile = io.StringIO(content)
        reader = csv.DictReader(csvfile)
        for line in reader:
            lines.append(dict(line))
            count += 1
    return lines, count

# Loads data from all csv's of a certain station.
def load_all_data(station):
    all_lines = []
    count = 0
    for root, _, files in os.walk('weather_data'):
        for file in files:
            if file == station + ".csv":
                path = os.path.join(root, file)
                new_lines, new_count = csv_load(path)
                all_lines = all_lines + new_lines
                count += new_count
    return all_lines, count

# Builds FNN model
def build_model(input):
    model = models.Sequential([
        layers.InputLayer(shape=(input,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Shuffles feature and label data in the same way
def shuffle(data, labels):
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    return data[idx], labels[idx]

def rmse_metric(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def r2_metric(y_true, y_pred):
    y_true_mean = tf.reduce_mean(y_true)
    ss_total = tf.reduce_sum(tf.square(y_true - y_true_mean))
    ss_residual = tf.reduce_sum(tf.square(y_true - y_pred))
    return 1-(ss_residual/(ss_total + tf.keras.backend.epsilon()))

def main():
    threshold = input('Enter threshold (Any number 0.0-10.0): ')
    if not os.path.isdir("fnn_models" + str(threshold)):
        os.mkdir("fnn_models" + str(threshold))
    start_time = time.time()
    count=1
    with open('new_station_list', 'r') as file:
        # Change 'w' to 'a' to append to file if program crashed.
        with open('fnn_results_' + str(threshold) + 'threshold', 'w') as results:
            for line in file:
                # Change number count is greater than to decide when to resume.
                if count > 0:
                    print('File ' + str(count) + ' OUT OF 10156 || Station: ' + line[:-1])
                    station = line[:-1]
                    data, _ = load_all_data(station)
                    data, labels = extract_features(data)
                    data, labels = shuffle(data, labels)
                    train_ratio = 0.8
                    split = int(len(data)*train_ratio)
                    train_data, val_data = data[:split], data[split:]
                    train_labels, val_labels = labels[:split], labels[split:]
                    model = build_model(train_data.shape[1])
                    model.compile(
                        optimizer=optimizers.Adam(learning_rate=1e-4),
                        loss=losses.BinaryCrossentropy(),
                        metrics=[metrics.BinaryAccuracy(threshold=float(threshold)*.1), metrics.MeanSquaredError(), rmse_metric, r2_metric]
                    )
                    class EpochLogger(tf.keras.callbacks.Callback):
                        def __init__(self, results):
                            self.results = results
                            self.best_val_accuracy = 0
                            self.best_epoch = 0

                        def on_epoch_end(self, epoch, logs=None):
                            current_val_accuracy = logs.get('val_binary_accuracy', 0)
                            self.results.write(
                                f"Epoch {epoch + 1}/{self.params['epochs']} - "
                                f"val_accuracy: {current_val_accuracy:.6f}\n"
                            )
                            if current_val_accuracy > self.best_val_accuracy:
                                self.results.write(
                                    f"Epoch {epoch + 1}/{self.params['epochs']} - "
                                    f"val_accuracy IMPROVED: {self.best_val_accuracy:.6f} to {current_val_accuracy:.6f}\n"
                                )
                                self.best_val_accuracy = current_val_accuracy
                                self.best_epoch = epoch + 1
                    epoch_logger = EpochLogger(results)
                    history = model.fit(
                        train_data, train_labels,
                        validation_data=(val_data, val_labels),
                        epochs=10,
                        batch_size=32,
                        verbose=0,
                        callbacks=[epoch_logger]
                    )
                    results.write("Station: " + station + '\n')
                    results.write("Training Metrics: ")
                    train_metrics = model.evaluate(train_data, train_labels, verbose=0)
                    results.write("Accuracy: " + str(train_metrics[1]) + '\n')
                    results.write("RMSE:     " + str(train_metrics[2]) + '\n')
                    results.write("MAE:      " + str(train_metrics[3]) + '\n')
                    results.write("R^2:      " + str(train_metrics[4]) + '\n')
                    results.write("Validation Metrics: ")
                    val_metrics = model.evaluate(val_data, val_labels, verbose=0)
                    results.write("Accuracy: " + str(val_metrics[1]) + '\n')
                    results.write("RMSE:     " + str(val_metrics[2]) + '\n')
                    results.write("MAE:      " + str(val_metrics[3]) + '\n')
                    results.write("R^2:      " + str(val_metrics[4]) + '\n')
                    results.write("Best val_accuracy: "+str(epoch_logger.best_val_accuracy)+" at epoch "+str(epoch_logger.best_epoch)+'\n\n')
                    model_path = os.path.join("fnn_models" + str(threshold), station + '_model.keras')
                    model.save(model_path)
                count+=1
            end_time = time.time()
            exe_time = end_time - start_time
            results.write('This program took ' + str(exe_time) + ' seconds to complete.')


if __name__ == "__main__":
    main()

