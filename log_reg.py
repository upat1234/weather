# Author: Patrick Underwood (upatrick@vt.edu)
# Version: 2024-12-10

from math import exp
import numpy as np
import random
import time
import csv
import io
import os

# This function computes the logistic function.
def logistic(x):
    s = 1.0 / (1 + exp(-x))
    return s

# This function computes the dot product.
def dot(x, y):
    s = 0.0
    for i in range(len(x)):
        s += (x[i] * y[i])
    return s

# This function makes a prediction based on a list of lists of feature values.
def predict(model, point):
    product = dot(model, point['features'])
    return logistic(product)

# This function returns the accuracy of predictions.
def accuracy(data, predictions, threshold):
    threshold = float(threshold)*0.1
    correct = 0
    for i in range(len(data)):
        # Change the double to any number 0.0 <= number <= 1.0 you desire.
        # A higher threshold will give you more false negatives and a
        # lower one will give you false positives.
        p_label = predictions[i] >= threshold
        r_label = data[i]['label']
        if p_label == r_label:
            correct += 1
    return float(correct)/len(data)

# Extracts the necessarry features and label from the csv data.
def extract_features(raw):
    data = []
    for r in raw:
        point = {}
        # Label is based on the 'RSH' values of 'FRSHTT' from GSOD data.
        point['label'] = (r['FRSHTT'][1] == '1' or r['FRSHTT'][2] == '1' or r['FRSHTT'][3] == '1')

        # Features numbers are adjusted based on unit of measurement.
        features = []
        features.append(1.)
        features.append((float(r['DATE'][:4])-2013)/12)
        features.append((((float(r['DATE'][5:7])-1)*30)+float(r['DATE'][8:10]))/365)
        if r['ELEVATION'] == '' or r['ELEVATION'] == 'NULL':
            features.append(float(0))
        else:
            features.append(float(r['ELEVATION'])/4000)
        if r['TEMP'] == '9999.9':
            features.append(float(0))
        else:
            features.append(((((float(r['TEMP'])-32)*5)/9)+125.15)/200)
        if r['DEWP'] == '9999.9':
            features.append(float(0))
        else:
            features.append(((((float(r['DEWP'])-32)*5)/9)+125.15)/200)
        point['features'] = features
        data.append(point)
    return data

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
            
# Initializes log_reg model
def initialize_model(k):
    return [random.gauss(0, 1) for x in range(k)]

# Trains log_reg model
def train(data, vdata, epochs, rate, lam, threshold):
    model = initialize_model(len(data[0]['features']))
    n = len(data)
    d = len(data[0]['features'])
    val_accs = []
    for e in range(epochs):
        grad = []
        selected_data = []
        for i in range(n):
            idx = random.randint(0,n-1)
            point = data[idx]
            selected_data.append(point)
        for i in range(d):
            grad_d = 0
            for j in range(n):
                point = selected_data[j]
                grad_d += rate * (point['label']-predict(model, point)) * point['features'][i]
            grad_d -= rate * lam * model[i]
            grad.append(grad_d)
        delta = grad
        new_model = []
        d = len(delta)
        for i in range(d):
            new_model.append(model[i] + delta[i])
        model = new_model

        vp = [predict(model, p) for p in vdata]
        vacc = accuracy(vdata, vp, threshold)
        val_accs.append(vacc)
    return model, val_accs

# Splts data based on a ratio into two groups.
def split_data(data, train_ratio=0.8):
    random.shuffle(data)
    split_index = int(len(data) * train_ratio)
    train_data = data[:split_index]
    valid_data = data[split_index:]
    return train_data, valid_data

def rmse(data, predictions):
    errors = [(point['label'] - pred) ** 2 for point, pred in zip(data, predictions)]
    return np.sqrt(np.mean(errors))

def mae(data, predictions):
    errors = [abs(point['label'] - pred) for point, pred in zip(data, predictions)]
    return np.mean(errors)

def r2(data, predictions):
    actuals = [point['label'] for point in data]
    mean_actual = np.mean(actuals)
    ss_total = sum((label - mean_actual) ** 2 for label in actuals)
    ss_residual = sum((point['label'] - pred) ** 2 for point, pred in zip(data, predictions))
    return 1 - (ss_residual / ss_total if ss_total != 0 else 1)

def main():
    threshold = input('Enter threshold (Any number 0.0-10.0): ')
    if not os.path.isdir("log_reg_models" + str(threshold)):
        os.mkdir("log_reg_models" + str(threshold))
    start_time = time.time()
    count = 1
    with open('new_station_list', 'r') as file:
        with open('log_reg_results_' + str(threshold) + 'threshold', 'w') as results:
            for line in file:
                random.seed(1)
                data, _ = load_all_data(line[:-1])
                print('File ' + str(count) + ' OUT OF 10156 || Station: ' + line[:-1])
                train_data, valid_data = split_data(data)
                train_data = extract_features(train_data)
                valid_data = extract_features(valid_data)
                model, val_accs = train(train_data, valid_data, 10, 1e-4, 1e-3, threshold)
                train_predictions = [predict(model, p) for p in train_data]
                valid_predictions = [predict(model, p) for p in valid_data]
                train_rmse = rmse(train_data, train_predictions)
                train_mae = mae(train_data, train_predictions)
                train_r2 = r2(train_data, train_predictions)
                valid_rmse = rmse(valid_data, valid_predictions)
                valid_mae = mae(valid_data, valid_predictions)
                valid_r2 = r2(valid_data, valid_predictions)
                results.write('Station: ' + line[:-1] + '\n')
                results.write("Training Accuracy    " + str(accuracy(train_data, train_predictions, threshold)) + '\n')
                results.write("Training RMSE:       " + str(train_rmse) + '\n')
                results.write("Training MAE:        " + str(train_mae) + '\n')
                results.write("Training R^2:        " + str(train_r2) + '\n')
                results.write("Validation Accuracy: " + str(accuracy(valid_data, valid_predictions, threshold)) + '\n')
                results.write("Validation RMSE:     " + str(valid_rmse) + '\n')
                results.write("Validation MAE:      " + str(valid_mae) + '\n')
                results.write("Validation R^2:      " + str(valid_r2) + '\n')
                c = 0
                bestacc = 0
                idx = 1
                while c < len(val_accs):
                    if c == 0:
                        results.write("Epoch 1/10  - val_accuracy is " + str(val_accs[c]) + '\n')
                    elif c == len(val_accs)-1:
                        if val_accs[c] >= val_accs[c-1]:
                            results.write("Epoch 10/10 - val_accuracy IMPROVED: " +
                                str(val_accs[c-1]) + " to " + str(val_accs[c]) + "\n\n")
                        else:
                            results.write("Epoch 10/10 - val_accuracy  DROPPED: " +
                                str(val_accs[c-1]) + " to " + str(val_accs[c]) + "\n\n")
                    else:
                        if val_accs[c] >= val_accs[c-1]:
                            results.write("Epoch " + str(c+1) + "/10  - val_accuracy IMPROVED: " +
                                str(val_accs[c-1]) + " to " + str(val_accs[c]) + "\n")
                        else:
                            results.write("Epoch " + str(c+1) + "/10  - val_accuracy  DROPPED: " +
                                str(val_accs[c-1]) + " to " + str(val_accs[c]) + "\n")
                    if val_accs[c] > bestacc:
                        bestacc = val_accs[c]
                        idx = c+1
                    c += 1
                results.write("Best val_accuracy: " + str(bestacc) + " at epoch " + str(idx) + "\n\n")

                    
                file_path = os.path.join("log_reg_models" + str(threshold), line[:-1] + '_model.txt')
                with open(file_path, 'w') as model_file:
                    model_file.write(str(model))
                count += 1
            end_time = time.time()
            exe_time = end_time - start_time
            results.write('This program took ' + str(exe_time) + ' seconds to complete.')
    

if __name__ == "__main__":
    main()