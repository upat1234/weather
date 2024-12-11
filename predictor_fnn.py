# Author: Patrick Underwood (upatrick@vt.edu)
# Version: 2024-12-10

import os
import feedforward_neural_network as fnn
import tensorflow as tf
import numpy as np

# Get information from saved model file.
def load_model(station, model_num):
    folder = 'fnn_models' + str(model_num)
    path = os.path.join(folder, str(station) + '_model.keras')
    model = tf.keras.models.load_model(
        path, 
        custom_objects={
            'rmse_metric': fnn.rmse_metric,
            'r2_metric': fnn.r2_metric
        }
    )
    return model

# Return a set of predictions based on a model and prediction data.
def predict_weather(model, data):
    features, _ = fnn.extract_features(data)
    predictions = model.predict(features)
    return predictions

def main():
    station = input("Enter Station number: ")
    vstation = False
    with open('new_station_list', 'r') as file:
        for line in file:
            if line[:-1] == station:
                vstation = True
                break
    if not vstation:
        print("This is not a valid station.\nPlease check 'new_station_list' for valid station numbers.")
        return
    date = input("Enter date (yyyy-mm-dd): ")
    small_date = date[5:]
    folder = station[0] + '_files'
    date_data = []
    for i in range(14, 25):
        yfolder = '20' + str(i) + '_weather_data'
        path = os.path.join('weather_data', yfolder)
        path = os.path.join(path, folder)
        path = os.path.join(path, station + '.csv')
        if os.path.isfile(path):
            lines, _ = fnn.csv_load(path)
            for dline in lines:
                if dline['DATE'][5:] == small_date:
                    date_data.append(dline)
                    break
    if date_data == []:
        print("There is not sufficient data on this date at this station to make a prediction.\nPlease check the 'weather_data' folder and choose a similar date that has data to predict.")
        return
    model_num = input("Enter threshold number (the number after a 'fnn_models' folder): ")
    model = load_model(station, model_num)
    predictions = predict_weather(model, date_data)
    pred_avg = np.mean(predictions)
    total = 0
    yes = 0
    for pred in predictions:
        if pred >= float(model_num) * 0.1:
            yes+=1
        total+=1
    print("Prediction data for:\nStation: " + str(station) + "\nDate: " + str(date))
    if pred_avg > float(model_num) * 0.1:
        print("Prediction: It will percipitate.")
        print(str(yes) + " out of " + str(total) + " calculations predict this.")
    else:
        print("Prediction: It will not precipitate.")
        print(str(total-yes) + " out of " + str(total) + " calculations predict this.")
    print("Probability of percipitation is " + str(pred_avg*100.0) + '%')
    results_file = 'fnn_data_dist_results_' + str(model_num) + 'threshold'
    overall_acc = 0
    with open(results_file, 'r') as rfile:
        i=1
        for rline in rfile:
            if i == 4:
                overall_acc = rline[34:-1]
                break
            i+=1
    this_acc = 0
    results_file2 = 'fnn_results_' + str(model_num) + 'threshold'
    found = False
    with open(results_file2, 'r') as rfile2:
        i=1
        for rline2 in rfile2:
            if rline2[0] == 'S':
                tstation = rline2[9:-1]
                if tstation == str(station):
                    found = True
            if rline2[0] == 'V':
                if found:
                    this_acc = rline2[30:-1]
                    break
            i+=1
    print("The accuracy for this station's model is " + str(float(this_acc)*100.0) + '%')
    print("The overall accuracy accross all models is " + str(float(overall_acc)*100.0) + '%')

if __name__ == "__main__":
    main()