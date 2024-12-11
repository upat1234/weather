# Author: Patrick Underwood (upatrick@vt.edu)
# Version: 2024-12-10

import log_reg
import os

# Get information from saved model file.
def load_model(station, model_num):
    folder = 'log_reg_models' + str(model_num)
    path = os.path.join(folder, str(station) + '_model.txt')
    with open(path, 'r') as file:
        model = eval(file.read())
    return model

# Find the average of numbers in a list.
def average(list):
    sum = 0.0
    for item in list:
        sum += float(item)
    return sum/len(list)

# Return a set of predictions based on a model and prediction data.
def predict_weather(model, data):
    new_data = log_reg.extract_features(data)
    predictions = [log_reg.predict(model, p) for p in new_data]
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
            lines, _ = log_reg.csv_load(path)
            for dline in lines:
                if dline['DATE'][5:] == small_date:
                    date_data.append(dline)
                    break
    if date_data == []:
        print("There is not sufficient data on this date at this station to make a prediction.\nPlease check the 'weather_data' folder and choose a similar date that has data to predict.")
        return
    model_num = input("Enter threshold number (the number after a 'log_reg_models' folder): ")
    model = load_model(station, model_num)
    predictions = predict_weather(model, date_data)
    pred_avg = average(predictions)
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
    results_file = 'log_reg_data_dist_results_' + str(model_num) + 'threshold'
    overall_acc = 0
    with open(results_file, 'r') as rfile:
        i=1
        for rline in rfile:
            if i == 4:
                overall_acc = rline[34:-1]
                break
            i+=1
    this_acc = 0
    results_file2 = 'log_reg_results_' + str(model_num) + 'threshold'
    found = False
    with open(results_file2, 'r') as rfile2:
        i=1
        for rline2 in rfile2:
            if (i-1)%22 == 0:
                tstation = rline2[9:-1]
                if tstation == str(station):
                    found = True
            if (i-6)%22 == 0:
                if found:
                    this_acc = rline2[21:-1]
                    break
            i+=1
    print("The accuracy for this station's model is " + str(float(this_acc)*100.0) + '%')
    print("The overall accuracy accross all models is " + str(float(overall_acc)*100.0) + '%')

if __name__ == "__main__":
    main()