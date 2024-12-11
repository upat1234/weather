Project Name: ML Models for Weather Prediction

Description: Provides a logistic regression model and feedforward neural network for weather prediction.

Installation:
 - Install .py files
 - Install numpy
 - Install tensorflow
 - Install bs4 (BeautifulSoup)

Usage:
ALL EXAMPLES OF RESULTS ARE DERIVED FROM 2014 TO 2024 DATA!!!
 - Run webcrawler.py to download GSOD data from selected years.
 - Run file_name_getter.py to create a list of stations.
 - Run station_list_edit.py to create a list of stations with enough data to train.
 - (1) Run log_reg.py and follow instructions to create log_reg models and results.
 - (1) Run log_reg_data_dist.py and follow instructions to get data distributions.
 - (1) Run predictor_log_reg.py to get weather predictions.
 - (2) Run feedforward_neural_network.py and follow instructions to create log_reg models and results.
 - (2) Run fnn_data_dist.py and follow instructions to get data distributions.
 - (2) Run predictor_fnn.py to get weather predictions.
(You can check isd-history.csv to find stations for location names)
(An example of how to run the predictors can be found in example.jpg)

Dependencies: Python 3.11.9

Author: Patrick Underwood (upatrick@vt.edu)
