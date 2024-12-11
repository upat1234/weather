# Author: Patrick Underwood (upatrick@vt.edu)
# Version: 2024-12-10

# This script creates a results file based on the results of the FNN models being made
# and finds information regarding the success of the models.
def main():
    threshold = input('Enter threshold (Any number 0.0-10.0): ')
    val_bins = [0] * 10
    train_bins = [0] * 10
    best_val_bins = [0] * 10
    with open("fnn_results_" + str(threshold) + "threshold", 'r') as file:
        i=1
        for line in file:
            if line[0] == 'T' and line[1] != 'h':
                acc = float(line[28:-1])
                if acc < 0.1:
                    train_bins[0]+=1
                elif acc < .2:
                    train_bins[1]+=1
                elif acc < .3:
                    train_bins[2]+=1
                elif acc < .4:
                    train_bins[3]+=1
                elif acc < .5:
                    train_bins[4]+=1
                elif acc < .6:
                    train_bins[5]+=1
                elif acc < .7:
                    train_bins[6]+=1
                elif acc < .8:
                    train_bins[7]+=1
                elif acc < .9:
                    train_bins[8]+=1
                else:
                    train_bins[9]+=1
            elif line[0] == 'V':
                acc = float(line[30:-1])
                if acc < 0.1:
                    val_bins[0]+=1
                elif acc < .2:
                    val_bins[1]+=1
                elif acc < .3:
                    val_bins[2]+=1
                elif acc < .4:
                    val_bins[3]+=1
                elif acc < .5:
                    val_bins[4]+=1
                elif acc < .6:
                    val_bins[5]+=1
                elif acc < .7:
                    val_bins[6]+=1
                elif acc < .8:
                    val_bins[7]+=1
                elif acc < .9:
                    val_bins[8]+=1
                else:
                    val_bins[9]+=1
            elif line[0] == 'B':
                if line[19:22] == '0 a':
                    best_val_bins[0]+=1
                else:
                    acc = float(line[19:22])
                    if acc < 0.1:
                        best_val_bins[0]+=1
                    elif acc < .2:
                        best_val_bins[1]+=1
                    elif acc < .3:
                        best_val_bins[2]+=1
                    elif acc < .4:
                        best_val_bins[3]+=1
                    elif acc < .5:
                        best_val_bins[4]+=1
                    elif acc < .6:
                        best_val_bins[5]+=1
                    elif acc < .7:
                        best_val_bins[6]+=1
                    elif acc < .8:
                        best_val_bins[7]+=1
                    elif acc < .9:
                        best_val_bins[8]+=1
                    else:
                        best_val_bins[9]+=1
            i+=1
    train_avg_min = train_bins[1]*0.1 + train_bins[2]*0.2 + train_bins[3]*0.3 + train_bins[4]*0.4 + train_bins[5]*0.5 + train_bins[6]*0.6 + train_bins[7]*0.7 + train_bins[8]*0.8 + train_bins[9]*0.9
    val_avg_min = val_bins[1]*0.1 + val_bins[2]*0.2 + val_bins[3]*0.3 + val_bins[4]*0.4 + val_bins[5]*0.5 + val_bins[6]*0.6 + val_bins[7]*0.7 + val_bins[8]*0.8 + val_bins[9]*0.9
    best_val_avg_min = best_val_bins[1]*0.1 + best_val_bins[2]*0.2 + best_val_bins[3]*0.3 + best_val_bins[4]*0.4 + best_val_bins[5]*0.5 + best_val_bins[6]*0.6 + best_val_bins[7]*0.7 + best_val_bins[8]*0.8 + val_bins[9]*0.9
    train_avg_min = train_avg_min/10156.0
    val_avg_min = val_avg_min/10156.0
    best_val_avg_min = best_val_avg_min/10156.0

    train_avg_max = train_bins[0]*0.1 + train_bins[1]*0.2 + train_bins[2]*0.3 + train_bins[3]*0.4 + train_bins[4]*0.5 + train_bins[5]*0.6 + train_bins[6]*0.7 + train_bins[7]*0.8 + train_bins[8]*0.9 + train_bins[9]
    val_avg_max = val_bins[0]*0.1 + val_bins[1]*0.2 + val_bins[2]*0.3 + val_bins[3]*0.4 + val_bins[4]*0.5 + val_bins[5]*0.6 + val_bins[6]*0.7 + val_bins[7]*0.8 + val_bins[8]*0.9 + val_bins[9]
    best_val_avg_max = best_val_bins[0]*0.1 + best_val_bins[1]*0.2 + best_val_bins[2]*0.3 + best_val_bins[3]*0.4 + best_val_bins[4]*0.5 + best_val_bins[5]*0.6 + best_val_bins[6]*0.7 + best_val_bins[7]*0.8 + best_val_bins[8]*0.9 + val_bins[9]
    train_avg_max = train_avg_max/10156.0
    val_avg_max = val_avg_max/10156.0
    best_val_avg_max = best_val_avg_max/10156.0

    train_avg = (train_avg_min + train_avg_max)/2
    val_avg = (val_avg_min + val_avg_max)/2
    best_val_avg = (best_val_avg_min + best_val_avg_max)/2
    with open("fnn_data_dist_results_" + str(threshold) + "threshold", 'w') as wfile:
        wfile.write("Training Accuracy Spread:         " + str(train_bins) + '\n')
        wfile.write("Training Accuracy Average:        " + str(train_avg) + '\n')
        wfile.write("Validation Accuracy Spread:       " + str(val_bins) + '\n')
        wfile.write("Validation Accuracy Average:      " + str(val_avg) + '\n')
        wfile.write("Best Validation Accuracy Spread:  " + str(val_bins) + '\n')
        wfile.write("Best Validation Accuracy Average: " + str(best_val_avg) + '\n')
        wfile.write("Averages are rough estimates based on the spreads.")

if __name__ == "__main__":
    main()