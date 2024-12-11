# Author: Patrick Underwood (upatrick@vt.edu)
# Version: 2024-12-10

import log_reg

# Creates a list of stations which includes every station from
# 'station_list' that has over 3000 data entries from 2014-2024
def main():
    count = 0
    with open('station_list', 'r') as file:
        with open('new_station_list', 'w') as nfile:
            for line in file:
                _, count = log_reg.load_all_data(line[:-1])
                if count >= 3000:
                    nfile.write(line)
                    print(line[:-1] + ' added')
                else:
                    count += 1
                    print(line[:-1] + ' REMOVED')
    print('Total removed: ' + str(count))

if __name__ == "__main__":
    main()