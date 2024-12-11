# Author: Patrick Underwood (upatrick@vt.edu)
# Version: 2024-12-10

import os

# Creates a file called 'station_list' which includes
# all station names in the data scraped from GSOD.
def main():
    ufiles = set()
    with open('station_list', 'w') as nfile:
        for _, _, files in os.walk('weather_data'):
            for file in files:
                ufiles.add(file[:-4])
        for filename in sorted(ufiles):
            nfile.write(filename + '\n')

if __name__ == "__main__":
    main()