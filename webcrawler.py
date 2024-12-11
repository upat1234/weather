# Author: Patrick Underwood (upatrick@vt.edu)
# Version: 2024-12-10

import requests
from bs4 import BeautifulSoup
from crawler import Crawler
import os
import time

def download_csv_files(start, end, max_retries=3, timeout=10):
    if not os.path.isdir("weather_data"):
        os.mkdir("weather_data")
    parent_url = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/"
    crawler = Crawler()
    r_links = []

    # Choose links in year range.
    # Works for 1929 to 2024.
    i = int(start)
    while i <= int(end):
        r_links.append(parent_url + str(i) + "/")
        i += 1

    # Download and store data files for each year.
    for link in r_links:
        file_count = 0
        folder_path = os.path.join("weather_data", link[64:68] + "_weather_data")
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
            print("New FOLDER for year " + link[64:68] + " created.\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        year_page = crawler.download_url(link)
        year_links = crawler.get_links(link, year_page)
        for y_link in year_links:
            if ".csv" in y_link:
                folder = os.path.basename(y_link)[0]
                num_folder = os.path.join(folder_path, folder + "_files")
                if not os.path.isdir(num_folder):
                    os.makedirs(num_folder)
                    print("New FOLDER for code number created. Folder number: " + folder + "\n-------------------------------------------------------")
                file_path = os.path.join(num_folder, os.path.basename(y_link))
                if not os.path.isfile(file_path):
                    for attempt in range(max_retries):
                        try:
                            response = requests.get(y_link, timeout=timeout)
                            if response.status_code == 200:
                                file_count += 1
                                with open(file_path, "wb") as file:
                                    file.write(response.content)
                                print("FILE: " + str(file_count) + ", YEAR: " + link[64:68] + ", FOLDER: " + folder)
                                break
                        except (requests.ConnectionError, requests.Timeout) as e:
                            print("Attempt " + str(attempt + 1) + " failed for " + os.path.basename(y_link) + ". RETRYING.")
                            time.sleep(2)
                    else:
                        print("FAILED to download " + os.path.basename(y_link) + ".")
                else:
                    print("File: " + os.path.basename(y_link) + " was skipped.")
        print(link[64:68] + " weather data downloaded.")

if __name__ == "__main__":
    # Edit these numbers to download different years.
    start = input("Enter start year: ")
    end = input("Enter end year: ")
    download_csv_files(start, end)