import pandas as pd
import os
import json
import requests
import ast


def download_images():
    column_names_media = ["steam_appid", "header_image", "screenshots", "background", "movies"]

    df_media = pd.read_csv("steam_media_data.csv", names=column_names_media)

    if not os.path.exists("all_images"):
        os.mkdir("all_images")

    # iterate over rows with iterrows()
    apps_prev_row = 0

    # Loop through every game in steam_media_data.csv
    for index_media, row_media in df_media.head(df_media.size).iterrows():
        # access data using column names
        if index_media == 0:
            continue

        # Convert screenshot column to a JSON string
        json_string = "[" + row_media['screenshots'][1:-2].replace("\'", "\"") + "}]"
        screenshots = json.loads(json_string)

        # Write URLs to a file to download later.
        # Thus we do not need to move the file later and we can start from that file instead.
        # TODO This file is appended to every time. Not intended.
        #  Should look if the line is already in there and append if not. Not important right now.
        with open("files_to_download.txt", "a") as file:
            for screenshot in screenshots:
                file.write(str(row_media["steam_appid"]) + "|"
                           + screenshot["path_thumbnail"] + "\n")

def download_images_from_file():
    if not os.path.exists("files_to_download.txt"):
        raise Exception("File does not exist")

    file = open("files_to_download.txt", "r")
    lines = file.readlines()

    screenshot_counter = 0
    previous_id = lines[0].split("|")[0]  # First app id

    # Iterate through all lines in the file.
    for line in lines:
        app_id = line.split("|")[0]
        app_url = line.split("|")[1]

        print("ID:" + app_id + " - URL:" + app_url)

        if line.split("|")[0] == previous_id:
            screenshot_counter += 1
        else:
            screenshot_counter = 0

        # Check if the image is already there
        if not os.path.exists("all_images/" + app_id + "_" + str(screenshot_counter) + ".jpg"):
            # Download the file
            file = open("all_images/" + app_id + "_" + str(screenshot_counter) + ".jpg", "wb")
            response = requests.get(app_url)
            file.write(response.content)
            file.close()

        # Keep track of the previous id to know when to reset the counter.
        previous_id = app_id

