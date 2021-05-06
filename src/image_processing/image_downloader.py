import pandas as pd
import os
import json
import requests
from PIL import Image


def download_images():
    column_names_media = ["steam_appid", "header_image", "screenshots", "background", "movies"]

    df_media = pd.read_csv("../resources/steam_media_data.csv", names=column_names_media)

    if not os.path.exists("../resources/all_images"):
        os.mkdir("../resources/all_images")

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


def download_images_from_file(debug=False):
    if not os.path.exists("files_to_download.txt"):
        raise Exception("File does not exist")

    file = open("files_to_download.txt", "r")
    lines = file.readlines()
    size = 300, 300

    screenshot_counter = 0
    previous_id = 0

    # Iterate through all lines in the file.
    for line in lines:
        app_id = line.split("|")[0]
        app_url = line.split("|")[1]

        if debug:
            print("ID:" + app_id + " - URL:" + app_url)

        if line.split("|")[0] == previous_id:
            screenshot_counter += 1
        else:
            screenshot_counter = 0

        file_name = app_id + "_" + str(screenshot_counter)

        # Check if the image is already there
        if not os.path.exists("../resources/all_images/" + file_name + "_resized.jpg"):
            if not os.path.exists("../resources/all_images/" + file_name + ".jpg"):
                # Download the file
                file = open("../resources/all_images/" + file_name + ".jpg", "wb")
                response = requests.get(app_url)
                file.write(response.content)
                file.close()

            # Create a thumbnail of the image with specified method.
            # Image.resize does not work here.
            try:
                im = Image.open("../resources/all_images/" + file_name + ".jpg")
                im.thumbnail(size, Image.NEAREST)
                im.save("../resources/all_images/" + file_name + "_resized.jpg", "JPEG")
            except IOError as e:
                # If opening the image fails possibly due to interrupting the program,
                # the program will try and open the image next time the program is run.
                print(e)

            # Remove the downloaded image in order to conserve storage.
            os.remove("../resources/all_images/" + file_name + ".jpg")

        # Keep track of the previous id to know when to reset the counter.
        previous_id = app_id

