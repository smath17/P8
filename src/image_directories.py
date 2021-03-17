import time

import pandas as pd
import os
import json
from shutil import copyfile
import requests

def download_images():
    column_names_media = ["steam_appid", "header_image", "screenshots", "background", "movies"]

    column_names_apps = ["appid", "name", "release_date", "english",
                    "developer", "publisher", "platforms", "required_age",
                    "categories", "genres", "steamspy_tags", "achievements",
                    "positive_ratings", "negative_ratings", "average_playtime",
                    "median_playtime", "owners", "price"]

    df_media = pd.read_csv("steam_media_data.csv", names=column_names_media)
    df_apps = pd.read_csv("steam.csv", names=column_names_apps)

    if not os.path.exists("genres"):
        os.mkdir("genres")

    if not os.path.exists("test_downloads"):
        os.mkdir("test_downloads")

    # iterate over rows with iterrows()
    apps_prev_row = 0

    for index_media, row_media in df_media.head(df_media.size).iterrows():
        # access data using column names
        if index_media > 4115: # Fejl i csv ved 4118
            if index_media % 1 == 0:
                json_string = "[" + row_media['screenshots'][1:-2].replace("\'", "\"") + "}]"
                screenshots = json.loads(json_string)
                screenshot_link = screenshots[0]["path_thumbnail"]

                if not os.path.exists("test_downloads/" + row_media["steam_appid"] + ".jpg"):

                    found = False
                    for index_apps, row_apps in df_apps.tail(df_apps.size - apps_prev_row).iterrows():
                        if row_media["steam_appid"] == row_apps["appid"]:
                            print(row_media["steam_appid"], row_apps["genres"])
                            found = True

                            response = requests.get(screenshot_link)
                            file = open("test_downloads/" + row_media["steam_appid"] + ".jpg", "wb")
                            file.write(response.content)
                            file.close()

                            genres = row_apps["genres"].split(";")
                            for genre in genres:
                                if not os.path.exists("genres/" + genre):
                                    os.mkdir("genres/" + genre)
                                copyfile("test_downloads/" + row_media["steam_appid"] + ".jpg", "genres/" + genre + "/" + row_media["steam_appid"] + ".jpg")
                            break
                        else:
                            apps_prev_row += 1
                    if not found:
                        print(row_media["steam_appid"], "not found")


time_before = time.time()
download_images()
print("Time spent on loading images: " + str(time.time() - time_before))
