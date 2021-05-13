import pandas as pd
from os import walk


def label_images():
    # setup columns based on tags.txt
    columns = ["appid"]
    with open("../../resources/tags.txt") as file:
        for line in file:
            columns.append(line[:-1])

    # read from steamspy_tag_data.csv using previous columns
    df_tags = pd.read_csv("../../resources/steamspy_tag_data.csv", usecols=columns)

    # write appid + tags to app_labels.txt
    file = open("../app_labels.txt", "w")

    app_labels = {}

    # iterate through csv as tuples
    for row in df_tags.head(df_tags.size).itertuples():
        index = 1
        tags = []
        output = str(row[1]) + "|"
        # iterate through columns (tags)
        while index < len(columns):
            if row[index + 1] > 0:
                tags.append(columns[index])
            index += 1
        # skip if no tags applied to appid
        if len(tags) > 0:
            app_labels[str(row[1])] = tags
            output = output + str(tags)
            file.write(output + "\n")
    file.close()

    _, _, filenames = next(walk("../../resources/all_images"))

    file = open("../image_labels.txt", "w")

    for filename in filenames:
        app_id = str(filename.split("_")[0])
        if app_labels.get(app_id) is not None:
            output = filename + "|" + str(app_labels.get(app_id))
            file.write(output + "\n")

    file.close()


def label_images_with_rest():
    # setup columns based on tags.txt
    columns = ["appid"]
    with open("../../resources/tags.txt") as file:
        for line in file:
            columns.append(line[:-1])

    # read from steamspy_tag_data.csv using previous columns
    df_tags = pd.read_csv("../../resources/steamspy_tag_data.csv", usecols=columns)

    # write appid + tags to app_labels.txt
    file = open("../app_labels_2.txt", "w")

    app_labels = {}
    high_data_list = ["adventure", "action", "simulation", "strategy", "rpg"]

    # iterate through csv as tuples
    for row in df_tags.head(df_tags.size).itertuples():
        index = 1
        tags = []
        output = str(row[1]) + "|"
        # iterate through columns (tags)
        while index < len(columns):
            if row[index + 1] > 0:
                if columns[index] in high_data_list:
                    tags.append(columns[index])
                else:
                    if "rest" not in tags:
                        tags.append("rest")
            index += 1
        # skip if no tags applied to appid
        if len(tags) > 0:
            app_labels[str(row[1])] = tags
            output = output + str(tags)
            file.write(output + "\n")
    file.close()

    _, _, filenames = next(walk("../../resources/all_images"))

    file = open("../image_labels_2.txt", "w")

    for filename in filenames:
        app_id = str(filename.split("_")[0])
        if app_labels.get(app_id) is not None:
            output = filename + "|" + str(app_labels.get(app_id))
            file.write(output + "\n")

    file.close()
