import pandas as pd
import os
import requests
index_cap = 20000
image_amount = 500
hit_url_list = []
non_hit_url_list = []
label_counter = 0
non_hit_label_counter = 0
tags = open("selected_tags.txt", "r")
selected_labels = []
df_files_to_download = pd.read_csv("../all_files_to_download.txt", sep='|')
df_app_labels = pd.read_csv("app_labels.txt", sep='|')
df_app_non_labels = pd.read_csv("app_non-labels.txt", sep='|')
for index in tags:
    selected_labels.append(index.strip())

# create label/non-label-directories for each labels
for label in selected_labels:
    urls = []
    labelled_images_found = 0
    index = 0
    label_path = "../../resources/binary_training_data/train_" + label + "/" + label
    non_label_path = "../../resources/binary_training_data/train_" + label + "/" + "non-" + label
    print("-------------------------------------------------")
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    else:
        print(label + "-label-" + "directory already exist")
    if not os.path.exists(non_label_path):
        os.makedirs(non_label_path)
    else:
        print("non-" + label + "-directory already exist")
    print("Gathering images for", label + "...")

    # collect labelled images
    found_all = False
    print("Searching for images with label:", label)
    for i, app_label in df_app_labels.iterrows():
        if i > index_cap or found_all:
            break
        tags = app_label[1].replace("[", "").replace("]", "").replace("'", "").replace(" ", "").split(",")
        # check if label is found among tags, if true download all pictures for that game to the right folder
        for tag in tags:
            if label == tag:
                previous_id = 0
                while len(df_files_to_download) - 1 > index and not found_all:
                    if not previous_id == df_files_to_download.loc[index][0]:
                        if df_files_to_download.loc[index][0] > app_label[0]:
                            break
                        else:
                            if df_files_to_download.loc[index][0] == app_label[0]:
                                while df_files_to_download.loc[index][0] == app_label[0] and len(
                                        df_files_to_download) - 1 > index:
                                    if labelled_images_found % 100 == 0 and labelled_images_found > 0:
                                        print("Found", labelled_images_found, "labelled images")
                                    if labelled_images_found == image_amount:
                                        found_all = True
                                        break
                                    labelled_images_found += 1
                                    urls.append(df_files_to_download.loc[index][1] + "|" + str(
                                        df_files_to_download.loc[index][0]))
                                    index += 1
                    else:
                        previous_id = df_files_to_download.loc[index][0]
                    index += 1
    hit_url_list.append(urls)

    # check if directory is empty
    num = 0
    if len(os.listdir(label_path)) == 0:
        for url in hit_url_list[label_counter]:
            link = url.split("|")[0]
            app_id = url.split("|")[1]
            file = open(label_path + "/" + label + app_id + "_" + str(num) + ".jpg", "wb")
            response = requests.get(link)
            file.write(response.content)
            file.close()
            num += 1
            if num % 100 == 0 and num > 0:
                print("Downloaded {}/{} so far".format(num, labelled_images_found))
    else:
        print("Directory is not empty")

    # collect non-label images
    index = 0
    found_all = False
    non_labelled_images_found = 0
    skip = False
    urls = []
    print("Searching for images with label:", "non-" + label)
    for i, app_label in df_app_non_labels.iterrows():
        skip = False
        if i > index_cap or found_all:
            break
        tags = app_label[1].replace("[", "").replace("]", "").replace("'", "").replace(" ", "").split(",")
        # check if label is found among tags, if true download all pictures for that game to the right folder
        for tag in tags:
            if (label == tag):
                skip = True
                break
        if not skip:
            previous_id = 0
            while len(df_files_to_download) - 1 > index and not found_all:
                if not previous_id == df_files_to_download.loc[index][0]:
                    if df_files_to_download.loc[index][0] > app_label[0]:
                        break
                    else:
                        if df_files_to_download.loc[index][0] == app_label[0]:
                            while df_files_to_download.loc[index][0] == app_label[0] and len(
                                    df_files_to_download) - 1 > index:
                                if non_labelled_images_found % 100 == 0 and non_labelled_images_found > 0:
                                    print("Found", non_labelled_images_found, "labelled images")
                                if non_labelled_images_found == labelled_images_found:
                                    found_all = True
                                    break
                                non_labelled_images_found += 1
                                urls.append(str(df_files_to_download.loc[index][1]) + "|" + str(
                                    df_files_to_download.loc[index][0]))
                                index += 1
                else:
                    previous_id = df_files_to_download.loc[index][0]
                index += 1
    non_hit_url_list.append(urls)

    # check if directory is empty
    num = 0
    if len(os.listdir(non_label_path)) == 0:
        for url in non_hit_url_list[label_counter]:
            link = url.split("|")[0]
            app_id = url.split("|")[1]
            file = open(non_label_path + "/" + "non-" + label + app_id + "_" + str(num) + ".jpg", "wb")
            response = requests.get(link)
            file.write(response.content)
            file.close()
            num += 1
            if num % 100 == 0 and num > 0:
                print("Downloaded {}/{} so far".format(num, labelled_images_found))
    else:
        print("Directory is not empty")
    label_counter += 1