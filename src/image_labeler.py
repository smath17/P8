import pandas as pd


def label_images():
    # setup columns based on tags.txt
    columns = ["appid"]
    with open("tags.txt") as file:
        for line in file:
            columns.append(line[:-1])

    # read from steamspy_tag_data.csv using previous columns
    df_tags = pd.read_csv("steamspy_tag_data.csv", usecols=columns)

    # write appid + tags to labels.txt
    file = open("labels.txt", "w")

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
            output = output + str(tags)
            file.write(output + "\n")

    file.close()
