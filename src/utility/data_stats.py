import pandas as pd

# setup columns based on tags.txt
columns = ["appid"]
with open("../../resources/tags.txt") as file:
    for line in file:
        columns.append(line[:-1])

# read from steamspy_tag_data.csv using previous columns
df_tags = pd.read_csv("../../resources/steamspy_tag_data.csv", usecols=columns)

tag_amount = len(columns)-1
game_amount = len(df_tags)

max_games = game_amount  # this can be changed for debugging purposes

# set up matrix
relation_matrix = []
for y in range(tag_amount):
    inner_list = []
    for x in range(tag_amount):
        inner_list.append(0)
    relation_matrix.append(inner_list)

game = 0
# iterate through csv as tuples
for row in df_tags.head(df_tags.size).itertuples():
    if game > max_games:
        break

    if game % 1000 == 0:
        print(game, "/", game_amount)

    index_h = 0
    index_v = 0

    # print("APPID: {}".format(row[1]))

    # iterate through columns (tags)
    while index_h < tag_amount:
        tag_value_h = row[index_h + 2]
        has_tag_h = tag_value_h > 0

        while index_v < tag_amount:
            tag_value_v = row[index_v + 2]
            has_tag_v = tag_value_v > 0

            if has_tag_h and has_tag_v:
                # print("{}, {}".format(columns[index_h+1], columns[index_v+1]))
                # print("{} {}".format(columns[index_h+1], row[index_h+2]))
                relation_matrix[index_h][index_v] += 1

            index_v += 1

        index_v = 0
        index_h += 1
    game += 1

# write csv to file
file = open("../../resources/tags_correspondence.csv", "w")

# first row
file.write("-,")
col = 0
while col < tag_amount:
    file.write(columns[col+1])
    if col < tag_amount-1:
        file.write(",")
    col += 1
file.write("\n")

# the rest of the rows
row = 0
while row < tag_amount:
    # print(columns[row+1].rjust(25), relation_matrix[row])
    file.write(columns[row+1])
    file.write(",")
    col = 0
    while col < tag_amount:
        file.write(str(relation_matrix[row][col]))
        if col < tag_amount-1:
            file.write(",")
        col += 1
    file.write("\n")
    row += 1

file.close()
