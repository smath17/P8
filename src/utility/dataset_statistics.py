import os


def genres_statistics():
    """
    This function is for the old dataset. Don't use this one.
    """

    # Path joining
    directory = 'genres/'
    genres_array = [name for name in os.listdir(directory)]
    for genre in genres_array:
        files_in_dir = len([name for name in os.listdir(directory + genre)])
        print(genre + ": " + str(files_in_dir))


def class_distribution_in_dataset():
    genre_list = {}

    with open("../image_labels.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            genres_array = line.split("|")[1]
            genres = genres_array.replace("'", "")\
                                 .replace("[", "")\
                                 .replace(" ", "")\
                                 .replace("]", "")\
                                 .replace("\n", "")\
                                 .split(",")
            for genre in genres:
                if genre not in genre_list:  # Create entry in genre_list
                    genre_list[genre] = 1
                else:                        # Increment genre in genre_list
                    genre_list[genre] += 1

    return genre_list


def multiclass_distribution_in_dataset():
    genres_list = {}

    with open("../image_labels.txt", "r") as file:
        lines = file.readlines()
        for line in lines:
            genres_array = line.split("|")[1]
            genres = genres_array.replace("'", "")\
                                 .replace("[", "")\
                                 .replace(" ", "")\
                                 .replace("]", "")\
                                 .replace("\n", "")\
                                 .split(",")

            if str(genres) not in genres_list:  # Create entry in genre_list
                genres_list[str(genres)] = 1
            else:  # Increment genre in genre_list
                genres_list[str(genres)] += 1

    return genres_list


distribution = class_distribution_in_dataset()
# print(distribution)
for label in distribution:
    print(label + "," + str(distribution[label]))
