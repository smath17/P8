import os


def genres_statistics():
    genres_array = ["Action", "Adventure", "Animation & Modeling", "Casual", "Early Access", "Free to Play",
                    "Gore", "Indie", "Massively Multiplayer", "Nudity", "Racing", "RPG", "Simulation", "Sports",
                    "Strategy", "Video Production", "Violent"]
    # path joining version for other paths
    DIR = 'genres/'
    genres_array = [name for name in os.listdir(DIR)]
    for genre in genres_array:
        files_in_dir = len([name for name in os.listdir(DIR + genre)])
        print(genre + ": " + str(files_in_dir))


genres_statistics()

