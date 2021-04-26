tags = open("tags.txt", "r")
labels = []
for x in tags:
  labels.append(x.strip())
print(labels)

stop = 0
counter = 0
games = open("app_labels.txt", "r")
game_list = [[False]*41]*100

# Prepare list with labels
for game in games:
  split = game.split("[")
  split = split[1].split("]")
  game_tags = split[0]
  if stop == 1: break
  stop += 1
  print(game_tags)
  game_tags = game_tags.split(",")
  print(labels[0], game_tags[0])

  print(game_list[0])
  print(game_list)


# Replace matching labels with true at given index
  print(labels[0], game_tags[0])
  for game_tag in game_tags:
    for label in labels:
      print("is", game_tag, "=", label)
      if game_tag != label:
        print("no")
        counter += 1
      else:
        print("yes")
        print("counter = ", counter)
        game_list[stop].insert(counter,True)
        counter = 0
        break
print(game_list[0])

'''
  game_tags = game_tags.split(",")
  print(labels[0], game_tags[0])
  for game_tag in game_tags:
    for label in labels:
      print("is", game_tag, "=", label)
      if game_tag != label:
        print("no")
        counter += 1
      else:
        print("yes")
        print("counter = ", counter)
        game_list[stop].append([True])
        counter = 0
        break
print(game_list)
'''

    #game_list[game.split(",").append()]
'''
tags = open("tags.txt", "r")
labels = []
for x in tags:
  labels.append(x.strip())
print(labels)

stop = 0
counter = 0
games = open("app_labels.txt", "r")
game_list = []

# Prepare list with labels
for game in games:
  split = game.split("[")
  split = split[1].split("]")
  game_tags = split[0]
  if stop == 1: break
  stop += 1
  print(game_tags)

# Replace matching labels with true at given index
  game_tags = game_tags.split(",")
  print(labels[0], game_tags[0])
  for game_tag in game_tags:
    for label in labels:
      print("is", game_tag, "=", label)
      if game_tag != label:
        print("no")
        counter += 1
      else:
        print("yes")
        print("counter = ", counter)
        game_list[stop].insert(True,counter)
        counter = 0
        break

'''