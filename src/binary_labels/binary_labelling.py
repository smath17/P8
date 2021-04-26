tags = open("tags.txt", "r")
labels = []
for x in tags:
  labels.append(x.strip())
print("All labels:")
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
  if stop == 2: break
  stop += 1
  print("This game has following tags:")
  print(game_tags)
  game_tags = game_tags.split(",")


# Replace matching labels with true at given index
  #print(labels[0], game_tags[0])
  for game_tag in game_tags:
    for label in labels:
      #print("is", game_tag, "=", label)
      if game_tag.strip() != label:
        #print("no")
        counter += 1
      else:
        #print("yes")
        #print("counter = ", counter)
        game_list[stop].insert(counter,True)
        break
  print(game_list[0])
  counter = 0
