import os

stats = {"busy" : {}, "free" : {}}

for root, dirs, files in os.walk("./results"):
    for name in files:
        status, i, emotion = name.split("_")
        emotion = emotion.rstrip(".png")
        stats[status][emotion] = stats[status].get(emotion, 0) + 1

        