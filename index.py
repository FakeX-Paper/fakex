import json
import sys
import os

# Index snapshot and save it locally as JSON file in desired format
def main():
    modes = ["Efficient lookups", "Extended extension info", "Strictly multi lookup"]
    
    if not len(sys.argv) or not os.path.exists(sys.argv[1]):
        return print(f"Invalid argument path...")

    if not os.path.exists("index/"):
        os.mkdir("index")

    print("==========================\n= Index to datastructure =\n= Please enter a format  =\n==========================")

    for i, mode in enumerate(modes):
        print(f"{i} : {mode}")

    format = int(input())

    match format:
        case 0:
            lookupMaps(sys.argv[1])
        case 1:
            extraExtensionMaps(sys.argv[1])
        case 2:
            multiUserLookup(sys.argv[1])
        case _:
            print("Invalid format! Please try again...")
            main()

# Index data structures containing all information from given directory, following structure shown below
# users = { id : { ExtID : [userID, userName, reviewText, reviewTimestamp], ... }, ... }
# extensions = { id : { userID : [userID, userName, reviewText, reviewTimestamp], ... }, ... }
def lookupMaps(path, ret=False):
    users = {}
    extensions = {}

    count = 0
    totalLength = len(os.listdir(path))

    for file in os.listdir(path):
        extension = json.loads(open(path + "/" + file, encoding="utf8").read().split("\n")[2])
        reviews = extension[0][1][4]

        for review in reviews:
            if review[2]: 
                username = "No name found" if len(review[2]) <= 2 else review[2][1]
                eid = review[1][0][-32:]
                uid = review[2][0]
                text = review[4]
                time = review[6]
                rating = review[3]

                current = users.setdefault(uid, {})
                current[eid] = [uid, username, text, time, rating]
                users[uid] = current

                currentExt = extensions.setdefault(eid, {})
                currentExt[uid] = [uid, username, text, time, rating]
        
        count += 1
        if not (count % 11): 
            print(render_loading("Index data", count, totalLength), end="\r")
    print(render_loading("Index data", count, totalLength))

    if ret:
        return users, extensions

    if not os.path.exists(f"index/{path}_lookup"):
        os.mkdir(f"index/{path}_lookup")
    

    print(render_loading(f"Write file - users", 0, 1), end="\r")
    with open(f"index/{path}_lookup/users", "w+", encoding="utf-8") as f:
        json.dump(users, f)
    print(render_loading(f"Write file - users", 1, 1))

    print(render_loading(f"Write file - extensions", 0, 1), end="\r")
    with open(f"index/{path}_lookup/extensions", "w+", encoding="utf-8") as f:
        json.dump(extensions, f)
    print(render_loading(f"Write file - extensions", 1, 1))

def extraExtensionMaps(path):
    extraExtensions = {}

    for line in open(path, encoding="utf-8").readlines():
        data = line.split("\t")
        extraExtensions[data[0].lower()] = [data[1].removesuffix(' - Chrome Web Store'), float(data[2]), int(data[3].strip())]

    print(render_loading(f"Write file - extensionInfo", 0, 1), end="\r")
    with open(f"index/extensionInfo", "w+", encoding="utf-8") as f:
        json.dump(extraExtensions, f)
    print(render_loading(f"Write file - extensionInfo", 1, 1))

def multiUserLookup(path):
    users, _ = lookupMaps(path, ret=True)
    extensions = {} 

    print(render_loading(f"Filtering", 0, 1), end="\r")
    singleUsers = []
    for uid, reviews in users.items():
        if len(reviews.items()) == 1:
            singleUsers.append(uid)

    for uid in singleUsers:
        del users[uid]

    for uid, reviews in users.items():
        for eid, review in reviews.items():
            extensions.setdefault(eid, {})[uid] = review
    print(render_loading(f"Filtering", 1, 1))
 
    if not os.path.exists(f"index/{path}_multiLookup"):
        os.mkdir(f"index/{path}_multiLookup")

    print(render_loading(f"Write file - users", 0, 1), end="\r")
    with open(f"index/{path}_multiLookup/users", "w+", encoding="utf-8") as f:
        json.dump(users, f)
    print(render_loading(f"Write file - users", 1, 1))

    print(render_loading(f"Write file - extensions", 0, 1), end="\r")
    with open(f"index/{path}_multiLookup/extensions", "w+", encoding="utf-8") as f:
        json.dump(extensions, f)
    print(render_loading(f"Write file - extensions", 1, 1))

# Render loading bar
def render_loading(action, current, total):
    if not total:
        return f"{action.ljust(40)} | <▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉> | {('{:.2f}'.format(100)).ljust(6)}% | {1} / {1}"
    char = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]
    parts = 0 if not total else int((current / total) * ((len(char) - 1) * 20))
    bar = ""

    for _ in range(20):
        if parts >= 7:
            bar += char[7]
            parts -= 7
        elif parts > 0:
            bar += char[parts]
            parts -= 7
        else:
            bar += char[0]

    return f"{action.ljust(40)} | <{bar}> | {('{:.2f}'.format((current / total) * 100)).ljust(6)}% | {current} / {total}"


main()