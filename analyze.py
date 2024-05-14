import json
import math
import os
import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np
import pandas as pd
import seaborn
from scipy import stats
import pickle
from scipy.stats import percentileofscore
from upsetplot import UpSet
import matplotlib_venn as venn

def main():
    path = selectSnapshot()
    selectModule(path)

def selectSnapshot():
    snapshots = {i: snapshot for i, snapshot in enumerate(os.listdir('./index')) if 'reviews' in snapshot}

    if not len(snapshots.items()):
        print('No snapshots found...')
        return

    print("==========================\n= Analyze file           =\n= Please select snapshot =\n==========================")

    for i, snapshot in snapshots.items():
        print(f'{i} : {snapshot}')

    choice = int(input())
    
    if choice not in snapshots.keys():
        print('Invalid choice! try again...')
        return selectSnapshot()

    path = f'index/{snapshots[choice]}'

    return path

def selectModule(path):
    modes = ["Static time window", "Aggregated time windows", "Snapshot stats",
            "Extract strictly multi", "Plot review length",
            "Extract common reviews",
            "Extract specific comment", "Written ratio", "coAuthor",
            "User lookup", "Flag Merge", "Spam detection",
            "Review Distribution - Histogram", "Review Distribution - Boxplot",
            "Review Distribution - Violin",
            "Review Distribution - COA Histogram",
            "Review Distribution - ATW Histogram",
            "Review Distribution - ATW + COA Stacked Histogram",
            "Cluster Visualization - COA", "Cluster Visualization - ATW",
            "Review delta distribution",
            "Extension overlap visualization - Venn",
            "Extension overlap visualization - UpSet",
            "Review spikes"
            ]

    print("==========================\n= Analyze file           =\n= Please enter format    =\n==========================")

    for i, mode in enumerate(modes):
        print(f"{i} : {mode}")

    if not os.path.exists("out"):
        os.mkdir("out")
    
    if not os.path.exists(f"out/{path[6:]}"):
        os.mkdir(f"out/{path[6:]}")

    format = int(input())
    loadPlotConfig(0)

    match format:
        case 0:
            minutes = input("Enter the wanted time window size in minutes (Comma separated if several): ")
            if len(minutes.split(",")) == 1:
                staticTimeWindow(path, float(minutes), save=True)
            else:
                for time in minutes.split(","):
                    staticTimeWindow(path, float(time), save=True)
                    print(render_loading(f"Static window analysis - {float(time)}", 1, 1))
        case 1:
            minutes = input("Enter the wanted time window size in minutes (Comma separated if several): ")
            if len(minutes.split(",")) == 1:
                timeWindwos = staticTimeWindow(path, float(minutes) / 10)
                aggregatedTimeWindow(timeWindwos, float(minutes), path)
            else:
                for time in minutes.split(","):
                    timeWindwos = staticTimeWindow(path, float(time) / 10)
                    aggregatedTimeWindow(timeWindwos, float(time), path)
                    print(render_loading(f"Aggregated window analysis - {float(time)}", 1, 1))

        case 2: 
            snapshotStats(path)

        case 3:
            extractStrictlyMulti(path)

        case 4:
            plotReviewLength(path)

        case 5:
            maxLength = int(input("Enter the max length of a given message (0 = no limit): "))
            extractCommonReviews(path, maxLength)

        case 6:
            message = input("Enter message: ").lower()
            extractSpecificComment(path, message)

        case 7:
            writtenRatio(path)

        case 8:
            coAuthor(path)

        case 9:
            users = input("Enter user/users (Comma separated if several): ")
            userLookup(path, users)

        case 10:
            flagMerge(path)
        
        case 11:
            threshold = input('Enter threshold in minutes: ')
            specificExtension = input('Enter specific extension to plot over time (Empty if none): ')
            spamDetection(path, threshold, specificExtension)

        case 12:
            reviewDistributionHistogram(path)

        case 13:
            reviewDistributionBoxplot(path)

        case 14: 
            reviewDistributionViolin(path)

        case 15:
            reviewDistributionCOA(path)

        case 16:
            reviewDistributionATW(path)

        case 17:
            reviewDistributionStacked(path)

        case 18:
            ClusterVisualizationCOA(path)

        case 19:
            instances = input('Enter the wanted thresholds (Comma separated if several): ')

            for instance in instances.split(','):
                if not os.path.isfile(f'out/{path.split("/")[1]}/atw_{instance}'):
                    print(f'Missing file: out/{path.split("/")[1]}/atw_{instance}, please produce prerequisites...')
                    break
            else:
                ClusterVisualizationATW(path, instances.split(','))
            
        case 20:
            reviewDeltaDistribution(path)

        case 21:
            extensionOverlapVisualizationVenn(path)

        case 22:
            extensionOverlapVisualizationUpSet(path)

        case 23:
            reviewSpikes(path)

        case _:
            print("Invalid method! Please try again...")

def staticTimeWindow(path, minutes, save=False):
    users, _ = loadData(path)
    res = {}
    timeWindow = int(1000 * 60 * minutes)

    print(render_loading("Timebased clustering", 0,1), end="\r")
    for uid, reviews in users.items():
        for eid, review in reviews.items():
            current = res.setdefault(review[3] - (review[3] % timeWindow), {})
            currentEid = current.setdefault(eid, {})
            currentEid[uid] = review
    print(render_loading("Timebased clustering", 1,1))

    res = sorted([[key, value] for key, value in res.items() if len(value.items()) > 0], key=lambda x: -len(x[1].items()))

    if save:
        output = []

        for day, extension in res:
            exList = [[key, value] for key, value in extension.items() if len(value.items()) > 4]
            if len(exList) > 3:
                output.append(f"Time window: {day} Total extensions: {len(extension.keys())}")
                for eid, rev in exList:
                    output.append(f"\t{eid} - {len(rev.values())}")

        data = '\n'.join(output)
        saveFile(path, f'stw_{minutes}', data)
    
    return {key: value for key, value in res}

def aggregatedTimeWindow(timeWindows, minutes, path):
    _, extensions = loadData(path)
    res = {}
    count = 0
    gap = int((minutes / 10) * 60 * 1000)
    start = int(sorted(timeWindows.keys())[0] - (10 * gap))
    end = sorted(timeWindows.keys())[0]
    final = sorted(timeWindows.keys())[-1]
    window = {}
    totalCount = int((final - start) / gap)

    extensionInfo = loadExtensionInfo("index/extensionInfo")

    print(render_loading("Aggregated clustering", 0, totalCount), end="\r")
    while start <= final:
        for eid in window.keys() & timeWindows.setdefault(end, {}).keys():
            window[eid].update(timeWindows[end][eid])

        for eid in timeWindows.setdefault(end, {}).keys() - window.keys():
            window[eid] = timeWindows[end][eid]

        for eid in timeWindows.setdefault(start, {}).keys():
            if eid in window.keys() and len(window[eid]) == len(timeWindows[start][eid]):
                del window[eid]

            if eid in window.keys():
                window[eid] = {uid: review for uid, review in window[eid].items() if uid not in timeWindows[start][eid].keys()}

        for eid, reviews in timeWindows.setdefault(start + (5 * gap), {}).items():
            for oppEid, oppReviews in window.items():
                if oppEid == eid:
                    continue

                extReviews = [[uid, review] for uid, review in reviews.items()]
                oppExtReviews = [[uid, review] for uid, review in oppReviews.items()]
                inCommon = min(len(extReviews), len(oppExtReviews))

                res.setdefault(eid, {}).setdefault(oppEid, {})["count"] = res.setdefault(eid, {}).setdefault(oppEid, {}).setdefault("count", 0) + inCommon
                res.setdefault(eid, {}).setdefault(oppEid, {}).setdefault("pairs", []).append([extReviews, oppExtReviews])

        count += 1
        if not (count % 11): 
            print(render_loading("Aggregated clustering", count, totalCount), end="\r")
                        
        start += gap
        end += gap 
    print(render_loading("Aggregated clustering", totalCount, totalCount))

    print(render_loading("Filtering", 0, 1), end="\r")
    res = {eid: {oppEid: reviews for oppEid, reviews in oppExt.items() if not (reviews["count"] < 0.4 * max(len(extensions[oppEid]), len(extensions[eid])) or reviews["count"] <= 3)} for eid, oppExt in res.items()}

    res = {eid: oppExt for eid, oppExt in res.items() if len(oppExt)}

    for eid, oppExt in res.items():
        res[eid] = sorted([[oppEid, reviews["count"], reviews["pairs"]] for oppEid, reviews in oppExt.items()], key=lambda x: -(x[1] / len(extensions[oppEid].items())))

    res = sorted([[eid, oppExt] for eid, oppExt in res.items()], key=lambda x: -(x[1][0][1] / len(extensions[eid].items())))   
    print(render_loading("Filtering", 1, 1))

    print(render_loading("Discrete optimization", 0, len(res)), end="\r")
    count = 0
    for eid, oppExt in res:
        for index, [oppEid, totalInCommon, pairs] in enumerate(oppExt):
            rev = {}
            connections = []
            currentRev = 0
            for extReviews, oppExtReviews in pairs:
                for uid, review in extReviews:
                    if f"{eid}{uid}{review[3]}" not in rev.keys():
                        rev[f"{eid}{uid}{review[3]}"] = currentRev 
                        currentRev += 1

                    for oppUid, oppReview in oppExtReviews:
                        if f"{oppEid}{oppUid}{oppReview[3]}" not in rev.keys():
                            rev[f"{oppEid}{oppUid}{oppReview[3]}"] = currentRev 
                            currentRev += 1

                        connections.append([rev[f"{eid}{uid}{review[3]}"], rev[f"{oppEid}{oppUid}{oppReview[3]}"]])

            graph = np.array([[1 if k == i or k == j else 0 for k in range(currentRev)] for i, j in connections]).T

            X = cp.Variable(len(connections))
            constraints = [graph @ X.T <= np.ones(currentRev).T, X >= 0]
            objective = cp.Maximize(np.ones(len(connections)).T @ X)
            problem = cp.Problem(objective, constraints)
            solution = problem.solve()
            oppExt[index][1] = int(solution)
        count += 1
        print(render_loading("Discrete optimization", count, len(res)), end="\r")
    print(render_loading("Discrete optimization", len(res), len(res)))

    print(render_loading("Filtering", 0, 1), end="\r")
    res = [[eid, list(filter(lambda x: x[1] > (0.1 * len(extensions[x[0]].items())) and x[1] > 2, oppExt))] for [eid, oppExt] in res]
    res = list(filter(lambda x: len(x[1]), res))

    for i, [eid, oppExt] in enumerate(res):
        #score = sum([(matched * ((matched / len(extensions[oppEid].items())) ** 2)) for [oppEid, matched, _] in oppExt])
        score = sum([1 for [oppEid, matched, _] in oppExt])
        res[i] = [eid, sorted(sorted(oppExt, key=lambda x: -x[1]), key=lambda x: -(x[1] / len(extensions[x[0]].items()))), score]

    res = sorted(res, key=lambda x: -(x[2]))
    print(render_loading("Filtering", 1, 1))

    data = ''
    shown = set()
    for eid, oppExt, score in res:
        if len(list(filter(lambda x: x in shown ,[eidOpp for eidOpp, _, _ in oppExt] + [eid]))) > (0.5 * (len(oppExt) + 1)):
            continue

        shown.add(eid)
        data += f"{eid},{extensionInfo[eid][0]},{len(extensions[eid].items())},{score}\n"
        for oppEid, totalInCommon, pairs in oppExt:
            shown.add(oppEid)
            data += f"\t{oppEid},{extensionInfo[oppEid][0]},{len(extensions[oppEid].items())},{totalInCommon}\n"

    saveFile(path, f'atw_{int(minutes)}', data)

def snapshotStats(path):
    users, extensions = loadData(path)

    print(render_loading(f"Analyze", 0, 1), end="\r")

    totalReviews = sum([len(reviews.items()) for reviews in extensions.values()])
    totalExtensions = len(extensions.items())

    totalUsers = len(users.items())
    singleUsers = len([user for user, reviews in users.items() if len(reviews.items()) == 1])
    multiUsers = totalUsers - singleUsers
    multiReviews = totalReviews - singleUsers

    multiReviewUsers = {user: reviews for user, reviews in users.items() if len(reviews.items()) > 1}
    singleReviewUsers = [[user, reviews] for user, reviews in users.items() if len(reviews.items()) == 1]

    singlecoveredExtensions = set()
    multiCoveredExtensions = set()

    for _, reviews in multiReviewUsers.items():
        for eid in reviews.keys():
            multiCoveredExtensions.add(eid)

    for _, reviews in singleReviewUsers:
        for eid in reviews.keys():
            singlecoveredExtensions.add(eid)

    reviewLengthAll = sum([len(review[2]) for _, reviews in users.items() for _, review in reviews.items()]) 
    reviewLengthSingle = sum([len(review[2]) for _, reviews in singleReviewUsers for _, review in reviews.items()]) 
    reviewLengthMulti = sum([len(review[2]) for _, reviews in multiReviewUsers.items() for _, review in reviews.items()]) 

    multiReviewUsersShort = {user: reviews for user, reviews in users.items() if len(reviews.items()) > 1}
    delList = set()

    for user, reviews in multiReviewUsersShort.items():
        for _, review in reviews.items():
            if len(review[2]) > 85:
                delList.add(user)

    for user in delList:
        del multiReviewUsersShort[user]  

    multiGaps = [0, 0]
    multiShortGaps = [0, 0]

    for _, reviews in multiReviewUsers.items():
        last = 0
        for _, review in sorted([[key, value] for key, value in reviews.items()], key=lambda x: x[1][3]):
            if last != 0:
                multiGaps[0] += review[3] - last
                multiGaps[1] += 1
            last = review[3]

    for _, reviews in multiReviewUsersShort.items():
        last = 0
        for _, review in sorted([[key, value] for key, value in reviews.items()], key=lambda x: x[1][3]):
            if last != 0:
                multiShortGaps[0] += review[3] - last
                multiShortGaps[1] += 1
            last = review[3]

    multiGaps = multiGaps[0] / multiGaps[1]
    multiShortGaps = multiShortGaps[0] / multiShortGaps[1]

    stats = [
        ["Total extensions: " , totalExtensions],
        ["Total Users: " , totalUsers],
        ["Total Reviews: " , totalReviews],

        ["", ""],

        ["Average review length (all): ", "{:.2f}".format(reviewLengthAll / totalUsers)],
        ["Average review length (single): " , "{:.2f}".format(reviewLengthSingle / singleUsers)],
        ["Average review length (multi): ", "{:.2f}".format(reviewLengthMulti / multiUsers)],

        ["", ""],

        ["Users with a single review: " , singleUsers],
        ["Single review user of total users rate: " , "{:.2f}%".format(singleUsers / totalUsers * 100)],
        ["Single reviews: " , singleUsers],
        ["Single review of total reviews: " , "{:.2f}%".format(singleUsers / totalReviews * 100)],
        ["Extensions covered by single reviews: " , len(singlecoveredExtensions)],
        ["Extensions covered by single reviews of total extensions: " , "{:.2f}%".format(len(singlecoveredExtensions) / totalExtensions * 100)],

        ["", ""],
        
        ["Users with multi reviews: " , totalUsers - singleUsers],
        ["Users with multi reviews excluding long: " , len(multiReviewUsersShort.items())],
        ["Multi review user of total users rate: " , "{:.2f}%".format(multiUsers / totalUsers * 100)],
        ["Multi reviews: " , multiReviews],
        ["Multi review of total reviews: " , "{:.2f}%".format(multiReviews / totalReviews * 100)],
        ["Extensions covered by multi reviews: " , len(multiCoveredExtensions)],
        ["Extensions covered by multi reviews of total extensions: " , "{:.2f}%".format(len(multiCoveredExtensions) / totalExtensions * 100)],
        ["Average reviews of multi review users: " , "{:.2f}".format(multiReviews / len(multiReviewUsers.items()))], 
        ["Median reviews of multi review users: " , len(sorted(multiReviewUsers.items(), key=lambda x: len(x[1].items()))[int(len(multiReviewUsers.items()) / 2)][1])],

        ["", ""],

        ["Average gap: ", "{:.2f}".format(multiGaps / (60 * 60 * 1000))],
        ["Average gap long excluded: ", "{:.2f}".format(multiShortGaps / (60 * 60 * 1000))],

    ]
    print(render_loading(f"Analyze", 1, 1))

    data = '\n'.join([' '.join([stat, str(value)]) for stat, value in stats])

    saveFile(path, 'snapshot_stats', data, '.txt')

def extractStrictlyMulti(path):
    users, extensions = loadData(path)

    singleReviewUsers = [[user, reviews] for user, reviews in users.items() if len(reviews.items()) == 1]
    singlecoveredExtensions = set()

    for _, reviews in singleReviewUsers:
            for eid in reviews.keys():
                singlecoveredExtensions.add(eid)

    strictExtensions = {eid : reviews for eid, reviews in extensions.items() if eid not in singlecoveredExtensions}

    data = '\n'.join([eid for eid, _ in strictExtensions.items()])

    saveFile(path, 'strict_multi', data)

def plotReviewLength(path):
    users, _ = loadData(path)
    results = [{}, {}, {}]

    multiReviewUsers = {user : reviews for user, reviews in users.items() if len(reviews.items()) > 1}
    singleReviewUsers = {user : reviews for user, reviews in users.items() if len(reviews.items()) == 1}

    for _, reviews in users.items():
        for _, review in reviews.items():
            results[0][len(review[2])] = results[0].setdefault(len(review[2]), 0) + 1

    for _, reviews in singleReviewUsers.items():
        for _, review in reviews.items():
            results[1][len(review[2])] = results[1].setdefault(len(review[2]), 0) + 1

    for _, reviews in multiReviewUsers.items():
        for _, review in reviews.items():
            results[2][len(review[2])] = results[2].setdefault(len(review[2]), 0) + 1

    for i, ele in enumerate(results):
        results[i] = {key: value for key, value in sorted([[key, value] for key, value in ele.items()], key= lambda x: x[0])}

    sums = [sum(results[0].values()), sum(results[1].values()), sum(results[2].values())]

    fig, axs = plt.subplots()
    axs.plot(results[0].keys(), [val / sums[0] for val in results[0].values()], "tab:blue")
    axs.plot(results[1].keys(), [val / sums[1] for val in results[1].values()], "tab:red")
    axs.plot(results[2].keys(), [val / sums[2] for val in results[2].values()], "tab:green")
    axs.set_title("Combined")

    seaborn.despine()
    fig.tight_layout()

    axs.set_ylabel('Frequency')
    axs.set_xlabel('Length')

    plt.setp(axs, xlim=(0, 15))
    savePlot(path, 'rev_len_15')

    plt.setp(axs, xlim=(0, 25))
    savePlot(path, 'rev_len_25')

    plt.setp(axs, xlim=(0, 50))
    savePlot(path, 'rev_len_50')

    plt.setp(axs, xlim=(0, 250))
    savePlot(path, 'rev_len_250')

    plt.setp(axs, xlim=(0, 500))
    savePlot(path, 'rev_len_500')

    plt.setp(axs, xlim=(0, 1000))
    savePlot(path, 'rev_len_1000')

def extractCommonReviews(path, maxLength):
    users, _ = loadData(path)
    message = {}

    for _, reviews in users.items():
        for _, review in reviews.items():
            msg = review[2].lower()
            message[msg] = message.setdefault(msg, 0) + 1

    message = sorted([[msg, count] for msg, count in message.items() if count > 1 and (len(msg) > maxLength or not maxLength)], key=lambda x: -x[1]) 

    data = '\n'.join([f'{msg} - {count}' for msg, count in message])

    saveFile(path, f'common_messages_{maxLength}', data)

def extractSpecificComment(path, message):
    users, _ = loadData(path)
    result = []

    for _, reviews in users.items():
        for eid, review in reviews.items():
            if review[2].lower() == message:
                result.append([eid, review])

    extensionInfo = loadExtensionInfo("index/extensionInfo")

    data = f'{message}\n' + '\n'.join([f'{eid} ({extensionInfo[eid][0]}) {review[0]} {review[1]} {review[3]}' for eid, review in result])
    saveFile(path, f'extracted_message_{"".join(filter(str.isalnum, message))[:10]}', data)

def writtenRatio(path):
    _, extensions = loadData(path)
    extensionInfo = loadExtensionInfo("index/extensionInfo")
    results = sorted(sorted([[eid, extensionInfo[eid][0], extensionInfo[eid][2], len(reviews.items()), (len(reviews.items()) / (extensionInfo[eid][2] if extensionInfo[eid][2] else 1)) * 100] for eid, reviews in extensions.items() if eid in extensionInfo.keys()], key=lambda x: -x[2]), key=lambda x: -x[4])

    results = [result for result in results if result[-1] <= 100]

    #totalWrittenRatio = sum([result[3] for result in results]) / sum([result[2] for result in results])
    runningResult = results[:]
    runningResStored = {}

    thresholds = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 100, 250, 500, 1000]
    thresholds2 = [0, 25, 100, 1000]
    thresholds3 = [num for num in range(0, 5001, 1)]

    for min in thresholds3:
        runningResult = [result for result in runningResult if result[2] > min]
        runningResStored[min] = [result[4] for result in runningResult]

    results = [result for result in results if result[-1] >= 90 and result[2] > 5]

    expanded_data = {"x_axis": [], "y_axis": []}

    for category, values in [[key, runningResStored[key]] for key in thresholds]:
        for value in values:
            expanded_data["x_axis"].append(category)
            expanded_data["y_axis"].append(value)

    violinData = pd.DataFrame(expanded_data)

    ax = seaborn.violinplot(data=violinData, x="x_axis", y="y_axis", cut=0, scale='width', inner=None, linewidth=1)
    ax.set_ylabel('Ratio')
    ax.set_xlabel('Minimum Number of Reviews')
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position("left")
    plt.xticks(rotation=90)
    plt.yticks([0, 50, 100])
    seaborn.despine()
    plt.tight_layout()
    savePlot(path, 'written_ratio_violin_vertical')
    saveDump(path, 'written_ratio_violin', violinData)

    plt.figure()
    ax = seaborn.violinplot(data=violinData, x="x_axis", y="y_axis", cut=0, scale='width', inner=None, linewidth=1)
    ax.set_ylabel('Ratio', labelpad=10)
    ax.set_xlabel('Minimum Number of Reviews')
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position("left")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.xaxis.set_ticks_position('top')
    plt.xticks(rotation=-90)
    plt.yticks([0, 50, 100])
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.yaxis.get_label().set_rotation(-90)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    savePlot(path, 'written_ratio_violin_horizontal')

    data = '\n'.join([f"\t{eid},{name},{total},{written},{ratio},{stats.percentileofscore(runningResStored[math.floor((total % 5000) - (total % 5))],ratio)}" for eid, name, total, written, ratio in results])

    saveFile(path, 'written_ratio', data)

def coAuthor(path):
    users, extensions = loadData(path)
    extensionInfo = loadExtensionInfo("index/extensionInfo")
    result = {}

    count = 0
    for _, reviews in extensions.items():
        for uid, _ in reviews.items():
            for oppUid, _ in reviews.items():
                if oppUid == uid:
                    continue
                result.setdefault(uid, {})[oppUid] = result.setdefault(uid, {}).setdefault(oppUid, 0) + 1
        count += 1
        print(render_loading("Matching", count, len(extensions.items())), end="\r")    
    print(render_loading("Matching", count, len(extensions.items())))

    print(render_loading("Filtering & Sorting", 0, 1), end="\r")
    result = [[key, value] for key, value in result.items()]
    result = [[uid, sorted(list(filter(lambda x: x[1] > 2, opp.items())), key= lambda x: -(x[1] / len(users[uid].items())))] for uid, opp in result]
    result = list(filter(lambda x: len(x[1]) > 1 and len(users[x[0]].items()) > 2, result))

    for i, [uid, opp] in enumerate(result):
        extensionsFound = {}

        for eid in users[uid].keys():
                extensionsFound[eid] = extensionsFound.setdefault(eid, 0) + 1

        for oppUid, overlap in opp:
            for eid in users[oppUid].keys():
                    extensionsFound[eid] = extensionsFound.setdefault(eid, 0) + 1

        extensionsFound = dict(filter(lambda x: x[1] > 1 and x[1] / (len(opp) + 1) > 0.25, extensionsFound.items()))
        result[i].append(extensionsFound)

    result = list(filter(lambda x: len(x[2]), result))   
    result = sorted(result, key=lambda x: -(sum([overlap * ((overlap / len(users[x[0]].items()))**2) for _, overlap in x[1]]))/len(x[2]))
    print(render_loading("Filtering & Sorting", 1, 1))

    data = ''
    shown = set()
    for uid, opp, extensionsFound in result:
        if len(list(filter(lambda x: x in shown, extensionsFound.keys()))) > 0.5 * len(extensionsFound.items()):
            continue 

        if not len(opp) > 1:
            continue

        validExtensions = sorted([[key, value] for key, value in extensionsFound.items()], key= lambda x: -x[1])

        validExtensions = list(filter(lambda x: x[1] / len(extensions[x[0]]) > 0.1, validExtensions))

        if len(validExtensions) <= 1:
            continue

        data += (f"{uid},{len(users[uid])}\n")
        for oppUid, overlap in opp:
            data += (f"\t{oppUid},{len(users[oppUid])},{overlap},{'{:.2f}'.format(overlap / len(users[uid]))}\n")

        data += (f"\tTotal extensions in cluster: {len(extensionsFound)}\n")
        for eid, overlap in sorted(validExtensions, key= lambda x: -x[1]):
            if overlap > 1:
                data += (f"\t\t{eid},{extensionInfo[eid][0]},{overlap},{overlap / (len(opp) + 1)}\n")

        for eid, overlap in extensionsFound.items():
            shown.add(eid)

    saveFile(path, 'coAuthor', data)

def userLookup(path, lookupList):
    users, _ = loadData(path)
    extensionInfo = loadExtensionInfo("index/extensionInfo")

    if "," in lookupList:
        lookupList = lookupList.split(",") 
    else:
        lookupList = [lookupList]

    data = ''
    for uid in lookupList:
        data += (f"{uid}\n")
        if uid in users.keys():
            for eid, review in users[uid].items():
                data += (f"\t{eid},{extensionInfo[eid][0]},{review[1]},{review[2]},{review[3]}\n")

    saveFile(path, 'user_lookup', data)

def flagMerge(path): 
    extensionInfo = loadExtensionInfo("index/extensionInfo")

    config =    ['out/' + path[6:] + '/atw_60', 
                'out/' + path[6:][:-6] + 'multiLookup/coAuthor',
                'out/' + path[6:] + '/written_ratio',
                'out/' + path[6:] + '/spam_detection_60'
                ]

    for file in config:
        if not os.path.isfile(file):
            print(f'Missing file: {file}, please produce prerequisites...')
            return

    atwFlagged = loadFlaggedExtensions(config[0])
    coaFlagged = loadFlaggedExtensions(config[1])
    writtenFlagged = loadFlaggedExtensions(config[2])
    spamFlagged = loadFlaggedExtensions(config[3])

    allFlags = atwFlagged.union(coaFlagged, writtenFlagged, spamFlagged)
    
    result = {eid : [[1 if eid in atwFlagged else 0, 1 if eid in coaFlagged else 0, 1 if eid in writtenFlagged else 0, 1 if eid in spamFlagged else 0], extensionInfo[eid][2]] for eid in allFlags}
    result = dict(sorted(result.items(), key=lambda x: -x[1][1]))
    result = sorted(result.items(), key=lambda x: -sum(x[1][0]))

    data = ''
    data += (f'Total flags per method,{len(atwFlagged)},{len(coaFlagged)},{len(writtenFlagged)},{len(spamFlagged)}\n')
    data += (f'ATW overlap,{len(atwFlagged)},{len(atwFlagged.intersection(coaFlagged))},{len(atwFlagged.intersection(writtenFlagged))},{len(atwFlagged.intersection(spamFlagged))},New:{len(atwFlagged) - len(coaFlagged.union(writtenFlagged, spamFlagged).intersection(atwFlagged))}\n')
    data += (f'COA overlap,{len(coaFlagged.intersection(atwFlagged))},{len(coaFlagged)},{len(coaFlagged.intersection(writtenFlagged))},{len(coaFlagged.intersection(spamFlagged))},New:{len(coaFlagged) - len(atwFlagged.union(writtenFlagged, spamFlagged).intersection(coaFlagged))}\n')
    data += (f'Written overlap,{len(writtenFlagged.intersection(atwFlagged))},{len(writtenFlagged.intersection(coaFlagged))},{len(writtenFlagged)},{len(writtenFlagged.intersection(spamFlagged))},New:{len(writtenFlagged) - len(atwFlagged.union(coaFlagged, spamFlagged).intersection(writtenFlagged))}\n')
    data += (f'Spam overlap,{len(spamFlagged.intersection(atwFlagged))},{len(spamFlagged.intersection(coaFlagged))},{len(spamFlagged.intersection(writtenFlagged))},{len(spamFlagged)},New:{len(spamFlagged) - len(atwFlagged.union(coaFlagged, writtenFlagged).intersection(spamFlagged))}\n')
    for eid, [flags, reviews] in result:
        data += (f"{'X' if flags[0] else ' '},{'X' if flags[1] else ' '},{'X' if flags[2] else ' '},{'X' if flags[3] else ' '},{reviews},{eid},{extensionInfo[eid][0]}\n")

    saveFile(path, 'flag_merge', data)

def spamDetection(path, inThreshold, specific):
    _, extensions = loadData(path)
    extensionInfo = loadExtensionInfo("index/extensionInfo")
    result = {}
    spamRating = {}
    nonSpamRating = {}
    timeline = {}
    spamRatings = []

    threshold = 1000 * 60 * int(inThreshold)
    month = 1000 * 60 * 60 * 24 * 30

    for eid, reviews in extensions.items():
        sortedRevs = sorted(reviews.values(), key=lambda x: x[3])
        last = 0
        for review in sortedRevs:
            if last == 0:
                last = int(review[3])
                continue
            if (int(review[3]) - last) < threshold:
                last = int(review[3])
                result[eid] = result.setdefault(eid, 0) + 1
                if review[4]:
                    spamRating[eid] = [spamRating.setdefault(eid, [0, 0])[0] + review[4], spamRating.setdefault(eid, [0, 0])[1] + 1]
                    spamRatings.append(review[4])
                continue

            if review[4]:
                nonSpamRating[eid] = [nonSpamRating.setdefault(eid, [0, 0])[0] + review[4], nonSpamRating.setdefault(eid, [0, 0])[1] + 1]
            last = int(review[3])

            if specific and eid == specific:
                timeline[int(review[3]) - (int(review[3]) % month)] = [result.setdefault(eid, 0), spamRating.setdefault(eid, [0, 0])[0] / (spamRating.setdefault(eid, [0, 0])[1] if spamRating.setdefault(eid, [0, 0])[1] else 1), nonSpamRating.setdefault(eid, [0, 0])[0] / (nonSpamRating.setdefault(eid, [0, 0])[1] if nonSpamRating.setdefault(eid, [0, 0])[1] else 1), spamRating.setdefault(eid, [0, 0])[1] / (spamRating.setdefault(eid, [0, 0])[1] + nonSpamRating.setdefault(eid, [0, 0])[1]), abs((spamRating.setdefault(eid, [0, 0])[0] / (spamRating.setdefault(eid, [0, 0])[1] if spamRating.setdefault(eid, [0, 0])[1] else 1)) - (nonSpamRating.setdefault(eid, [0, 0])[0] / (nonSpamRating.setdefault(eid, [0, 0])[1] if nonSpamRating.setdefault(eid, [0, 0])[1] else 1)))]

    result = sorted([[eid, count] for eid, count in result.items() if len(extensions[eid]) > 3 and count > 3], key=lambda x: -x[1])  

    plt.figure()
    plt.hist(spamRatings, bins='auto')  
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.xticks(range(1, 6))

    seaborn.despine()
    plt.tight_layout()

    savePlot(path, 'spam_distribution')

    spamRatingsDf = pd.DataFrame(spamRatings, columns=['spam_ratings'])
    saveDump(path, 'spam_distribution', spamRatingsDf)

    data = '\n'.join(f'{eid},{count},{extensionInfo[eid][1]},{spamRating.setdefault(eid, [0, 0])[0] / (spamRating.setdefault(eid, [0, 0])[1] if spamRating.setdefault(eid, [0, 0])[1] else 1)},{nonSpamRating.setdefault(eid, [0, 0])[0] / (nonSpamRating.setdefault(eid, [0, 0])[1] if nonSpamRating.setdefault(eid, [0, 0])[1] else 1)},{spamRating.setdefault(eid, [0, 0])[1] / (spamRating.setdefault(eid, [0, 0])[1] + nonSpamRating.setdefault(eid, [0, 0])[1])},{abs((spamRating.setdefault(eid, [0, 0])[0] / (spamRating.setdefault(eid, [0, 0])[1] if spamRating.setdefault(eid, [0, 0])[1] else 1)) - (nonSpamRating.setdefault(eid, [0, 0])[0] / (nonSpamRating.setdefault(eid, [0, 0])[1] if nonSpamRating.setdefault(eid, [0, 0])[1] else 1)))}' for eid, count in result)
    saveFile(path, f'spam_detection_{inThreshold}', data) 

    if specific:
        time = timeline.keys()
        spamOnlyRating = [spamRating for _, spamRating, _, _, _ in timeline.values()]
        spamExcludedRating = [spamExcludeRating for _, _, spamExcludeRating, _, _ in timeline.values()]
        #spamCount = [count for count, _, _, _, _ in timeline.values()]
        #spamOfTotalRatio = [spamRatio for _, _, _, spamRatio, _ in timeline.values()]
        #Impact = [diff for _, _, _, _, diff in timeline.values()]

        plt.figure(figsize=(12, 6))
        plt.plot(time, spamOnlyRating, label='Spam Only')
        plt.plot(time, spamExcludedRating, label='Spam Excluded')

        plt.xlabel('Time')
        plt.ylabel('Rating')
        plt.legend()

        seaborn.despine()
        plt.tight_layout()

        savePlot(path, specific)

        plotData = pd.DataFrame({
            'time': time,
            'spamOnlyRating': spamOnlyRating,
            'spamExcludedRating': spamExcludedRating
        })
        saveDump(path, specific, plotData)

def reviewDistributionHistogram(path):
    users, _ = loadData(path)

    reviewCount = {user: len(pairs.items()) for user, pairs in users.items()}

    plt.figure(figsize=(10, 5))
    seaborn.histplot(reviewCount.values(), bins=np.arange(min(reviewCount.values())-0.5, max(reviewCount.values())+1, 1), edgecolor='black', log_scale=False)
    plt.yscale('log')
    plt.xlabel('Extensions Reviewed')
    plt.ylabel('Users')

    plt.gca().legend_.remove()

    seaborn.despine()
    plt.tight_layout()

    savePlot(path, 'review_distribution_histogram')

    plotData = pd.DataFrame(list(reviewCount.items()), columns=['user', 'review_count'])
    saveDump(path, 'review_distribution_histogram', plotData)

    # print(f"Average review count: {plotData['review_count'].mean()}")
    # print(f"Median review count: {plotData['review_count'].median()}")

def reviewDistributionBoxplot(path):
    users, _ = loadData(path)

    reviewCount = {user: len(pairs.items()) for user, pairs in users.items()}

    plt.figure(figsize=(10, 5))
    seaborn.boxplot(x=list(reviewCount.values()))  
    plt.xlabel('Extensions Reviewed')
    plt.ylabel('Users')

    seaborn.despine()
    plt.tight_layout()

    savePlot(path, 'review_distribution_boxplot')

    plotData = pd.DataFrame(list(reviewCount.items()), columns=['user', 'review_count'])
    saveDump(path, 'review_distribution_boxplot', plotData)

def reviewDistributionViolin(path):
    users, _ = loadData(path)

    reviewCount = {user: len(pairs.items()) for user, pairs in users.items()}

    plt.figure(figsize=(10, 5))
    
    seaborn.violinplot(x=np.log1p(list(reviewCount.values())))  
    plt.xlabel('Extensions Reviewed (log scale)')
    plt.ylabel('Users')

    seaborn.despine()
    plt.tight_layout()

    savePlot(path, 'review_distribution_violin')

    plotData = pd.DataFrame(list(reviewCount.items()), columns=['user', 'review_count'])
    saveDump(path, 'review_distribution_violin', plotData)

def reviewDistributionCOA(path):
    users, _ = loadData(path)
    
    file = f'out/{path.split("/")[1][:-6]}multiLookup/coAuthor'
    if not os.path.isfile(file):
            print(f'Missing file: {file}, please produce prerequisites...')
            return
    
    coaUsers = loadFlaggedUsers(file)

    reviewCount = {user: len(pairs.items()) for user, pairs in users.items() if user in coaUsers}

    plt.figure(figsize=(10, 5))
    seaborn.histplot(reviewCount.values(), bins=np.arange(min(reviewCount.values())-0.5, max(reviewCount.values())+1, 1), edgecolor='black', log_scale=False)
    plt.yscale('log')
    plt.xlabel('Extensions Reviewed')
    plt.ylabel('Users')

    plt.gca().legend_.remove()

    seaborn.despine()
    plt.tight_layout()

    savePlot(path, 'review_distribution_CoA')

    plotData = pd.DataFrame(list(reviewCount.items()), columns=['user', 'review_count'])
    saveDump(path, 'review_distribution_CoA', plotData)

def reviewDistributionATW(path):
    users, _ = loadData(path)

    file = 'out/' + path.split('/')[1] + '/atw_60'
    if not os.path.isfile(file):
            print(f'Missing file: {file}, please produce prerequisites...')
            return
    
    atwUsers = loadFlaggedUsers(file)

    reviewCount = {user: len(pairs.items()) for user, pairs in users.items() if user in atwUsers}

    plt.figure(figsize=(10, 5))
    seaborn.histplot(reviewCount.values(), bins=np.arange(min(reviewCount.values())-0.5, max(reviewCount.values())+1, 1), edgecolor='black', log_scale=False)
    plt.yscale('log')
    plt.xlabel('Extensions Reviewed')
    plt.ylabel('Users')

    plt.gca().legend_.remove()

    seaborn.despine()
    plt.tight_layout()

    savePlot(path, 'review_distribution_ATW')

    plotData = pd.DataFrame(list(reviewCount.items()), columns=['user', 'review_count'])
    saveDump(path, 'review_distribution_ATW', plotData)

def reviewDistributionStacked(path):
    users, _ = loadData(path)

    file = 'out/' + path.split('/')[1][:-6] + 'multiLookup/coAuthor'
    if not os.path.isfile(file):
            print(f'Missing file: {file}, please produce prerequisites...')
            return
    
    coaUsers = loadFlaggedUsers(file)
    reviewCountCOA = {user: len(pairs.items()) for user, pairs in users.items() if user in coaUsers}

    file = 'out/' + path.split('/')[1] + '/atw_60'
    if not os.path.isfile(file):
            print(f'Missing file: {file}, please produce prerequisites...')
            return

    atwUsers = loadFlaggedUsers(file)
    reviewCountATW = {user: len(pairs.items()) for user, pairs in users.items() if user in atwUsers}
    
    data = pd.DataFrame({
        "reviewCount": list(reviewCountCOA.values()) + list(reviewCountATW.values()),
        "group": ['COA'] * len(reviewCountCOA) + ['ATW'] * len(reviewCountATW)
    })

    plt.figure(figsize=(10, 5))
    seaborn.histplot(data, x="reviewCount", hue="group", bins=np.arange(min(data.reviewCount) - 0.5, max(data.reviewCount) + 1, 1), edgecolor='black', log_scale=False, multiple='stack', palette='Dark2')

    plt.yscale('log')
    plt.xlabel('Extensions Reviewed')
    plt.ylabel('Users')

    plt.gca().legend_.remove()

    seaborn.despine()
    plt.tight_layout()

    savePlot(path, 'review_distribution_stacked')

    saveDump(path, 'review_distribution_stacked', data)

def ClusterVisualizationCOA(path):
    clusters, _ = loadClusterInfo(f'out/{path.split("/")[1][:-6]}multiLookup/coAuthor')

    plotData = pd.DataFrame(clusters, columns=['extensions', 'users'])

    fig = plt.figure(figsize=(10, 10))

    ax_scatter = plt.subplot2grid((4,4), (1,0), colspan=3, rowspan=3)
    ax_box_x = plt.subplot2grid((4,4), (0,0), colspan=3)
    ax_box_y = plt.subplot2grid((4,4), (1,3), rowspan=3)

    seaborn.scatterplot(data=plotData, x='users', y='extensions', ax=ax_scatter)
    ax_scatter.set_xscale('log')
    ax_scatter.set_yscale('log')
    ax_scatter.set_xlabel('Number of Users')
    ax_scatter.set_ylabel('Number of Extensions')

    seaborn.boxplot(x=plotData['users'], ax=ax_box_x)
    ax_box_x.axis('off')
    ax_box_x.set_xscale('log')

    seaborn.boxplot(y=plotData['extensions'], ax=ax_box_y, orient='v')
    ax_box_y.axis('off')
    ax_box_y.set_yscale('log')

    plt.tight_layout()

    savePlot(path, 'CoA_visualization')
    saveDump(path, 'CoA_visualization', plotData)

def ClusterVisualizationATW(path, instances):
    clusters, _ = [loadClusterInfo(f'out/' + path.split('/')[1] + f'/atw_{instance}')[0] for instance in instances]

    scores = [score for sublist in clusters for score in sublist]
    instance_labels = [instance for instance, sublist in zip(instances, clusters) for _ in sublist]

    plotData = pd.DataFrame({'clusterScores': scores, 'instances': instance_labels})

    fig, ax = plt.subplots(figsize=(10, 6))
    seaborn.boxplot(data=plotData, x='clusterScores', y='instances', orient='h', ax=ax)

    ax.set_xticks(range(int(min(scores)), int(max(scores)) + 1, 1))

    ax.set_xlabel('Cluster Sizes (Extensions)')
    ax.set_ylabel('Burst (Minutes)')

    seaborn.despine()
    plt.tight_layout()

    savePlot(path, 'ATW_visualization')
    saveDump(path, 'ATW_visualization', clusters)

def reviewDeltaDistribution(path):
    _, extensions = loadData(path)
    deltas = []

    count = 0
    total = len(extensions.items())
    for _, reviews in extensions.items():
        print(render_loading(f"Gather review deltas", count, total), end="\r")
        sortedRevs = sorted(reviews.values(), key=lambda x: x[3])
        last = 0
        for review in sortedRevs:
            if last == 0:
                last = int(review[3])
                continue
            else:
                value = int((int(review[3]) - last) / 1000)
                #value = int(review[3]) - last
                deltas.append(value if value else 1)
                last = int(review[3])
        count += 1
    print(render_loading(f"Gather review deltas", count, total))

    df = pd.DataFrame(deltas, columns=['Deltas'])

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": (.15, .85), "hspace": 0.3}, constrained_layout=True)

    seaborn.boxplot(x=df["Deltas"], ax=axs[0], orient='h')
    axs[0].set_xscale('log')
    axs[0].set(xlabel=None)
    axs[0].tick_params(left=False, bottom=False)
    seaborn.despine(ax=axs[0], left=True, bottom=True, right=True, top=True)

    seaborn.histplot(df["Deltas"], ax=axs[1], log_scale=True, bins=100)
    axs[1].set_xlabel("Deltas (seconds)")
    axs[1].set_ylabel("Count")
    seaborn.despine(ax=axs[1])

    xmin = min(df["Deltas"])
    xmax = max(df["Deltas"])
    axs[0].set_xlim([xmin, xmax])
    axs[1].set_xlim([xmin, xmax])

    savePlot(path, 'delta_distribution')

    # print(f'Mean of the graph: {int(df["Deltas"].mean())}')
    # print(f'Percent below one minute: {percentileofscore(df["Deltas"], 60, kind="weak")}')
    # print(f'Percent below one hour: {percentileofscore(df["Deltas"], 3600, kind="weak")}')
    # print(f'Percent below two hours: {percentileofscore(df["Deltas"], 7200, kind="weak")}')

def extensionOverlapVisualizationVenn(path):
    _, COAExtensions = loadClusterInfo(f'out/{path.split("/")[1][:-6]}multiLookup/coAuthor')
    _, ATWExtensions = loadClusterInfo(f'out/' + path.split('/')[1] + f'/atw_60')

    vennDiagram = venn.venn2([COAExtensions, ATWExtensions], set_labels = ('COA', 'ATW'))

    for idx in ['10', '01', '11']:
        patch = vennDiagram.get_patch_by_id(idx)
        if patch:
            patch.set_edgecolor('white')
            patch.set_linewidth(5)
    savePlot(path, 'extension_overlap_venn')

def extensionOverlapVisualizationUpSet(path):
    _, COAExtensions = loadClusterInfo(f'out/{path.split("/")[1][:-6]}multiLookup/coAuthor')
    _, ATWExtensions = loadClusterInfo(f'out/' + path.split('/')[1] + f'/atw_60')
    _, SpamExtensions = loadClusterInfo(f'out/' + path.split('/')[1] + f'/spam_detection_60')
    _, WrittenExtensions = loadClusterInfo(f'out/' + path.split('/')[1] + f'/written_ratio')

    all_extensions = list(COAExtensions | ATWExtensions | SpamExtensions | WrittenExtensions )
    data = []
    for ext in all_extensions:
        data.append([ext, ext in COAExtensions, ext in ATWExtensions, ext in SpamExtensions, ext in WrittenExtensions])

    df = pd.DataFrame(data, columns=['Extension', 'COA', 'ATW', 'Spam', 'Written'])
    
    df.set_index(['COA', 'ATW', 'Spam', 'Written'], inplace=True)

    upsetPlot = UpSet(df, subset_size='count', orientation='horizontal')
    upsetPlot.plot()

    savePlot(path, 'extension_overlap_upset')
    saveDump(path, 'extension_overlap_upset', df)

def reviewSpikes(path):
    _, extensions = loadData(path)

    data = {}

    for num in [len(reviews) for _, reviews in extensions.items()]:
        data[num] = data.setdefault(num, 0) + 1

    data = sorted([f'{reviewCount}: {amount}' for reviewCount, amount in data.items()], key=lambda x: -int(x.split(": ")[1]))

    saveFile(path, 'review_spikes', "\n".join(data))

def loadClusterInfo(path):
    clusters = []
    with open(path, "r") as f:
        lines = f.readlines()

    totExtensions = set()
    extensions = 0
    users = 0
    clusterCount = 0

    for line in lines:
        if line.startswith("\t"):
            id_string = line.strip().split(',')[0]
            if len(id_string) == 32:
                totExtensions.add(id_string)
                extensions += 1
            elif len(id_string) == 16:
                users += 1
        else:
            if extensions > 0:
                clusters.append([extensions, users])
            extensions = 0
            users = 0
            
            id_string = line.strip().split(',')[0]
            if len(id_string) == 32:
                totExtensions.add(id_string)
                extensions += 1
            elif len(id_string) == 16:
                users += 1
            clusterCount += 1

    if extensions > 0:
        clusters.append([extensions, users])

    #print(f'{path} has {clusterCount} clusters containing {len(totExtensions)} extensions')

    return clusters, totExtensions

def savePlot(path, name):
    print(render_loading(f"Write file - {name}.pdf", 0, 1), end="\r")
    plt.savefig(f'out/{path[6:]}/{name}.pdf', bbox_inches='tight')
    print(render_loading(f"Write file - {name}.pdf", 1, 1))

def saveFile(path, name, data, suffix = ''):
    print(render_loading(f"Write file - {name}{suffix}", 0, 1), end="\r")
    with open(f"out/{path[6:]}/{name}{suffix}", "w+", encoding="utf-8") as f:
        f.write(data)
    print(render_loading(f"Write file - {name}{suffix}", 1, 1))    

def saveDump(path, name, data):
    print(render_loading(f"Write file - {name}.pkl", 0, 1), end="\r")
    with open(f'out/{path[6:]}/{name}.pkl', 'wb') as f:
        pickle.dump(data, f)
    print(render_loading(f"Write file - {name}.pkl", 1, 1))

def loadPlotConfig(num):

   match num:
        case 0:
            seaborn.set_context("poster")
            label_size = 12
            plt.rcParams['axes.labelsize'] = label_size
            plt.rcParams['axes.titlesize'] = label_size
            plt.rcParams['xtick.labelsize'] = label_size
            plt.rcParams['ytick.labelsize'] = label_size
        case 1:
            seaborn.set_context("poster")
            label_size = 20
            plt.rcParams['axes.labelsize'] = label_size
            plt.rcParams['axes.titlesize'] = label_size
            plt.rcParams['xtick.labelsize'] = label_size
            plt.rcParams['ytick.labelsize'] = label_size
            plt.rcParams["font.weight"] = "bold"
            plt.rcParams["axes.labelweight"] = "bold"
            plt.rcParams["axes.labelweight"] = "bold"
        case 2:
            seaborn.set_theme(style="ticks")
            seaborn.color_palette("colorblind")
            seaborn.set_context("paper", font_scale=1.8)
        case 3:
            seaborn.set_context("poster")
            seaborn.set_palette("husl")
            label_size = 20
            plt.rcParams['axes.labelsize'] = label_size
            plt.rcParams['axes.titlesize'] = label_size
            plt.rcParams['xtick.labelsize'] = label_size
            plt.rcParams['ytick.labelsize'] = label_size
            plt.rcParams['grid.color'] = 'grey'
            plt.rcParams['grid.linestyle'] = '--'
            plt.rcParams['grid.linewidth'] = 0.5
            plt.rcParams['legend.fontsize'] = label_size
            plt.figure(figsize=(20, 10))
            seaborn.despine()
    

def loadFlaggedExtensions(path):
    with open(path, 'r', encoding='utf-8') as f:
        res = set([line.strip().split(",")[0] for line in f.readlines() if len(line.strip().split(",")[0]) == 32])
    return res

def loadFlaggedUsers(path):
    with open(path, 'r', encoding='utf-8') as f:
        res = set([line.strip().split(",")[0] for line in f.readlines() if len(line.strip().split(",")[0]) == 16])
    return res

def loadData(path):
    users = {}
    extensions = {}

    print(render_loading(f"Load snapshot ({len(users.items())}) ({len(extensions.items())})", 0, 2), end="\r")

    with open(f"{path}/users", "r", encoding="utf-8") as f:
        users = json.load(f)
    print(render_loading(f"Load snapshot ({len(users.items())}) ({len(extensions.items())})", 1, 2), end="\r")

    with open(f"{path}/extensions", "r", encoding="utf-8") as f:
        extensions = json.load(f)
    print(render_loading(f"Load snapshot ({len(users.items())}) ({len(extensions.items())})", 2, 2))

    return users, extensions

def loadExtensionInfo(path):
    extensionInfo = {}

    with open(f"{path}", "r", encoding="utf-8") as f:
        extensionInfo = json.load(f)
    print(render_loading(f"Load extended info", 1, 1))

    return extensionInfo

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