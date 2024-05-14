import os

import pandas as pd
from tqdm import tqdm
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import langdetect
from datetime import datetime, date
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

def find_maximal_sets(sets):
    maximal_sets = []
    for s1 in sets:
        s1 = set(s1)
        is_maximal = True
        for s2 in sets:
            s2 = set(s2)
            if s1 != s2 and s1.issubset(s2):
                is_maximal = False
                break
        if is_maximal:
            maximal_sets.append(s1)
    return maximal_sets

def horizontal_vertical(df, epss_verti, epss_hori):

    if not os.path.isfile('dataframes/horizontal_vertical_filterByATW_{}.pkl'.format(filterByATW)):

        print('Horizontal Clustering...')

        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values(by='datetime')

        for eps in epss_hori:
            df['hori_dbscan_{}'.format(eps)] = -2
            df['centroids_hori_dbscan_{}'.format(eps)] = -2
        df['EpsilonHTest'] = -2
        df['centroids_EpsilonHTest'] = -2

        extensions = df['extension'].unique().tolist()
        for extension in tqdm(extensions, position=0, leave=True):
            x_train_hori = df[df['extension']==extension]['datetime'].values.astype(np.int64) // 10 ** 9
            x_train_hori = StandardScaler().fit_transform(np.array(x_train_hori.tolist()).reshape(-1, 1))

            for eps in epss_hori:
                clustering = DBSCAN(eps=eps, min_samples=1).fit(x_train_hori)
                df.loc[df['extension']==extension,'hori_dbscan_{}'.format(eps)] = clustering.labels_
                centroids = []
                for cluster in tqdm(clustering.labels_,position=1, leave=False):
                    centroids.append(np.mean(df[(df['extension'] == extension) & (df['hori_dbscan_{}'.format(eps)]==cluster)]['timestamp'],axis=0))
                df.loc[df['extension'] == extension, 'centroids_hori_dbscan_{}'.format(eps)] = centroids


        print('Vertical Clustering Using the Centroids of the Horizontal Clustering...')

        for eps_hori in epss_hori:
            df['datetime_centroids_hori_dbscan_{}'.format(eps_hori)] = pd.to_datetime(df['centroids_hori_dbscan_{}'.format(eps_hori)], unit='ms')

        for eps_hori in epss_hori:
            for eps in epss_verti:
                df['dbscan_hori_{}_verti_{}'.format(eps_hori, eps)] = -2

        for eps_hori in epss_hori:
            x_train_verti = df['datetime_centroids_hori_dbscan_{}'.format(eps_hori)].values.astype(np.int64) // 10 ** 9
            x_train_verti = StandardScaler().fit_transform(np.array(x_train_verti.tolist()).reshape(-1, 1))

            for eps in tqdm(epss_verti, position=0, leave=True):
                clustering = DBSCAN(eps=eps, min_samples=3).fit(x_train_verti)
                df['dbscan_hori_{}_verti_{}'.format(eps_hori, eps)] = clustering.labels_

        df = df.sort_values(by=['extension', 'datetime'])
        df.to_pickle('dataframes/horizontal_vertical_filterByATW_{}.pkl'.format(filterByATW))

    else:
        df = pd.read_pickle('dataframes/horizontal_vertical_filterByATW_{}.pkl'.format(filterByATW))

    return df


def plotDF(df, eps, plot_or_file="P", filename=""):
    df_aux = df.sort_values(['extension', eps, 'timestamp'])

    # Group by 'labels' and calculate the time range for each group
    result = df_aux.groupby(['extension', eps]).agg({'timestamp': ['min', 'max', 'mean'], 'extension': 'count'})
    # Calculate the time range between the shortest and longest elements
    result['time_range'] = result['timestamp']['max'] - result['timestamp']['min']
    result['time_range_s'] = (result['timestamp']['max'] - result['timestamp']['min']) / 1000
    result['time_range_s_centroid'] = pd.concat([result['timestamp']['max'] - result['timestamp']['mean'],
                                                 result['timestamp']['mean'] - result['timestamp']['min']], axis=1,
                                                keys=['S1', 'S2']).max(axis=1) / 1000
    result['datetime'] = pd.to_datetime(result['time_range'], unit='ms')
    result['datetime_s'] = pd.to_datetime(result['time_range_s'], unit='s')
    result.reset_index(drop=False, inplace=True)

    # print(result['time_range_s_centroid'].describe())
    result_df = pd.DataFrame({'time': result['time_range_s_centroid'],
                              'extensions': result['extension', 'count'],
                              'epsilon': eps})
    g = sns.scatterplot(y='extensions', x='time', data=result_df, palette='viridis')

    # Add labels and title
    g.set(ylabel='Number of Extensions per group')
    g.set(xlabel='Time Range (seconds)')
    g.set(title='{} eps has a mean of: {:.2f}s'.format(eps, result['time_range_s_centroid'].describe()[ 'mean']))
    sns.despine()
    plt.tight_layout()
    if plot_or_file == "P":
        plt.show()
    else:
        plt.savefig(filename, dpi=500, bbox_inches='tight')
    plt.close()

def horizontal_parallel(args):
    extension, df, epss_hori = args

    x_train_hori = df['datetime'].values.astype(np.int64) // 10 ** 9
    x_train_hori = StandardScaler().fit_transform(np.array(x_train_hori.tolist()).reshape(-1, 1))

    for eps in epss_hori:
        clustering = DBSCAN(eps=eps, min_samples=2).fit(x_train_hori)
        df['hori_dbscan_{}'.format(eps)] = clustering.labels_
        centroids = []
        for cluster in clustering.labels_:
            centroids.append(np.mean(df[df['hori_dbscan_{}'.format(eps)]==cluster]['timestamp'],axis=0))
        df['centroids_hori_dbscan_{}'.format(eps)] = centroids

    return df

def parallelFunction(epss_verti, epss_hori):
    import multiprocessing as mp
    from multiprocessing.pool import ThreadPool

    # REVIEW_DATAFRAME     = 'dataframes/reviews_reviews_20230209.pkl'
    REVIEW_DATAFRAME_DEC2022 = 'dataframes/reviews_reviews_20230209_december2022.pkl'
    REVIEW_DATAFRAME = REVIEW_DATAFRAME_DEC2022
    HORIZONTAL_DATAFRAME = 'dataframes/dfHori.pkl'
    VERTICAL_DATAFRAME   = 'dataframes/HoriVert_filterByATW_{}.pkl'.format(False)

    if not os.path.isfile(HORIZONTAL_DATAFRAME):
        print('----- Horizontal Analysis -----')
        df = pd.read_pickle(REVIEW_DATAFRAME)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values(by=['extension', 'datetime'])


        # df2022 = df[(df['datetime']>=pd.Timestamp(date(2022,12,1))) & (df['datetime']<pd.Timestamp(date(2023,1,1)))]
        # df.to_pickle(REVIEW_DATAFRAME_DEC2022)

        df.info()


        for eps in epss_hori:
            df['hori_dbscan_{}'.format(eps)] = -2
            df['centroids_hori_dbscan_{}'.format(eps)] = -2

        extensions = df['extension'].unique().tolist()
        extensions.sort()
        extensions = extensions

        with ThreadPool(mp.cpu_count() - 2) as pool:
            results = list(tqdm(pool.imap(horizontal_parallel, [(key,df[df['extension']==key],epss_hori) for key in extensions]), total=len(extensions)))
        df = pd.concat(results)
        df = df.sort_values(by=['extension', 'datetime'])
        df.to_pickle(HORIZONTAL_DATAFRAME)
    else:
        df = pd.read_pickle(HORIZONTAL_DATAFRAME)

    if not os.path.isfile(VERTICAL_DATAFRAME):
        print('Vertical Clustering Using the Centroids of the Horizontal Clustering...')

        for eps_hori in epss_hori:
            df['datetime_centroids_hori_dbscan_{}'.format(eps_hori)] = pd.to_datetime(df['centroids_hori_dbscan_{}'.format(eps_hori)], unit='ms')

        for eps_hori in epss_hori:
            for eps in epss_verti:
                df['dbscan_hori_{}_verti_{}'.format(eps_hori, eps)] = -2

        for eps_hori in epss_hori:
            df_aux = df[df['hori_dbscan_{}'.format(eps_hori)]>-1]

            x_train_verti = df_aux['datetime_centroids_hori_dbscan_{}'.format(eps_hori)].values.astype(np.int64) // 10 ** 9
            x_train_verti = StandardScaler().fit_transform(np.array(x_train_verti.tolist()).reshape(-1, 1))

            for eps in tqdm(epss_verti, position=0, leave=True):
                clustering = DBSCAN(eps=eps, min_samples=3).fit(x_train_verti)
                df.loc[df.index.isin(df_aux.index.tolist()), 'dbscan_hori_{}_verti_{}'.format(eps_hori, eps)] = clustering.labels_
        print()
        df = df.sort_values(by=['extension', 'datetime'])
        df.to_pickle(VERTICAL_DATAFRAME)
    else:
        df = pd.read_pickle(VERTICAL_DATAFRAME)
    return df

def filterDF(df, eps):
    df_aux = df[df[eps] > -1][['extension', 'user', 'review', 'timestamp', 'datetime', eps]]
    df_aux = df_aux.sort_values(['extension', eps, 'timestamp'])
    set_aux = df_aux.groupby(eps)['extension'].apply(set).tolist()
    print(f'For eps {eps}')
    print(f'There are {len(set_aux)} clusters')
    unique_clusters = list(set(tuple(subconjunto) for subconjunto in set_aux))
    unique_clusters_2 = [s for s in unique_clusters if len(s) > 1]
    print(f'{len(unique_clusters_2)} unique clusters with more than one extension:')
    print(unique_clusters_2)
    test = df_aux.groupby(eps)['extension'].agg(lambda x: set(x)).reset_index().drop_duplicates('extension')
    # Next line: Get clusters with more than 2 extensions --> This is what actually deletes POPULAR extensions
    # Later I get the maximals set!
    print(f'There are {len(unique_clusters)} unique clusters')
    print(f'These have an average length of {sum(len(m) for m in unique_clusters)/len(unique_clusters)}')
    maximals_set = [sublista for sublista in unique_clusters if len(sublista) > 2]
    print(f'Of which {len(maximals_set)} have length > 2')
    if len(maximals_set) > 0:
        print(f'These have an average length of {sum(len(m) for m in maximals_set)/len(maximals_set)}')
    maximals_set = find_maximal_sets(maximals_set)
    return df_aux, unique_clusters, maximals_set

def getFinalClusters(df, eps, plot=False):
    df_aux, unique_clusters, maximals_set = filterDF(df, eps)

    if plot:
        plotDF(df_aux, eps, plot_or_file="F", filename='output/before_{}.pdf'.format(eps))

    unique_clusters_df = df_aux.groupby(eps)['extension'].agg(lambda x: set(x)).reset_index().drop_duplicates('extension')
    grouped_df = unique_clusters_df[unique_clusters_df['extension'].apply(lambda x: len(x) > 1)][[eps, 'extension']]

    final_clusters = df_aux[df_aux[eps].isin(unique_clusters)]
    if len(maximals_set) > 0:
        final_clusters = grouped_df[grouped_df['extension'].apply(lambda x: x in maximals_set)][eps].to_list()
        final_clusters = df_aux[df_aux[eps].isin(final_clusters)]

    if plot:
        plotDF(final_clusters, eps, plot_or_file="F", filename='output/after_{}.pdf'.format(eps))

    if df_aux['extension'].unique().shape[0] != len(list(set(elem for item in unique_clusters for elem in item))):
        assert ('Unique Clusters: Integrity check error!!!')
    if final_clusters['extension'].unique().shape[0] != len(list(set(elem for item in maximals_set for elem in item))):
        assert ('Maximals Set: Integrity check error!!!')

    return df_aux, unique_clusters, maximals_set, final_clusters, grouped_df

def getFinalClustersOld(df_ori, epsilon, quantile=0.05):
    df_aux = df_ori[df_ori[epsilon] > -1]
    set_aux = df_aux.groupby(epsilon)['extension'].apply(set).tolist()

    # Convert sets to tuples and add to a set to remove duplicates
    set_aux2 = list(set(tuple(subconjunto) for subconjunto in set_aux))
    set_aux2_filtered = [sublista for sublista in set_aux2 if len(sublista) > 1]

    # Filter FP:
    item_counts = {}
    for sublist in set_aux2_filtered:
        for item in sublist:
            item_counts[item] = item_counts.get(item, 0) + 1
    df_FP = pd.DataFrame(list(item_counts.items()), columns=['extension', 'freq'])
    df_FP.reset_index(drop=True, inplace=True)
    df_FP = df_FP.sort_values(by=['freq','extension'])
    extensionsFP = set(df_FP[df_FP['freq']>=df_FP['freq'].quantile(quantile)]['extension'].tolist())

    return extensionsFP

def getStats(df, epsilons):
    output = []
    print('Getting Stats....\n')
    totalN = df['extension'].unique().shape[0]

    for eps in tqdm(epsilons):
        df_aux, unique_clusters, maximals_set, final_clusters, grouped_df = getFinalClusters(df, eps, plot=True)

        extensions_in_unique_clusters = len(set().union(*unique_clusters))
        extensions_in_maximals_set = final_clusters['extension'].unique().shape[0]

        output.append({'epsilon':eps,
                       'VC': len(df_aux[eps].unique()),
                       'UVC': len(unique_clusters),
                       'UVC > 2': len(maximals_set),
                       'Ext. in VC (%)': '{} ({:.1f})'.format(df_aux['extension'].unique().shape[0],(df_aux['extension'].unique().shape[0]*100)/totalN),
                       'Ext. in UVC (%)': '{} ({:.1f})'.format(extensions_in_unique_clusters,(extensions_in_unique_clusters*100)/totalN),
                       'Ext. in UVC > 2 (%)': '{} ({:.1f})'.format(extensions_in_maximals_set,(extensions_in_maximals_set*100)/totalN),
                  })


    output = pd.DataFrame(output)
    output = output.sort_values(by='VC', ascending=False)
    # print(output.to_latex(index=False))

    latex_table = r"""
    \documentclass{article}
    \usepackage{booktabs}
    \usepackage{graphicx}
    \usepackage{lscape}
    \begin{document}
    \begin{landscape}
    \begin{table}
    \centering
    \caption{VC: Vertical Clusters; UVC: Unique Vertical Cluster}
    \resizebox{1.9\textwidth}{!}{
    %s
    }
    \end{table}
    \end{landscape}
    \end{document}
    """ % output.to_latex(index=False)

    OUTPUT_LATEX = 'output/output.tex'
    with open(OUTPUT_LATEX, 'w') as f:
        f.write(latex_table)
        print(f'Wrote clusters stats to LaTeX table in {OUTPUT_LATEX}')

    print()

def getEpsStats(df, epsilons):
    print('Getting horizontal epsilon stats....\n')

    columns = [element for element in df.columns.to_list() if element.startswith("hori_dbscan")]
    for eps in tqdm(columns):
        df_aux = df[df[eps] > -1]

        df_aux = df_aux[df_aux[eps] > -1][
            ['extension', 'user', 'review', 'timestamp', 'datetime', eps, 'centroids_{}'.format(eps)]]
        df_aux = df_aux.sort_values(['extension', eps, 'timestamp'])

        # Group by 'labels' and calculate the time range for each group
        result = df_aux.groupby(['extension', eps]).agg({'timestamp': ['min', 'max', 'mean'], 'extension': 'count'})
        # Calculate the time range between the shortest and longest elements
        result['time_range'] = result['timestamp']['max'] - result['timestamp']['min']
        result['time_range_s'] = (result['timestamp']['max'] - result['timestamp']['min']) / 1000
        result['time_range_s_centroid'] = pd.concat([result['timestamp']['max'] - result['timestamp']['mean'],
                                                     result['timestamp']['mean'] - result['timestamp']['min']], axis=1,
                                                    keys=['S1', 'S2']).max(axis=1) / 1000
        result['datetime'] = pd.to_datetime(result['time_range'], unit='ms')
        result['datetime_s'] = pd.to_datetime(result['time_range_s'], unit='s')
        result.reset_index(drop=False, inplace=True)

        print(result['time_range_s_centroid'].describe())
        result_df = pd.DataFrame({'time': result['time_range_s_centroid'],
                                  'extensions': result['extension', 'count'],
                                  'epsilon': eps,
                                  'extension':result['extension','']})

        g = sns.scatterplot(y='extensions', x='time', data=result_df, palette='viridis')

        # Add labels and title
        g.set(ylabel='Number of Extensions per group')
        g.set(xlabel='Time Range (seconds)')
        g.set(title='{} extensions with {} eps has a mean of: {:.2f}s'.format(len(result_df['extension'].unique()), eps, result['time_range_s_centroid'].describe()['mean']))

        sns.despine()
        plt.tight_layout()
        # plt.show()
        HORIZONTAL_EPS_PDF = 'output/eps_hori_stats_{}.pdf'.format(eps)
        plt.savefig(HORIZONTAL_EPS_PDF, dpi=500, bbox_inches='tight')
        plt.close()
        print(f'Saved horizontal epsilon stats to {HORIZONTAL_EPS_PDF}')
        print()

def getEpsVerticalStats(df_ori, epsilons):
    print('Getting vertical epsilon stats....\n')

    for epsilon in tqdm(epsilons, position=0, leave=False):

        df_aux = df_ori[df_ori[epsilon] > -1][['extension','user','review','timestamp','datetime',epsilon]]
        df_aux = df_aux.sort_values([epsilon,'timestamp'])

        # Group by 'labels' and calculate the time range for each group
        result = df_aux.groupby(epsilon).agg({'timestamp': ['min', 'max', 'mean'], 'extension':'count'})
        # Calculate the time range between the shortest and longest elements
        result['time_range'] = result['timestamp']['max'] - result['timestamp']['min']
        result['time_range_s'] = (result['timestamp']['max'] - result['timestamp']['min'])/1000
        result['time_range_s_centroid'] = pd.concat([result['timestamp']['max'] - result['timestamp']['mean'], result['timestamp']['mean'] - result['timestamp']['min']], axis=1, keys=['S1','S2']).max(axis=1)/1000
        result['datetime'] = pd.to_datetime(result['time_range'], unit='ms')
        result['datetime_s'] = pd.to_datetime(result['time_range_s'], unit='s')
        result[epsilon] = df_aux[epsilon].unique()

        metric = 'time_range_s_centroid'
        # Merge the two results into a single DataFrame
        result_df = pd.DataFrame({'Time Range': result[metric], 'Number of Extensions': result['extension', 'count'], 'epsilon': result[epsilon]})

        g = sns.scatterplot(y='Number of Extensions', x='Time Range', data=result_df, palette='viridis')

        # Add labels and title
        g.set(ylabel='Number of Extensions per group')
        g.set(xlabel='Time Range (seconds)')
        g.set(title='{} eps has a mean of: {:.2f}s'.format(epsilon, result[ 'time_range_s_centroid'].describe()[ 'mean']))

        sns.despine()
        plt.tight_layout()
        VERTICAL_EPS_PDF = 'output/eps_vert_stats_{}.pdf'.format(epsilon)
        plt.savefig(VERTICAL_EPS_PDF, dpi=500, bbox_inches='tight')
        plt.close()
        # Show the plot
        # plt.show()
        print(f'Saved vertical epsilon stats to {VERTICAL_EPS_PDF}')
        print()

if __name__ == "__main__":
    epss_verti = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001]
    epss_hori =  [0.000001, 0.00001, 0.0001]

    # small_example parameters
    # epss_verti = [5e-06]
    # epss_hori = [1e-06]

    epsilons = ['dbscan_hori_{}_verti_{}'.format(eps_hori, eps) for eps in epss_verti for eps_hori in epss_hori]

    hoverDF = parallelFunction(epss_verti, epss_hori)
    getEpsStats(hoverDF, epsilons)
    getEpsVerticalStats(hoverDF, epsilons)
    getStats(hoverDF, epsilons)


