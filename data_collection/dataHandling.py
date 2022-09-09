import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import powerlaw


def createDFMentions(df, save):
    # liste per DF_MENTIONS

    dates = []
    times = []
    languages = []
    retweets = []
    hashtags = []
    tweets = []
    ids = []
    mentionedIds = []
    places = []
    usernames = []
    screenNames = []
    indexes = []

    replies = []
    replies_count = []
    retweets_count = []

    # POPOLO df_mentions (SOLO ACCOUNT CON MENZIONE)

    for i, list in enumerate(df['mentions']):
        for dict in list:
            if 'screen_name' in dict:
                ids.append(df['id'][i])
                usernames.append(df['username'][i])
                mentionedIds.append(dict['id'])
                screenNames.append(dict['screen_name'])
                indexes.append(df['mentions'].index[i])
                dates.append(df['date'][i])
                times.append(df['time'][i])
                places.append(df['created_at'][i])
                tweets.append(df['tweet'][i])
                hashtags.append(df['hashtags'][i])
                languages.append(df['language'][i])
                retweets.append(df['retweet'][i])
                retweets_count.append(df['retweets_count'][i])
                replies_count.append(df['replies_count'][i])
                replies.append(df['reply_to'][i])

    df_mentions = pd.DataFrame()

    df_mentions['created_at'] = places
    df_mentions['date'] = dates
    df_mentions['time'] = times
    df_mentions['id'] = ids
    df_mentions['username'] = usernames
    df_mentions['mentioned_id'] = mentionedIds
    df_mentions['screen_name'] = screenNames
    df_mentions['tweet'] = tweets
    df_mentions['language'] = languages
    df_mentions['hashtags'] = hashtags
    df_mentions['reply_to'] = replies
    df_mentions['replies_count'] = replies_count
    df_mentions['retweet'] = retweets
    df_mentions['retweets_count'] = retweets_count
    df_mentions['conversation_index'] = indexes

    if save == True:
        df_mentions.to_json('data_collection/data/df_mentions.json', orient='records', lines=True)

    return df_mentions


def createDiGraph(df, fileName):
    g = nx.DiGraph()
    for index, row in df.iterrows():
        if g.has_edge(row[0], row[1]):
            g[row[0]][row[1]]['weight'] += 1
        else:
            g.add_node(row[0])
            g.add_node(row[1])
            g.add_edge(row[0], row[1], weight=1)

    print('Number of nodes:', g.number_of_nodes(), 'Number of edges:', g.number_of_edges(),'Size:', g.size(weight='weight'))
    nx.write_gexf(g, 'data_collection/data/' + fileName +'.gexf')
    return g

def createGraph(df):
    g = nx.Graph()
    for index, row in df.iterrows():
        if g.has_edge(row[0], row[1]):
            g[row[0]][row[1]]['weight'] += 1
        else:
            g.add_node(row[0])
            g.add_node(row[1])
            g.add_edge(row[0], row[1], weight=1)

    return g


def createGraphNoWeights(df):
    g = nx.DiGraph()
    for index, row in df.iterrows():
        if not g.has_edge(row[0], row[1]):
            g.add_node(row[0])
            g.add_node(row[1])
            g.add_edge(row[0], row[1], weight=1)

    return g

def createGraphNoWeightsNotDirected(df):
    g = nx.Graph()
    for index, row in df.iterrows():
        if not g.has_edge(row[0], row[1]):
            g.add_node(row[0])
            g.add_node(row[1])
            g.add_edge(row[0], row[1], weight=1)

    return g

def createGraphForTrainTestSplit(df,trainGraph):
    g = nx.Graph()
    for index, row in df.iterrows():
        if (not g.has_edge(row[0], row[1])) and (not trainGraph.has_edge(row[0], row[1])):
            g.add_node(row[0])
            g.add_node(row[1])
            g.add_edge(row[0], row[1], weight=1)


    return g

def plotDist(g,title):
    """M = nx.to_scipy_sparse_matrix(g)
    xmin = min([d[1] for d in g.degree()])
    indegrees = np.asarray(M.sum(1)).flatten(order='C')
    degree = np.bincount(indegrees)
    fit = powerlaw.Fit(np.array(degree) + 1, fit_method='KS')  # xmin=xmin, xmax=max(degree)-xmin,discrete=True)

    fig = plt.figure(figsize=(16, 6))"""

    """ Plot Distribution """
    #plt.subplot(1, 3, 1)

    """plt.plot(range(len(degree)), degree, 'b.')
    plt.loglog()
    plt.xlim((min(degree), 10**1.3))
    plt.xlabel('Degree')
    plt.ylabel('P(k)')
    plt.show()"""

    hist = nx.degree_histogram(g)
    plt.plot(range(0, len(hist)), hist, ".")
    plt.title(title)
    plt.xlabel("Degree")
    plt.ylabel("#Nodes")
    plt.loglog()
    plt.show()

    """
    #Plot CDF 
    plt.subplot(1, 3, 2)
    fit.plot_cdf()
    plt.xlabel("Degree")
    plt.ylabel('CDF')

    # Plot CCDF 
    plt.subplot(1, 3, 3)
    fit.plot_ccdf()
    plt.ylabel('CCDF')
    plt.xlabel('Degree')
    plt.tight_layout()
    plt.show()"""

def plotComparison(gER, gWS, gCM,gBA):
    histER = nx.degree_histogram(gER)
    histWS = nx.degree_histogram(gWS)
    histCM = nx.degree_histogram(gCM)
    histBA = nx.degree_histogram(gBA)

    fig = plt.figure(figsize=(16, 6))

    plt.subplot(1, 4, 1)
    plt.plot(range(0, len(histER)), histER, ".")
    plt.title("Erdős-Rényi model")
    plt.xlabel("Degree")
    plt.ylabel("#Nodes")
    plt.loglog()

    plt.subplot(1, 4, 2)
    plt.plot(range(0, len(histWS)), histWS, ".")
    plt.title("Watts-Strogatz model")
    plt.xlabel("Degree")
    plt.ylabel("#Nodes")
    plt.loglog()

    plt.subplot(1, 4, 3)
    plt.plot(range(0, len(histCM)), histCM, ".")
    plt.title("Configuration model")
    plt.xlabel("Degree")
    plt.ylabel("#Nodes")
    plt.loglog()

    plt.subplot(1, 4, 4)
    plt.plot(range(0, len(histBA)), histBA, ".")
    plt.title("Barabàsi-Albert model")
    plt.xlabel("Degree")
    plt.ylabel("#Nodes")
    plt.loglog()


    plt.show()

def powerLawFit(g):
    M = nx.to_scipy_sparse_matrix(g)
    degrees = [d[1] for d in g.degree()]
    xmin = min(degrees)

    fit = powerlaw.Fit(degrees)  # xmin=xmin, xmax=max(degree)-xmin,discrete=True)

    return fit.alpha



