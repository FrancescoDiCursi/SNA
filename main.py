from data_collection.dataColl import collectData
from data_collection.dataHandling import *

import pandas as pd
import numpy as np
import networkx as nx

def main():
    # data collection, comment this if you have the virologists file up to the read_json(df_mentions)
    '''since = "2020-01-01"
    until = "2021-07-10"
    fileName = "virologists"
    virologists = ["Fabrizio Pregliasco", "Robert Gallo", "Roberto Gualtieri", "Andrea Crisanti", "Massimo Galli",
                   "Roberto Burioni", "Ilaria Capua", "Alberto Zangrillo", "Walter Ricciardi", "Matteo Bassetti",
                   "Silvio Brusaferro", "Pier Luigi Lopalco", "Nino Cartabellotta"]

    collectData(since, until, virologists, fileName)

    # data analysis
    df = pd.read_json('data_collection/data/virologists.json', orient='records', lines=True)

    #the following line saves the DF mentions on disk, this is useful on first run
    #but for following runs it's better to read from disk
    dfMentions = createDFMentions(df, True)'''

    dfMentions = pd.read_json('data_collection/data/df_mentions.json', orient='records', lines=True)

    #create minimal dataset
    dfMentions3c = dfMentions[['username', 'screen_name', 'date']]
    g = createDiGraph(dfMentions3c, 'virologist_mentions')
    numberOfNodes = g.number_of_nodes()
    numberOfEdges = g.number_of_edges()
    plotDist(g, "Real graph")

    # clustering coefficient and density analysis
    print("Average clustering real network:",nx.average_clustering(g))
    print("Density real network",nx.density(g))
    degreeList = [d[1] for d in g.degree()]
    print("Sum of degrees", np.sum(degreeList))
    meanDegree = np.mean(degreeList)
    print("average degree:",meanDegree)

    #plot of the distributions of an ER random graph with the same number of nodes and edges
    gER = nx.gnm_random_graph(n=numberOfNodes, m=numberOfEdges, directed=True)
    print(gER.number_of_nodes(), gER.number_of_edges())
    nx.write_gexf(gER, "data_collection/data/ERRandom.gexf")
    print("ER avg clustering", nx.average_clustering(gER))

    # Watts-Strogatz
    gWS = nx.watts_strogatz_graph(numberOfNodes,5,0.47)
    print('Number of nodes:', gWS.number_of_nodes(), 'Number of edges:', gWS.number_of_edges())
    print(nx.average_clustering(gWS))
    nx.write_gexf(gWS, "data_collection/data/WS.gexf")

    #configuration model
    gSup = createGraphNoWeights(dfMentions3c)
    dIn = [d[1] for d in gSup.in_degree()]
    dOut = [d[1] for d in gSup.out_degree()]

    gCM = nx.directed_configuration_model(dIn,dOut)
    gCM = nx.DiGraph(gCM)
    print("Clustering coefficient Configuration model:", nx.average_clustering(gCM))
    print("Density configuration model:", nx.density(gCM))
    nx.write_gexf(gCM, "data_collection/data/CM.gexf")



    #barabasi-albert

    gBA = nx.barabasi_albert_graph(n=numberOfNodes, m=2)
    print("Number of nodes BA:", gBA.number_of_nodes(),"Number of links BA:",gBA.number_of_edges())
    print("Average clustering coefficient BA", nx.average_clustering(gBA))
    print("Density BA", nx.density(gBA))
    nx.write_gexf(gBA, "data_collection/data/BA.gexf")
    plotDist(gBA,"Barabasi Albert")

    #plot degree distributions
    plotComparison(gER, gWS, gCM, gBA)

    alpha = powerLawFit(g)
    #SPL = nx.average_shortest_path_length(g) NOTE: CALCULATED WITH GEPHI
    print("Mean degree:", meanDegree)
    #print("Average shortest path length:", SPL)
    print(alpha)

    #centrality analysis
    #eigenvector centrality
    print("eigenvector centrality")
    eigenC = nx.eigenvector_centrality(g)
    
    print(sorted(eigenC, key=eigenC.__getitem__)[-10:-1])
    print(sorted(eigenC.values())[-10:-1])

    #pagerank
    print("pagerank")
    pageRankC = nx.pagerank(g)
    print(sorted(pageRankC, key=pageRankC.__getitem__)[-10:-1])
    print(sorted(pageRankC.values())[-10:-1])
    
    #closeness centrality
    print("closeness centrality")
    closenessC = nx.closeness_centrality(g)
    print(sorted(closenessC, key=closenessC.__getitem__)[-10:-1])
    print(sorted(closenessC.values())[-10:-1])

    #harmonic centrality
    print("harmonic centrality")
    harmonicC = nx.harmonic_centrality(g)
    print(sorted(harmonicC, key=harmonicC.__getitem__)[-10:-1])
    print(sorted(harmonicC.values())[-10:-1])

if __name__ == "__main__":
    main()



