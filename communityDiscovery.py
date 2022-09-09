from data_collection.dataColl import collectData
from data_collection.dataHandling import *
from cdlib.classes.node_clustering import NodeClustering
from cdlib import evaluation

import pandas as pd
from cdlib import algorithms

def main():
    dfMentions = pd.read_json('data_collection/data/df_mentions.json', orient='records', lines=True)

    # angel
    dfMentions3c = dfMentions[['username', 'screen_name', 'date']]
    g = createGraph(dfMentions3c)
    commsAngel = algorithms.angel(g, min_community_size=9, threshold=0.25)
    print("Angel algorithm")
    print("AID:", commsAngel.average_internal_degree())
    print("Internal density:",commsAngel.internal_edge_density())
    print("Girvan Newman modularity:", commsAngel.newman_girvan_modularity())
    print("Conductance:",commsAngel.conductance())
    print("Node coverage",commsAngel.node_coverage)
    print("Triangle participation ratio",commsAngel.triangle_participation_ratio())
    print("Average ODF degree",commsAngel.avg_odf())
    print("Fraction over median degree",commsAngel.fraction_over_median_degree())
    print("Expansion",commsAngel.expansion())
    print("Normalized cut",commsAngel.normalized_cut())
    print("")
    
    #louvain
    commsLou = algorithms.louvain(g)
    print("Louvain algorithm")
    print("AID:", commsLou.average_internal_degree())
    print("Internal density:", commsLou.internal_edge_density())
    print("Girvan Newman modularity:", commsLou.newman_girvan_modularity())
    print("Conductance:", commsLou.conductance())
    print("Node coverage",commsLou.node_coverage)
    print("Triangle participation ratio", commsLou.triangle_participation_ratio())
    print("Average ODF degree", commsLou.avg_odf())
    print("Fraction over median degree", commsLou.fraction_over_median_degree())
    print("Expansion", commsLou.expansion())
    print("Normalized cut", commsLou.normalized_cut())
    print("")

    commsLab = algorithms.label_propagation(g)
    print("Label propagation algorithm")
    print("AID:", commsLab.average_internal_degree())
    print("Internal density:", commsLab.internal_edge_density())
    print("Girvan Newman modularity:", commsLab.newman_girvan_modularity())
    print("Conductance:", commsLab.conductance())
    print("Node coverage", commsLab.node_coverage)
    print("Triangle participation ratio", commsLab.triangle_participation_ratio())
    print("Average ODF degree", commsLab.avg_odf())
    print("Fraction over median degree", commsLab.fraction_over_median_degree())
    print("Expansion", commsLab.expansion())
    print("Normalized cut", commsLab.normalized_cut())
    print("")

    commsWalkTrap = algorithms.walktrap(g)
    print("Walktrap algorithm")
    print("AID:", commsWalkTrap.average_internal_degree())
    print("Internal density:", commsWalkTrap.internal_edge_density())
    print("Girvan Newman modularity:", commsWalkTrap.newman_girvan_modularity())
    print("Conductance:", commsWalkTrap.conductance())
    print("Node coverage", commsWalkTrap.node_coverage)
    print("Triangle participation ratio", commsWalkTrap.triangle_participation_ratio())
    print("Average ODF degree", commsWalkTrap.avg_odf())
    print("Fraction over median degree", commsWalkTrap.fraction_over_median_degree())
    print("Expansion", commsWalkTrap.expansion())
    print("Normalized cut", commsWalkTrap.normalized_cut())
    print("")




    dfTweets = pd.read_json('data_collection/data/virologists.json', orient='records', lines=True)
    virologists = ["Fabrizio Pregliasco", "Robert Gallo", "Roberto Gualtieri", "Andrea Crisanti", "Massimo Galli",
                   "Roberto Burioni", "Ilaria Capua", "Alberto Zangrillo", "Walter Ricciardi", "Matteo Bassetti",
                   "Silvio Brusaferro", "Pier Luigi Lopalco", "Nino Cartabellotta"]
    groundTruthComms = [[], [], [], [], [], [], [], [], [], [], [], [], []]
    for i, tweet in enumerate(dfTweets['tweet']):
        for j, virologist in enumerate(virologists):
            if virologist in tweet:
                groundTruthComms[j].append(dfTweets['username'][i])



    for i,comm in enumerate(groundTruthComms):
        groundTruthComms[i] = list(set(groundTruthComms[i]))


    NodeClusteringObject = NodeClustering(groundTruthComms, g, "", None, True)
    valNf1Angel = evaluation.nf1(commsAngel, NodeClusteringObject)
    valF1Angel = evaluation.f1(commsAngel, NodeClusteringObject)
    print("Scores for Angel: ",valNf1Angel, " ", valF1Angel)
    print("Number of communities for Angel:", len(commsAngel.communities))

    valNf1Louvain = evaluation.nf1(commsLou, NodeClusteringObject)
    valF1Louvain = evaluation.f1(commsLou, NodeClusteringObject)
    print("Scores for Louvain:",valNf1Louvain, " ", valF1Louvain)
    print("Number of communities for Louvain:", len(commsLou.communities))

    valNf1Label= evaluation.nf1(commsLab, NodeClusteringObject)
    valF1Label = evaluation.f1(commsLab, NodeClusteringObject)
    print("Scores for label propagation:",valNf1Label, " ", valF1Label)
    print("Number of communities for label propagation:", len(commsLab.communities))

    valNf1Walktrap = evaluation.nf1(commsWalkTrap, NodeClusteringObject)
    valF1Waltrap = evaluation.f1(commsWalkTrap, NodeClusteringObject)
    print("Scores for walktrap:",valNf1Walktrap, " ", valF1Waltrap)
    print("Number of communities for walktrap:", len(commsWalkTrap.communities))

if __name__ == "__main__":
    main()