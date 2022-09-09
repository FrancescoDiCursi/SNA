from data_collection.dataHandling import *
import pandas as pd
from linkpred.evaluation import Pair
import linkpred
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import auc

def main():
    dfMentions = pd.read_json('data_collection/data/df_mentions.json', orient='records', lines=True)
    dfMentions3c = dfMentions[['username', 'screen_name', 'date']]
    sortedDf = dfMentions3c.sort_values(by='date')
    g = createGraph(sortedDf)
    print(len(g.nodes))
    print(len(g.edges))
    sortedSampledDf = sortedDf.sort_values(by='date').sample(frac=0.15, axis=0)
    sortedSampledDf = sortedSampledDf.sort_values(by='date')
    print(sortedSampledDf)

    g = createGraphNoWeightsNotDirected(sortedSampledDf)

    totalNodes = list(g.nodes())
    print(len(totalNodes))

    trainingLength = int(len(sortedSampledDf)*0.8)
    testLength = int(len(sortedSampledDf)*0.2+1)

    dfTrain = sortedSampledDf[0:(trainingLength-1)]
    dfTest = sortedSampledDf[trainingLength:(trainingLength+testLength-1)]

    trainG = createGraphNoWeightsNotDirected(dfTrain)
    testG = createGraphForTrainTestSplit(dfTest, trainG)

    # Compute the test set and the universe set
    trainEdges = [edge for edge in trainG.edges() if edge[0] != edge[1]]
    testEdges = [edge for edge in testG.edges() if edge[0] != edge[1]]
    test = [Pair(i) for i in testEdges]
    N = len(g)
    universe = N*(N-1) // 2
    #universe = set([Pair(i) for i in itertools.product(totalNodes, totalNodes) if i[0] != i[1]]) very often goes out of memory
    print("Finished building graphs")

    print("")
    print("Common Neighbours")
    cn = linkpred.predictors.CommonNeighbours(trainG, excluded=trainEdges)
    CNPredict = cn.predict()

    topCN = CNPredict.top(10)
    listPredLinks = []
    for authors, score in topCN.items():
        listPredLinks.append(authors)
        print(authors, score)
    print(len(listPredLinks))
    print("")
    print("Preferential Attachment")
    dp = linkpred.predictors.DegreeProduct(trainG, excluded=trainEdges)
    DPPredict = dp.predict()

    topDP = DPPredict.top(10)
    for authors, score in topDP.items():
        print(authors, score)

    print("")
    print("Adamic Adar")
    aa = linkpred.predictors.AdamicAdar(trainG, excluded=trainEdges)
    AAPredict = aa.predict()
    #print(AAPredict)

    topAA = AAPredict.top(10)
    for authors, score in topAA.items():
        print(authors, score)

    print("")
    print("Jaccard")
    jc = linkpred.predictors.Jaccard(trainG, excluded=trainEdges)
    JCPredict = jc.predict()
    #print(JCPredict)

    topJC = JCPredict.top(10)
    for authors, score in topJC.items():
        print(authors, score)

    cn_evaluation = linkpred.evaluation.EvaluationSheet(CNPredict, test, universe)
    aa_evaluation = linkpred.evaluation.EvaluationSheet(AAPredict, test, universe)
    jc_evaluation = linkpred.evaluation.EvaluationSheet(JCPredict, test, universe)
    dp_evaluation = linkpred.evaluation.EvaluationSheet(DPPredict, test, universe)

    print("Common neighours evaluation")
    print("Max F1 score")
    f1_score_cn_list = cn_evaluation.f_score().tolist()
    max_f1_cn = max(f1_score_cn_list)
    index_max_f1_cn = f1_score_cn_list.index(max_f1_cn)
    print(index_max_f1_cn)
    print(max_f1_cn)
    print("Precision")
    print(cn_evaluation.precision()[index_max_f1_cn])
    print("Recall")
    print(cn_evaluation.recall()[index_max_f1_cn])

    print("Adamic Adar evaluation")
    print("Max F1 score")
    f1_score_aa_list = aa_evaluation.f_score().tolist()
    max_f1_aa = max(f1_score_aa_list)
    index_max_f1_aa = f1_score_aa_list.index(max_f1_aa)
    print(index_max_f1_aa)
    print(max_f1_aa)
    print("Precision")
    print(aa_evaluation.precision()[index_max_f1_aa])
    print("Recall")
    print(aa_evaluation.recall()[index_max_f1_aa])

    print("Jaccard evaluation")
    print("Max F1 score")
    f1_score_jc_list = jc_evaluation.f_score().tolist()
    max_f1_jc = max(f1_score_jc_list)
    index_max_f1_jc = f1_score_jc_list.index(max_f1_jc)
    print(index_max_f1_jc)
    print(max_f1_jc)
    print("Precision")
    print(jc_evaluation.precision()[index_max_f1_jc])
    print("Recall")
    print(jc_evaluation.recall()[index_max_f1_jc])

    print("Prefential Attachment evaluation")
    print("Max F1 score")
    f1_score_dp_list = dp_evaluation.f_score().tolist()
    max_f1_dp = max(f1_score_dp_list)
    index_max_f1_dp = f1_score_dp_list.index(max_f1_dp)
    print(index_max_f1_dp)
    print(max_f1_dp)
    print("Precision")
    print(dp_evaluation.precision()[index_max_f1_dp])
    print("Recall")
    print(dp_evaluation.recall()[index_max_f1_dp])


    print(f"Common Neigh.: \t {auc(cn_evaluation.fallout(), cn_evaluation.recall())}")
    print(f"Adamic Adar: \t {auc(aa_evaluation.fallout(), aa_evaluation.recall())}")
    print(f"Jaccard: \t {auc(jc_evaluation.fallout(), jc_evaluation.recall())}")
    print(f"Degree Product: \t {auc(dp_evaluation.fallout(), dp_evaluation.recall())}")

    plt.plot(cn_evaluation.fallout(), cn_evaluation.recall(), label="Common Neighbors")
    plt.plot(aa_evaluation.fallout(), aa_evaluation.recall(), label="Adamic Adar")
    plt.plot(jc_evaluation.fallout(), jc_evaluation.recall(), label="Jaccard")
    plt.ylabel("TPR")
    plt.xlabel("FPR")
    plt.legend(prop={'size': 6}, bbox_to_anchor=(0, 1), loc='upper left')
    plt.savefig('report/img/AUCCAJPlot.png', bbox_inches='tight', dpi=100)
    plt.show()

    plt.plot(dp_evaluation.fallout(), dp_evaluation.recall(), label="Degree Product")
    plt.ylabel("TPR")
    plt.xlabel("FPR")
    plt.legend(prop={'size': 6}, bbox_to_anchor=(0, 1), loc='upper left')
    plt.savefig('report/img/AUCDPlot.png', bbox_inches='tight', dpi=100)
    plt.show()

if __name__ == "__main__":
    main()