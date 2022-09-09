import math

import numpy.random.mtrand

from data_collection.dataHandling import *
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import matplotlib.pyplot as plt
import pandas as pd
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
from ndlib.viz.mpl.DiffusionPrevalence import DiffusionPrevalence


def main():
    dfMentions = pd.read_json('data_collection/data/df_mentions.json', orient='records', lines=True)

    # create minimal dataset
    dfMentions3c = dfMentions[['username', 'screen_name', 'date']]
    g = createGraph(dfMentions3c)
    numberOfNodes = g.number_of_nodes()
    numberOfEdges = g.number_of_edges()
    gER = nx.gnm_random_graph(n=numberOfNodes, m=numberOfEdges, directed=False)
    gBA = nx.barabasi_albert_graph(n=numberOfNodes, m=2)

    mu = 0.5
    baseSigma = mu/5

    fraction = 0.01
    numberOfBins = int(math.sqrt(numberOfNodes))
    for i in range(1, 51):
        modelER = ep.ThresholdModel(g)
        sigma = i * baseSigma

        config = mc.Configuration()

        config.add_model_parameter("fraction_infected", fraction)
        thresholds = numpy.random.normal(mu, sigma, numberOfNodes)
        thresholds[thresholds < 0] = 0
        thresholds[thresholds > 1] = 1

        # plot empirical CDF

        fig, ax = plt.subplots(figsize=(8, 4))

        # plot the cumulative histogram
        n, bins, patches = ax.hist(thresholds, bins=numberOfBins, density=True, histtype='step',
                                   cumulative=True, label='Empirical')

        # Add a line showing the expected distribution.
        z = np.full((1, len(bins)), 1)
        z = z.cumsum()
        z = z / z[-1]

        ax.plot(bins, z, 'k--', linewidth=1.5, label='$y=x$ line')

        # tidy up the figure
        ax.legend(loc='right')
        ax.set_title('CDF of Thresholds')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Cumulative probability')

        plt.savefig(
            'report/img/diffusionModels/ourThreshold/CDFs/diffusionOurThresholdCDF_fraction =' + str(fraction) + '_mu'
            + str(mu) + "_sigma=" + str(sigma) + ".png")

        for index, node in enumerate(g.nodes()):
            config.add_node_configuration("threshold", node, thresholds.item(index))

        modelER.set_initial_status(config)

        iterationsER = modelER.iteration_bunch(200)

        trendsER = modelER.build_trends(iterationsER)
        vizER = DiffusionTrend(modelER, trendsER)
        vizERDP = DiffusionPrevalence(modelER, trendsER)

        vizER.plot(
            "report/img/diffusionModels/ourThreshold/diffusionTrend/diffusionOurThreshold_fraction=" + str(
                fraction) + "_mu" + str(
                mu) + "_sigma=" + str(sigma) + ".png")
        vizERDP.plot(
            "report/img/diffusionModels/ourThreshold/diffusionPrevalence/diffusionPrevalenceOurThreshold_fraction=" + str(
                fraction) + "_mu" + str(
                mu) + "_sigma=" + str(sigma) + ".png")


if __name__ == "__main__":
    main()
