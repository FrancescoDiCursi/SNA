import networkx as nx
from data_collection.dataHandling import *
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
from bokeh.io import output_notebook, show
from ndlib.viz.bokeh.MultiPlot import MultiPlot
import pandas as pd
from bokeh.io import export_png
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
from ndlib.viz.mpl.DiffusionPrevalence import DiffusionPrevalence


def main():
    dfMentions = pd.read_json('data_collection/data/df_mentions.json', orient='records', lines=True)

    #I changed this code for each model: SI, SIR, SIS. It works, but it has to be changed depending on which model you choose
    #(and on which graph you're running it)

    # create minimal dataset
    dfMentions3c = dfMentions[['username', 'screen_name', 'date']]
    g = createGraph(dfMentions3c)
    numberOfNodes = g.number_of_nodes()
    numberOfEdges = g.number_of_edges()
    gER = nx.gnm_random_graph(n=numberOfNodes, m=numberOfEdges, directed=False)
    gBA = nx.barabasi_albert_graph(n=numberOfNodes, m=2)


    betas = list(range(1, 11))
    betas = [i / 10 for i in betas]

    mus = list(range(1, 11))
    mus = [i / 10 for i in mus]

    fractions = list(range(1, 11))
    fractions = [i / 100 for i in fractions]

    for beta in betas:
        for fraction in fractions:
            for mu in mus:
                modelER = ep.SIRModel(g)
                config = mc.Configuration()
                config.add_model_parameter('beta', beta)
                config.add_model_parameter('gamma', mu)
                config.add_model_parameter("fraction_infected", fraction)
                modelER.set_initial_status(config)

                iterationsER = modelER.iteration_bunch(200)
                trendsER = modelER.build_trends(iterationsER)
                vizER = DiffusionTrend(modelER, trendsER)
                vizERDP = DiffusionPrevalence(modelER, trendsER)

                vizER.plot(
                    "report/img/diffusionModels/ourSIR/diffusionTrend/diffusionOurSIR_beta=" + str(beta) +"_mu" + str(mu)+"_frac=" + str(
                        fraction) + ".png")
                vizERDP.plot("report/img/diffusionModels/ourSIR/diffusionPrevalence/diffusionPrevalenceOurSIR_beta=" + str(
                    beta) +"_mu" + str(mu) + "_frac=" + str(fraction) + ".png")

if __name__ == "__main__":
    main()
