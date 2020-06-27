# -*- coding: utf-8 -*-

# License: GPL 3.0

import sys

import numpy as np
import matplotlib.pyplot as plt

from graphviz import Digraph
from scipy.cluster.hierarchy import dendrogram, linkage

from bhc.core.bhc import BayesianHierarchicalClustering
from bhc.core.brt import BayesianRoseTrees
from bhc.core.prior import NormalInverseWishart


def main():
    data = np.genfromtxt('data/data.csv', delimiter=',')

    plot_data(data)

    run_linkage(data, 'single')
    run_linkage(data, 'complete')
    run_linkage(data, 'average')

    run_bhc(data)
    run_brt(data)


def plot_data(data):
    plt.style.use('seaborn-poster')
    plt.figure(facecolor="white", figsize=(6, 4))
    ax = plt.gca()
    ax.set_axisbelow(True)

    ax.grid(True, color='lightgrey', linestyle='-', alpha=0.4)
    ax.tick_params(axis='both', which='both', length=0, labelcolor='0.5')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    for i in range(data.shape[0]):
        if i < 10:
            plt.scatter(data[i, 0], data[i, 1], s=200, color='#1F77B4',
                        marker=r"$ {} $".format(str(i)))
        else:
            plt.scatter(data[i, 0], data[i, 1], s=200, color='#FF7F0E',
                        marker=r"$ {} $".format(str(i)))

    plt.xlabel('x', fontsize=18, weight='light', color='0.35')
    plt.ylabel('y', fontsize=18, weight='light', color='0.35')

    plt.xticks(np.arange(0, 8 + 1, 1), fontsize=14)
    plt.yticks(np.arange(0, 11 + 1, 1), fontsize=14)

    plt.draw()
    plt.savefig('results/data_plot.png', format='png', dpi=100)


def run_linkage(data, method):
    plt.clf()
    Z = linkage(data, method)
    dn = dendrogram(Z)
    plt.draw()
    plt.savefig(
        'results/linkage_{0}_plot.png'.format(method), format='png', dpi=100)


def run_bhc(data):
    # Hyper-parameters (these values must be optimized!)
    g = 20
    scalling_factor = 0.001
    alpha = 1

    model = create_model(data, g, scalling_factor)

    bhc = BayesianHierarchicalClustering(data,
                                         model,
                                         alpha,
                                         cut_allowed=True)

    result = bhc.build()

    build_graph(result['node_ids'],
                result['arc_list'],
                'results/bhc_plot')


def run_brt(data):
    # Hyper-parameters (these values must be optimized!)
    g = 10
    scalling_factor = 0.001
    alpha = 0.5

    model = create_model(data, g, scalling_factor)

    brt = BayesianRoseTrees(data,
                            model,
                            alpha,
                            cut_allowed=True)

    result = brt.build()

    build_graph(result['node_ids'],
                result['arc_list'],
                'results/brt_plot')


def create_model(data, g, scalling_factor):
    degrees_of_freedom = data.shape[1] + 1
    data_mean = np.mean(data, axis=0)
    data_matrix_cov = np.cov(data.T)
    scatter_matrix = (data_matrix_cov / g).T

    return NormalInverseWishart(scatter_matrix,
                                scalling_factor,
                                degrees_of_freedom,
                                data_mean)


def build_graph(node_ids, arc_list, filename):
    dag = Digraph()

    for id in node_ids:
        dag.node(str(id))

    for arc in arc_list:
        dag.edge(str(arc.source), str(arc.target))

    dag.render(filename=filename, format='png', cleanup=True)


if __name__ == "__main__":
    main()
    sys.exit()
