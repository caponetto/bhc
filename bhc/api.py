# -*- coding: utf-8 -*-

# License: GPL 3.0

from abc import ABC, abstractmethod


class Result(object):
    def __init__(self,
                 arc_list,
                 node_ids,
                 last_log_p,
                 weights,
                 hierarchy_cut,
                 n_clusters):
        self.arc_list = arc_list
        self.node_ids = node_ids
        self.last_log_p = last_log_p
        self.weights = weights
        self.hierarchy_cut = hierarchy_cut
        self.n_clusters = n_clusters


class Arc(object):
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def __eq__(self, other):
        return self.source == other.source and self.target == other.target

    def __repr__(self):
        return '{0} -> {1}'.format(str(self.source), str(self.target))


class AbstractPrior(ABC):
    @abstractmethod
    def calc_log_mlh(self, x_mat):
        ...


class AbstractHierarchicalClustering(ABC):
    @abstractmethod
    def build(self):
        ...


class AbstractBayesianBasedHierarchicalClustering(
        AbstractHierarchicalClustering, ABC):
    def __init__(self, data, model, alpha, cut_allowed):
        self.data = data
        self.model = model
        self.alpha = alpha
        self.cut_allowed = cut_allowed
