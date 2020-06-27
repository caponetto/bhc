# -*- coding: utf-8 -*-

# License: GPL 3.0

from abc import ABC, abstractmethod


class Arc(object):
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def __repr__(self):
        return '{0} -> {1}'.format(str(self.source), str(self.target))


class AbstractPrior(ABC):
    @abstractmethod
    def calc_log_mlh(self, X):
        ...


class AbstractHierarchicalClustering(ABC):
    @abstractmethod
    def build(self):
        ...


class AbstractBayesianBasedHierarchicalClustering(AbstractHierarchicalClustering, ABC):
    def __init__(self, data, model, alpha, cut_allowed):
        self.data = data
        self.model = model
        self.alpha = alpha
        self.cut_allowed = cut_allowed
