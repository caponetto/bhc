# -*- coding: utf-8 -*-

# License: GPL 3.0

from bhc.core.bhc import BayesianHierarchicalClustering
from bhc.core.brt import BayesianRoseTrees
from bhc.core.prior import NormalInverseWishart

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions
