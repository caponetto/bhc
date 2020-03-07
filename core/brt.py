# -*- coding: utf-8 -*-

# License: GPL 3.0

import numpy as np

from abc import ABC, abstractmethod
from scipy.special import gammaln

from enum import IntEnum
from enum import unique

from core.api import Arc, AbstractBayesianBasedHierarchicalClustering


@unique
class MergeOperation(IntEnum):
    JOIN = 0
    ABSORB_LEFT = 1
    ABSORB_RIGHT = 2
    COLLAPSE = 3


class BayesianRoseTrees(AbstractBayesianBasedHierarchicalClustering):
    """
    Reference: BLUNDELL, Charles; TEH, Yee Whye; HELLER, Katherine A. 
               Bayesian rose trees. 
               arXiv preprint arXiv:1203.3468, 2012.
               https://arxiv.org/pdf/1203.3468.pdf
    """

    def __init__(self, data, model, alpha, cut_allowed):
        super().__init__(data, model, alpha, cut_allowed)

    def build(self):
        n_objects = self.data.shape[0]

        # active nodes
        active_nodes = np.arange(n_objects)
        # assignments - starting each point in its own cluster
        assignments = np.arange(n_objects)
        # stores information from temporary merges
        tmp_merge = None
        hierarchy_cut = False

        # for every single data point
        log_p = np.zeros(n_objects)
        pch = {}
        for i in range(n_objects):
            pch[i] = np.empty(0)
            log_p[i] = self.model.calc_log_mlh(self.data[i])

        # for every pair of data points
        for i in range(n_objects):
            for j in range(i + 1, n_objects):
                # compute log(f_k)
                data_merged = np.vstack((self.data[i], self.data[j]))
                log_f_k = self.model.calc_log_mlh(data_merged)
                # compute log(p_k) with m=join
                sum_log_p_ch = log_p[i] + log_p[j]
                log_p_k = BayesianRoseTrees.calc_log_p(
                    self.alpha, log_f_k, sum_log_p_ch, 2)
                lh_join = log_p_k - sum_log_p_ch
                # store results
                merge_info = [i, j, lh_join, -np.inf, -np.inf, -np.inf]
                tmp_merge = merge_info if tmp_merge is None \
                    else np.vstack((tmp_merge, merge_info))

        # find clusters to merge
        new = n_objects
        while active_nodes.size > 1:
            # find i, j with the highest probability of the merged hypothesis
            max_lh_ratio = np.max(tmp_merge[:, 2:])
            ids_matched = np.argwhere(tmp_merge[:, 2:] == max_lh_ratio)
            if ids_matched.shape[0] > 1:
                ids_matched = ids_matched[0][np.newaxis]
            pos, op = ids_matched.flatten()
            i, j, lh_ratio = tmp_merge[pos, [0, 1, op + 2]]
            i = int(i)
            j = int(j)
            op = MergeOperation(op)

            # cut if required and stop
            if self.cut_allowed is True and lh_ratio < 0:
                hierarchy_cut = True
                break

            # turn nodes i,j off
            tmp_merge[np.argwhere(
                tmp_merge[:, 0] == i).flatten(), 2:] = -np.inf
            tmp_merge[np.argwhere(
                tmp_merge[:, 1] == i).flatten(), 2:] = -np.inf
            tmp_merge[np.argwhere(
                tmp_merge[:, 0] == j).flatten(), 2:] = -np.inf
            tmp_merge[np.argwhere(
                tmp_merge[:, 1] == j).flatten(), 2:] = -np.inf

            # update assignments
            assignments[np.argwhere(assignments == i)] = new
            assignments[np.argwhere(assignments == j)] = new

            # delete i,j from active list
            i_idx = np.argwhere(active_nodes == i).flatten()
            j_idx = np.argwhere(active_nodes == j).flatten()
            active_nodes = np.delete(active_nodes, [i_idx, j_idx])

            # perform the selected operation
            if op == MergeOperation.JOIN:
                pch[new] = np.array([i, j])
            elif op == MergeOperation.ABSORB_LEFT:
                pch[new] = np.append(i, pch[j])
                del pch[j]
            elif op == MergeOperation.ABSORB_RIGHT:
                pch[new] = np.append(pch[i], j)
                del pch[i]
            elif op == MergeOperation.COLLAPSE:
                pch[new] = np.append(pch[i], pch[j])
                del pch[i], pch[j]
            else:
                raise NotImplementedError

            # compute log(p_k)
            sum_log_p_ch = log_p[i] + log_p[j]
            log_p_k = lh_ratio + sum_log_p_ch
            log_p = np.append(log_p, log_p_k)

            X_ij = self.data[np.argwhere(assignments == new).flatten()]

            # for every pair ij x active node
            for node in active_nodes:
                new_ch = pch[new]
                node_ch = pch[node]
                # compute log(f_k)
                node_data = self.data[np.argwhere(
                    assignments == node).flatten()]
                data_merged = np.vstack((X_ij, node_data))
                log_f_k = self.model.calc_log_mlh(data_merged)

                sum_log_p_new_node = log_p[new] + log_p[node]

                # always compute join
                log_p_k = BayesianRoseTrees.calc_log_p(
                    self.alpha, log_f_k, sum_log_p_new_node, 2)
                lh_join = log_p_k - sum_log_p_new_node

                # compute absorb_left if node_ch is an internal node
                lh_absorb_left = -np.inf
                if node_ch.size > 0:
                    n_ch = 1 + node_ch.size
                    sum_log_p_ch = log_p[new] + np.sum(log_p[node_ch])
                    log_p_k = BayesianRoseTrees.calc_log_p(
                        self.alpha, log_f_k, sum_log_p_ch, n_ch)
                    lh_absorb_left = log_p_k - sum_log_p_new_node

                # compute absorb_right if new_ch is an internal node
                lh_absorb_right = -np.inf
                if new_ch.size > 0:
                    n_ch = new_ch.size + 1
                    sum_log_p_ch = np.sum(log_p[new_ch]) + log_p[node]
                    log_p_k = BayesianRoseTrees.calc_log_p(
                        self.alpha, log_f_k, sum_log_p_ch, n_ch)
                    lh_absorb_right = log_p_k - sum_log_p_new_node

                # compute collapse
                lh_collapse = -np.inf
                if new_ch.size > 0 and node_ch.size > 0:
                    n_ch = new_ch.size + node_ch.size
                    sum_log_p_ch = np.sum(
                        log_p[new_ch]) + np.sum(log_p[node_ch])
                    log_p_k = BayesianRoseTrees.calc_log_p(
                        self.alpha, log_f_k, sum_log_p_ch, n_ch)
                    lh_collapse = log_p_k - sum_log_p_new_node

                # store results
                merge_info = [new, node, lh_join,
                              lh_absorb_left, lh_absorb_right, lh_collapse]
                tmp_merge = np.vstack((tmp_merge, merge_info))

            active_nodes = np.append(active_nodes, new)
            new += 1

        # create the arc list
        arc_list = np.empty(0, dtype=Arc)
        for parent in pch.keys():
            for c in pch[parent]:
                arc_list = np.append(arc_list, Arc(parent, c))

        return {
            'arc_list': arc_list,
            'hierarchy_cut': hierarchy_cut,
            'last_log_p': log_p[-1],
            'node_ids': np.array(list(pch.keys())),
            'n_clusters': len(np.unique(assignments))
        }

    @staticmethod
    def calc_log_p(alpha, log_f, sum_log_p_ch, n_ch):
        v = np.maximum(np.finfo(float).eps, (1 - alpha) ** (n_ch - 1))
        log_pi = np.log(1 - v)
        p_t1 = log_pi + log_f
        p_t2 = np.log(-np.expm1(log_pi)) + sum_log_p_ch
        a = np.maximum(p_t1, p_t2)
        b = np.minimum(p_t1, p_t2)
        log_p_k = a + np.log(1 + np.exp(b - a))
        return log_p_k
