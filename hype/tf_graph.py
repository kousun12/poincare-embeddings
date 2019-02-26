#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from collections import defaultdict as ddict
from numpy.random import choice
import pandas
import h5py
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops


def load_adjacency_matrix(path, format="hdf5", symmetrize=False):
    if format == "hdf5":
        with h5py.File(path, "r") as hf:
            return {
                "ids": hf["ids"].value.astype("int"),
                "neighbors": hf["neighbors"].value.astype("int"),
                "offsets": hf["offsets"].value.astype("int"),
                "weights": hf["weights"].value.astype("float"),
                "objects": hf["objects"].value,
            }
    elif format == "csv":
        df = pandas.read_csv(path, usecols=["id1", "id2", "weight"], engine="c")
        if symmetrize:
            rev = df.copy().rename(columns={"id1": "id2", "id2": "id1"})
            df = pandas.concat([df, rev])

        idmap = {}
        idlist = []

        # todo: we should be able to use tf.embedding_lookup
        def convert(id):
            if id not in idmap:
                idmap[id] = len(idlist)
                idlist.append(id)
            return idmap[id]

        df.loc[:, "id1"] = df["id1"].apply(convert)
        df.loc[:, "id2"] = df["id2"].apply(convert)

        groups = df.groupby("id1").apply(lambda x: x.sort_values(by="id2"))
        counts = df.groupby("id1").id2.size()

        ids = groups.index.levels[0].values
        offsets = counts.loc[ids].values
        offsets[1:] = np.cumsum(offsets)[:-1]
        offsets[0] = 0
        neighbors = groups["id2"].values
        weights = groups["weight"].values
        return {
            "ids": ids.astype("int"),
            "offsets": offsets.astype("int"),
            "neighbors": neighbors.astype("int"),
            "weights": weights.astype("float"),
            "objects": np.array(idlist),
        }
    else:
        raise RuntimeError(f"Unsupported file format {format}")


def load_edge_list(path, symmetrize=False):
    df = pandas.read_csv(path, usecols=["id1", "id2", "weight"], engine="c")
    df.dropna(inplace=True)
    if symmetrize:
        rev = df.copy().rename(columns={"id1": "id2", "id2": "id1"})
        df = pandas.concat([df, rev])
    idx, objects = pandas.factorize(df[["id1", "id2"]].values.reshape(-1))
    idx = idx.reshape(-1, 2).astype("int")
    weights = df.weight.values.astype("float")
    return idx, objects.tolist(), weights


def _apply_dense(inputs, model, grad, var, lr_t):
    # idx_batch = tf.gather(inputs, [0], axis=1)
    # idx = tf.reshape(idx_batch, [idx_batch.shape[0]])
    # import ipdb; ipdb.set_trace()
    # _msk = tf.tensor_scatter_add(-tf.ones_like(grad, grad.dtype), idx_batch, tf.ones_like(idx, dtype=grad.dtype))
    # indices = tf.constant([[0], [2]])
    # updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
    #                         [7, 7, 7, 7], [8, 8, 8, 8]],
    #                        [[5, 5, 5, 5], [6, 6, 6, 6],
    #                         [7, 7, 7, 7], [8, 8, 8, 8]]])
    # shape = tf.constant([4, 4, 4])

    # mask = tf.Variable(tf.ones_like(grad, dtype=grad.dtype))
    # _mmsk = tf.scatter_nd(idx, tf.tile(tf.ones_like(mask[0])), mask.shape)
    # mask = tf.assign(mask[idx], tf.zeros_like(mask[idx]))
    # grad = grad * mask
    d_p = model.manifold.rgrad(var, grad)
    update = model.manifold.expm(var, d_p, lr=lr_t)
    return var.assign(update)
    # return state_ops.assign_sub(var, update)


def train(model, inputs, outputs, learning_rate=tf.constant(0.1)):
    with tf.GradientTape() as t:
        t.watch([model.emb])
        pred = model(inputs)
        _loss = model.loss(pred, outputs)
    var_list = t.watched_variables()
    dEmb, *rest = t.gradient(_loss, var_list, None)
    _apply_dense(inputs, model, dEmb, model.emb, learning_rate)
    # print(f'loss: {_loss}')
    return _loss

    # loss, var_list = var_list
    # optimizer.compute_gradients(_loss, [model.emb])

    # with tf.GradientTape() as t:
    #     _loss = loss(model(inputs), outputs)
    # dEmb = t.gradient(_loss, [model.emb])
    # model.emb.assign_sub(learning_rate * dEmb)


class Embedding(tf.keras.Model):
    def __init__(self, size, dim, manifold, sparse=True):
        super(Embedding, self).__init__()
        self.dim = dim
        self.sparse = sparse
        self.nobjects = size
        self.manifold = manifold
        self.dist = manifold.distance
        self.pre_hook = None
        self.post_hook = None
        eps = 1e-6
        self.offset = tf.constant(eps, dtype=tf.float64)
        scale = 1e-4
        self.emb = tf.Variable(
            tf.random_uniform([size, dim], -scale, scale, dtype=tf.float64), name="emb"
        )

    def _forward(self, x):
        raise NotImplementedError()

    def loss(self, actual, expected):
        # tf.losses.softmax_cross_entropy(expected, actual)
        _exp = expected + self.offset
        return tf.reduce_mean(-tf.reduce_sum(tf.reshape(_exp, [actual.shape[0], 1]) * tf.log(actual), axis=-1))
        # tf.nn.softmax_cross_entropy_with_logits(labels=expected, logits=actual)
        # return tf.reduce_mean(tf.square(tf.cast(expected, dtype='float32') - actual))

    def call(self, inputs, training=False, **kwargs):
        # e = self.manifold.normalize(inputs)
        if self.pre_hook is not None:
            inputs = self.pre_hook(inputs)
        return self._forward(inputs)

    def embedding(self):
        return self.emb

    def build_graph(self):
        # Properly initialize all variables.
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()


# # This class is now deprecated in favor of BatchedDataset (graph_dataset.pyx)
class Dataset(object):
    _neg_multiplier = 1
    _ntries = 10
    _sample_dampening = 0.75

    def __init__(self, idx, objects, weights, nnegs, unigram_size=1e8):
        assert idx.ndim == 2 and idx.shape[1] == 2
        assert weights.ndim == 1
        assert len(idx) == len(weights)
        assert nnegs >= 0
        assert unigram_size >= 0

        print("Indexing data")
        self.idx = idx
        self.nnegs = nnegs
        self.burnin = False
        self.objects = objects

        self._weights = ddict(lambda: ddict(int))
        self._counts = np.ones(len(objects), dtype=np.float)
        self.max_tries = self.nnegs * self._ntries
        for i in range(idx.shape[0]):
            t, h = self.idx[i]
            self._counts[h] += weights[i]
            self._weights[t][h] += weights[i]
        self._weights = dict(self._weights)
        nents = int(np.array(list(self._weights.keys())).max())
        assert len(objects) > nents, f"Number of objects do no match"

        if unigram_size > 0:
            c = self._counts ** self._sample_dampening
            self.unigram_table = choice(
                len(objects), size=int(unigram_size), p=(c / c.sum())
            )

    def __len__(self):
        return self.idx.shape[0]

    def weights(self, inputs, targets):
        return self.fweights(self, inputs, targets)

    def nnegatives(self):
        if self.burnin:
            return self._neg_multiplier * self.nnegs
        else:
            return self.nnegs


# # This function is now deprecated in favor of eval_reconstruction
# def eval_reconstruction_slow(adj, lt, distfn):
#     ranks = []
#     ap_scores = []
#
#     for s, s_types in adj.items():
#         s_e = lt[s].expand_as(lt)
#         _dists = distfn(s_e, lt).data.cpu().numpy().flatten()
#         _dists[s] = 1e+12
#         _labels = np.zeros(lt.size(0))
#         _dists_masked = _dists.copy()
#         _ranks = []
#         for o in s_types:
#             _dists_masked[o] = np.Inf
#             _labels[o] = 1
#         for o in s_types:
#             d = _dists_masked.copy()
#             d[o] = _dists[o]
#             r = np.argsort(d)
#             _ranks.append(np.where(r == o)[0][0] + 1)
#         ranks += _ranks
#         ap_scores.append(
#             average_precision_score(_labels, -_dists)
#         )
#     return np.mean(ranks), np.mean(ap_scores)
#
#
# def reconstruction_worker(adj, lt, distfn, objects, progress=False):
#     ranksum = nranks = ap_scores = iters = 0
#     labels = np.empty(lt.size(0))
#     for object in tqdm(objects) if progress else objects:
#         labels.fill(0)
#         neighbors = np.array(list(adj[object]))
#         dists = distfn(lt[None, object], lt)
#         dists[object] = 1e12
#         sorted_dists, sorted_idx = dists.sort()
#         ranks, = np.where(np.in1d(sorted_idx.cpu().numpy(), neighbors))
#         # The above gives us the position of the neighbors in sorted order.  We
#         # want to count the number of non-neighbors that occur before each neighbor
#         ranks += 1
#         N = ranks.shape[0]
#
#         # To account for other positive nearer neighbors, we subtract (N*(N+1)/2)
#         # As an example, assume the ranks of the neighbors are:
#         # 0, 1, 4, 5, 6, 8
#         # For each neighbor, we'd like to return the number of non-neighbors
#         # that ranked higher than it.  In this case, we'd return 0+0+2+2+2+3=14
#         # Another way of thinking about it is to return
#         # 0 + 1 + 4 + 5 + 6 + 8 - (0 + 1 + 2 + 3 + 4 + 5)
#         # (0 + 1 + 2 + ... + N) == (N * (N + 1) / 2)
#         # Note that we include `N` to account for the source embedding itself
#         # always being the nearest neighbor
#         ranksum += ranks.sum() - (N * (N - 1) / 2)
#         nranks += ranks.shape[0]
#         labels[neighbors] = 1
#         ap_scores += average_precision_score(labels, -dists.cpu().numpy())
#         iters += 1
#     return float(ranksum), nranks, ap_scores, iters
#
#
# def eval_reconstruction(adj, lt, distfn, workers=1, progress=False):
#     '''
#     Reconstruction evaluation.  For each object, rank its neighbors by distance
#
#     Args:
#         adj (dict[int, set[int]]): Adjacency list mapping objects to its neighbors
#         lt (torch.Tensor[N, dim]): Embedding table with `N` embeddings and `dim`
#             dimensionality
#         distfn ((torch.Tensor, torch.Tensor) -> torch.Tensor): distance function.
#         workers (int): number of workers to use
#     '''
#     objects = np.array(list(adj.keys()))
#     if workers > 1:
#         with ThreadPool(workers) as pool:
#             f = partial(reconstruction_worker, adj, lt, distfn)
#             results = pool.map(f, np.array_split(objects, workers))
#             results = np.array(results).sum(axis=0).astype(float)
#     else:
#         results = reconstruction_worker(adj, lt, distfn, objects, progress)
#     return float(results[0]) / results[1], float(results[2]) / results[3]
