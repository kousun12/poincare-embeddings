#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tensorflow as tf
from .euclidean import EuclideanManifold


class PoincareManifold(EuclideanManifold):
    def __init__(self, eps=1e-5, **kwargs):
        super(PoincareManifold, self).__init__(**kwargs)
        self.eps = eps
        self.boundary = 1 - eps
        self.max_norm = self.boundary

    def distance(self, u, v):
        return Distance.forward(u, v, self.eps)

    def rgrad(self, p, d_p):
        p_sqnorm = tf.reduce_sum(p ** 2)
        return d_p * ((1 - p_sqnorm) ** 2 / 4)


class Distance:
    @staticmethod
    def grad(x, v, sqnormx, sqnormv, sqdist, eps):
        alpha = (1 - sqnormx)
        beta = (1 - sqnormv)
        z = 1 + 2 * sqdist / (alpha * beta)
        a = ((sqnormv - 2 * tf.reduce_sum(x * v, dim=-1) + 1) / tf.pow(alpha, 2))
        a = a * x - v / alpha
        z = tf.sqrt(tf.pow(z, 2) - 1)
        z = tf.clip_by_value(z * beta, clip_value_min=eps)
        return 4 * a / z

    @staticmethod
    def forward(u, v, eps):
        squnorm = tf.clip_by_value(tf.reduce_sum(u * u, axis=-1), clip_value_min=0, clip_value_max=1 - eps)
        sqvnorm = tf.clip_by_value(tf.reduce_sum(v * v, axis=-1), clip_value_min=0, clip_value_max=1 - eps)
        sqdist = tf.reduce_sum(tf.pow(u - v, 2), axis=-1)
        x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1
        z = tf.sqrt(tf.pow(x, 2) - 1)
        return tf.log(x + z)

