#!/usr/bin/env python3
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torch.optim.optimizer import Optimizer, required
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf


class RiemannianSGD(Optimizer):
    r"""Riemannian stochastic gradient descent.

    Args:
        rgrad (Function): Function to compute the Riemannian gradient
           from the Euclidean gradient
        retraction (Function): Function to update the retraction
           of the Riemannian gradient
    """

    def __init__(
            self,
            params,
            lr=required,
            rgrad=required,
            expm=required,
    ):
        defaults = {
            'lr': lr,
            'rgrad': rgrad,
            'expm': expm,
        }
        super(RiemannianSGD, self).__init__(params, defaults)

    def step(self, lr=None, counts=None, **kwargs):
        """Performs a single optimization step.

        Arguments:
            lr (float, optional): learning rate for the current update.
        """
        loss = None

        for group in self.param_groups:
            for p in group['params']:
                lr = lr or group['lr']
                rgrad = group['rgrad']
                expm = group['expm']

                if p.grad is None:
                    continue
                d_p = p.grad.data
                # make sure we have no duplicates in sparse tensor
                if d_p.is_sparse:
                    d_p = d_p.coalesce()
                d_p = rgrad(p.data, d_p)
                d_p.mul_(-lr)
                expm(p.data, d_p)

        return loss


class RSGDTF(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, use_locking=False, name="RSGDTF", rgrad=None, expm=None):
        super(RSGDTF, self).__init__(use_locking, name)
        self.rgrad = rgrad
        self.expm = expm
        self._lr = learning_rate

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        d_p = self.rgrad(var, grad)
        return state_ops.assign_sub(var, self.expm(var, d_p, lr=lr_t))

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")
