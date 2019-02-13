#!/bin/sh

python3 embed.py \
       -dim 2 \
       -lr 0.3 \
       -epochs 200 \
       -negs 50 \
       -burnin 20 \
       -ndproc 4 \
       -manifold poincare \
       -dset wordnet/mammal_closure.csv \
       -checkpoint mammals-2d.pth \
       -batchsize 10 \
       -eval_each 1 \
       -fresh \
       -sparse \
       -train_threads 6
