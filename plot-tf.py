import matplotlib.pyplot as plt
from collections import namedtuple
import re
import tensorflow as tf
tf.enable_eager_execution()

from embed import MANIFOLDS
from hype.sn import initialize
from hype.tf_graph import load_edge_list

plt.style.use('ggplot')


def pplot(names, embeddings, name='mammal'):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.cla()
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))
    ax.add_artist(plt.Circle((0, 0), 1., color='black', fill=False))
    for i, w in enumerate(names):
        c0, c1, *rest = embeddings[i]
        ax.plot(c0, c1, 'o', color='r')
        ax.text(c0 - .1, c1 + .04, re.sub('\.n\.\d{2}', '', w), color='b')
    fig.savefig('plots/' + name + '.png', dpi=fig.dpi)
'''
python3 embed.py \
       -dim 5 \
       -lr 0.3 \
       -epochs 300 \
       -negs 50 \
       -burnin 20 \
       -ndproc 4 \
       -manifold poincare \
       -dset wordnet/mammal_closure.csv \
       -checkpoint mammals.tf \
       -batchsize 2 \
       -eval_each 1 \
       -fresh \
       -sparse \
       -train_threads 2
'''
Opts = namedtuple("Opts", "manifold dim negs")

if __name__ == '__main__':
    opt = Opts('poincare', 5, 50)
    manifold = MANIFOLDS[opt.manifold](debug=False, max_norm=500000)
    idx, objects, weights = load_edge_list('wordnet/mammal_closure.csv', False)
    model, data, model_name, conf = initialize(
        manifold, opt, idx, objects, weights, sparse=False
    )
    chpt = model.load_weights('mammals.tf')
    pplot(objects[:400], model.emb.numpy(), 'tf-mammals')
    plt.show()
