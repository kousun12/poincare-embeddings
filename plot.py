import matplotlib.pyplot as plt
import re
import torch

plt.style.use('ggplot')


def pplot(names, embeddings, name='mammal'):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.cla()
    ax.set_xlim((-1.1, 1.1))
    ax.set_ylim((-1.1, 1.1))
    ax.add_artist(plt.Circle((0, 0), 1., color='black', fill=False))
    for i, w in enumerate(names):
        c0, c1 = [e.item() for e in embeddings[i]]
        ax.plot(c0, c1, 'o', color='r')
        ax.text(c0 - .1, c1 + .04, re.sub('\.n\.\d{2}', '', w), color='b')
    fig.savefig('plots/' + name + '.png', dpi=fig.dpi)


if __name__ == '__main__':
    checkpoint = torch.load('mammals-2d.pth')
    import ipdb; ipdb.set_trace()
    pplot(checkpoint['objects'][:400], checkpoint['embeddings'], 'mammals')
    plt.show()
