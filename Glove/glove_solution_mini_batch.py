#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    print("loading cooccurrence matrix")
    cooc_path = os.path.join(SCRIPT_DIR, "cooc.pkl")
    with open(cooc_path, "rb") as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    nmax = 100
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    embedding_dim = 75
    scale = 0.01
    xs = np.random.normal(scale=scale, size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(scale=scale, size=(cooc.shape[1], embedding_dim))

    # eta = 0.001
    eta_base = 0.01
    alpha = 3 / 4

    epochs = 30
    batch_size = 128
    nnz = cooc.nnz

    for epoch in range(epochs):
        print("epoch {}".format(epoch))

        eta = eta_base / (1 + 0.1 * epoch)   # simple decay
        idxs = np.random.permutation(nnz)

        for start in range(0, nnz, batch_size):
            batch_idx = idxs[start:start + batch_size]

            ix = cooc.row[batch_idx]
            jy = cooc.col[batch_idx]
            n  = cooc.data[batch_idx].astype(float)

            logn = np.log(n)
            fn = np.minimum(1.0, (n / nmax) ** alpha)

            x = xs[ix, :]        # shape: (B, d)
            y = ys[jy, :]        # shape: (B, d)

            dot = np.sum(x * y, axis=1)  # shape: (B,)

            scale_vec = 2.0 * eta * fn * (logn - dot)

            gx = scale_vec[:, None] * y
            gy = scale_vec[:, None] * x

            np.add.at(xs, ix, gx)
            np.add.at(ys, jy, gy)

    embeddings = xs + ys
    embeddings_path = os.path.join(SCRIPT_DIR, "embeddings.npy")
    np.save(embeddings_path, embeddings)

if __name__ == "__main__":
    main()
