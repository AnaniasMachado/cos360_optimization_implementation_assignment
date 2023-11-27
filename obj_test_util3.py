import numpy as np
from penalization_fn import p, partial_p, hessian_term_p

def prod(x, exclude):
    out = 1
    for i in range(0, 5):
        if i not in exclude:
            out = out * x[i] * (1 - x[i])
    return out

def f(x):
    return prod(x, exclude=[]) + 1 + p(x)

def partial_f(x, j):
    return (1 - 2 * x[j]) * prod(x, exclude=[j]) + partial_p(x, j)

def gradient_f(x):
    gradient = []
    for i in range(0, 5):
        gradient.append(partial_f(x, i))
    return np.array(gradient)

def hessian_term_f(x, j, k):
    if j != k:
        return (1 - 2 * x[j]) * (1 - 2 * x[k]) * prod(x, exclude=[j, k])
    elif j == k:
        return -2 * prod(x, exclude=[j])

def hessian_f(x):
    H = []
    for i in range(0, 5):
        row = []
        for j in range(0, 5):
            hessian_ij = hessian_term_f(x, i, j) + hessian_term_p(x, i, j)
            row.append(hessian_ij)
        H.append(row)
    return H