import numpy as np
from penalization_fn import p, partial_p, hessian_term_p

def h(x, exclude=[]):
    out = 1
    for i in range(0, 5):
        if i not in exclude:
            out = out * x[i] * (1 - x[i])
    return out + 1

def l(x):
    return np.log(np.sqrt(h(x))) + p(x)

def partial_h(x, j):
    return h(x, exclude=[j]) * (1 - 2 * x[j])

def partial_l(x, j):
    return (partial_h(x, j) / (2 * h(x))) + partial_p(x, j)

def gradient_l(x):
    gradient = []
    for i in range(0, 5):
        gradient.append(partial_l(x, i))
    return np.array(gradient)

def hessian_term_h(x, j, k):
    if j != k:
        return h(x, exclude=[j, k]) * (1 - 2 * x[j]) * (1 - 2 * x[k])
    elif j == k:
        return -2 * h(x, exclude=[j])

def hessian_term_l(x, j, k):
    term_1 = partial_h(x, j) * partial_h(x, k) / (2 * np.square(h(x)))
    term_2 = hessian_term_h(x, j, k) / (2 * h(x))
    return term_1 + term_2 + hessian_term_p(x, j, k)

def hessian_l(x):
    H = []
    for i in range(0, 5):
        row = []
        for j in range(0, 5):
            hessian_ij = hessian_term_l(x, i, j)
            row.append(hessian_ij)
        H.append(row)
    return H