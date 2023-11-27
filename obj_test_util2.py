import numpy as np

def prod(x, exclude):
    out = 1
    for i in range(0, 5):
        if i not in exclude:
            out = out * x[i] * (1 - x[i])
    return out

def g(x):
    return prod(x, exclude=[]) + 1

def f(x):
    return np.exp(g(x))

def partial_g(x, j):
    return (1 - 2 * x[j]) * prod(x, exclude=[j])

def partial_f(x, j):
    return g(x) * f(x) * partial_g(x, j)

def gradient_f(x):
    gradient = []
    for i in range(0, 5):
        gradient.append(partial_f(x, i))
    return np.array(gradient)

def hessian_term_g(x, j, k):
    if j != k:
        return (1 - 2 * x[j]) * (1 - 2 * x[k]) * prod(x, exclude=[j, k])
    elif j == k:
        return -2 * prod(x, exclude=[j])

def hessian_term_f(x, j, k):
    if j != k:
        term_1 = partial_g(x, k) * f(x) * partial_g(x, j)
        term_2 = g(x) * partial_f(x, k) * partial_g(x, j)
        term_3 = g(x) * f(x) * hessian_term_g(x, j, k)
        return term_1 + term_2 + term_3
    elif j == k:
        term_1 = partial_g(x, j) * f(x) * partial_g(x, j)
        term_2 = g(x) * partial_f(x, j) * partial_g(x, j)
        term_3 = g(x) * f(x) * hessian_term_g(x, j, j)
        return term_1 + term_2 + term_3

def hessian_f(x):
    H = []
    for i in range(0, 5):
        row = []
        for j in range(0, 5):
            hessian_ij = hessian_term_f(x, i, j)
            row.append(hessian_ij)
        H.append(row)
    return H