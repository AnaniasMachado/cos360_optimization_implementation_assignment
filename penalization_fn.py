import numpy as np

def h(x, exclude=[]):
    out = 1
    for i in range(0, 5):
        if i not in exclude:
            out = out * x[i] * (1 - x[i])
    return out

def p(x):
    return np.square(max(0, -1 * h(x)))

def partial_h(x, j):
    return h(x, exclude=[j]) * (1 - 2 * x[j])

def partial_abs_h(x, j):
    return np.sign(h(x)) * partial_h(x, j)

def partial_p(x, j):
    return 0.5 * max(0, -1 * h(x)) * partial_h(x, j) * (-1 + np.sign(h(x)))

def hessian_term_h(x, j, k):
    if j != k:
        return h(x, exclude=[j, k]) * (1 - 2 * x[j]) * (1 - 2 * x[k])
    elif j == k:
        return -2 * h(x, exclude=[j])

def hessian_term_p(x, j, k):
    term_1 = 0.25 * (partial_abs_h(x, k) - partial_h(x, k)) * partial_h(x, j) * (np.sign(h(x)) - 1)
    term_2 = 0.5 * max(0, -1 * h(x)) * hessian_term_h(x, j, k) * (np.sign(h(x)) - 1)
    term_3 = 0.5 * max(0, -1 * h(x)) * partial_h(x, j) * ((partial_h(x, j) / np.abs(h(x))) - ((h(x)**2) * partial_h(x, k) / ((np.abs(h(x))**3))))
    return term_1 + term_2 + term_3