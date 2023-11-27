import numpy as np
from penalization_fn import p, partial_p, hessian_term_p

def l(x):
    out = 0
    for i in range(0, 5):
        out = out + np.log(x[i] * (1 - x[i]))
    return out + p(x)

def partial_l(x, j):
    return ((1 - 2 * x[j]) / (x[j] * (1 - x[j]))) + partial_p(x, j)

def gradient_l(x):
    gradient = []
    for i in range(0, 5):
        gradient.append(partial_l(x, i))
    return np.array(gradient)

def hessian_term_l(x, j, k):
    if j != k:
        return hessian_term_p(x, j, k)
    elif j == k:
        term_1 = -2 / (x[j] * (1 - x[j]))
        term_2 = -1 * ((1 - x[j])**2) / ((x[j] * (1 - x[j]))**2)
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