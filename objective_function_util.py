import numpy as np

def f(x):
    prod = 1
    for i in range(0, 5):
        prod = prod * x[i] * (1 - x[i])
    return np.sqrt(np.log(prod + 1))

def partial_f(x, j):
    prod_num = 1
    prod_den = 1
    for i in range(0, 5):
        prod_den = prod_den * x[i] * (1 - x[i])
        if i != j:
            prod_num = prod_num * x[i] * (1 - x[i])
    num = prod_num * (1 - 2 * x[j])
    den = 2 * f(x) * prod_den
    return num / den

def gradient_f(x):
    gradient = []
    for i in range(0, 5):
        gradient.append(partial_f(x, i))
    return np.array(gradient)

def hessian_term_f(x, j, k):
    if j != k:
        prod_1 = 1
        prod_2 = 1
        prod_3 = 1
        prod_4 = 1
        for i in range(0, 5):
            prod_1 = prod_1 * x[i] * (1 - x[i])
            if i != j:
                prod_2 = prod_2 * x[i] * (1 - x[i])
            if i != k:
                prod_3 = prod_3 * x[i] * (1 - x[i])
            if i != j and i != k:
                prod_4 = prod_4 * x[i] * (1 - x[i])
        term_1 = -1 * (f(x) ** -2) * partial_f(x, k) * ((prod_1 + 1) ** -1) * prod_2
        term_2 = -1 * (f(x) ** -1) * ((prod_1 + 1) ** -2) * prod_3 * (1 - 2 * x[k])
        term_3 = -2 * (f(x) ** -1) * ((prod_1 + 1) ** -1) * prod_4 * (1 - 2 * x[k])
        return (1 - x[j]) * (term_1 + term_2 + term_3) / 2
    elif j == k:
        prod_1 = 1
        prod_2 = 1
        for i in range(0, 5):
            prod_1 = prod_1 * x[i] * (1 - x[i])
            if i != j:
                prod_2 = prod_2 * x[i] * (1 - x[i])
        term_1 = -1 * (f(x) ** -2) * partial_f(x, j) * ((prod_1 + 1) ** -1) * (1 - 2 * x[j])
        term_2 = -1 * (f(x) ** -1) * ((prod_1 + 1) ** -2) * prod_2 * ((1 - 2 * x[j]) ** 2)
        term_3 = -2 * (f(x) ** -1) * ((prod_1 + 1) ** -1)
        return prod_1 * (term_1 + term_2 + term_3) / 2

def hessian_f(x):
    H = []
    for i in range(0, 5):
        row = []
        for j in range(0, 5):
            hessian_ij = hessian_term_f(x, i, j)
            row.append(hessian_ij)
        H.append(row)
    return H