import numpy as np

epsilon = 10 ** -6

def Busca_Armijo(f, df, x0, d, gamma=0.8, eta=0.25):
    t = 1
    max_iterations = 500
    count = 0
    while (f(x0 + t*d) > f(x0) + eta * t * np.matmul(np.transpose(df(x0)), d)) and (count < max_iterations):
        t = gamma * t
        count = count + 1
    return t

def Gradiente(f, df, x0, k=1000):
    count = 0
    armijo = 0
    while (np.linalg.norm(df(x0)) > epsilon) and (count < k):
        d = -1 * df(x0)
        gamma = 0.8
        eta = 0.25
        t = Busca_Armijo(f, df, x0, d, gamma, eta)
        armijo = armijo + 1
        for i in range(0, 3):
            if f(x0 + t*d) < f(x0):
                break
            else:
                eta = 0.75 * eta
                t = Busca_Armijo(f, df, x0, d, gamma, eta)
                armijo = armijo + 1
        x0 = x0 + t*d
        count = count + 1
    print("Armijo calls:" + str(armijo))
    return x0

def Newton_1D(f, df, Hf, x0, k=1000):
    for i in range(0, k):
        d = -df(x0)/Hf(x0)
        t = Busca_Armijo(f, df, x0, d)
        x0 = x0 + t*d
    return x0

def Newton_ND(f, df, Hf, x0, k=1000):
    for i in range(0, k):
        # print(f"Hessian: {Hf(x0)}")
        Hf_inv = np.linalg.inv(Hf(x0))
        if (not np.all(np.isfinite(Hf_inv))):
            print("Nan value in matrix Hf_inv")
            return x0
        d = -1 * np.matmul(Hf_inv, df(x0))
        t = Busca_Armijo(f, df, x0, d)
        x0 = x0 + t*d
        if (np.linalg.norm(df(x0+t*d) -df(x0)) / np.linalg.norm(t*d) < epsilon):
                return x0
    return x0

def H_BFGS(H, p, q):
    term_1 = (1 + np.matmul(np.matmul(q, H),np.transpose(q)) / (np.matmul(p,np.transpose(q))))
    term_2 = np.matmul(np.transpose(p), p) / (np.matmul(p, np.transpose(q)))
    num = np.matmul(np.outer(p, q), H) + np.matmul(H, np.outer(q, p))
    den = np.matmul(p, np.transpose(q))
    term_3 = num / den
    return term_1 * term_2 - term_3

def BFGS(f, df, x0, dimensions, k=1000):
    Hf = np.identity(dimensions)
    Hf_inv = np.identity(dimensions)
    p = np.array([[]])
    q = np.array([[]])
    for i in range(0, k):
        if i == 0:
            d = -1 * np.matmul(Hf_inv, df(x0))
            t = Busca_Armijo(f, df, x0, d)
            p = np.concatenate((p, [t*d]), axis=1)
            q = np.concatenate((q, [df(x0 + t*d) - df(x0)]), axis=1)
            x0 = x0 + t*d
        elif i < dimensions:
            d = -1 * np.matmul(Hf_inv, df(x0))
            t = Busca_Armijo(f, df, x0, d)
            p = np.concatenate((p, [t*d]), axis=0)
            q = np.concatenate((q, [df(x0 + t*d) - df(x0)]), axis=0)
            x0 = x0 + t*d
        elif i == dimensions:
            for i in range(0, 10):
                if (np.linalg.det(p) == 0):
                    # p = p + np.random.normal(-0.01*np.linalg.norm(p), 0.01*np.linalg.norm(p), p.shape)
                    p = p + np.random.normal(-0.1, 0.1, p.shape)
                if (np.linalg.det(q) == 0):
                    # q = q + np.random.normal(-0.01*np.linalg.norm(q), 0.01*np.linalg.norm(q), q.shape)
                    q = q + np.random.normal(-0.1, 0.1, q.shape)
            if (np.linalg.det(np.transpose(p)) == 0) or (np.linalg.det(np.transpose(q)) == 0):
                print("Elif.")
                print("Matrix p or q is singular.")
                return x0
            if (not np.all(np.isfinite(p))) or (not np.all(np.isfinite(q))):
                print("Elif.")
                print("Nan value in matrix p or q.")
                print(f"Matrix p: {p}")
                print(f"Matrix q: {q}")
                return x0
            Hf = np.matmul(np.transpose(q), np.linalg.inv(np.transpose(p)))
            if (np.linalg.det(Hf) == 0):
                print("Elif.")
                print("Matrix Hf is singular.")
                return x0
            if (not np.all(np.isfinite(Hf))):
                print("Elif.")
                print("Nan value in matrix Hf.")
                return x0
            if (np.linalg.det(Hf) == 0):
                print("Elif.")
                print("Matrix Hf is singular.")
                return x0
            Hf_inv = np.matmul(np.transpose(p), np.linalg.inv(np.transpose(q)))
            if (not np.all(np.isfinite(Hf_inv))):
                print("Elif.")
                print("Nan value in matrix Hf_inv.")
                return x0
            d = -1 * np.matmul(Hf_inv, df(x0))
            t = Busca_Armijo(f, df, x0, d)
            p = np.concatenate((p, [t*d]), axis=0)
            p = p[1:]
            q = np.concatenate((q, [df(x0 + t*d) - df(x0)]), axis=0)
            q = q[1:]
            x0 = x0 + t*d
        else:
            if (np.linalg.det(np.transpose(p)) == 0) or (np.linalg.det(np.transpose(q)) == 0):
                print("Else.")
                print("Matrix p or q is singular.")
                return x0
            if (not np.all(np.isfinite(p))) or (not np.all(np.isfinite(q))):
                print("Else.")
                print("Nan value in matrix p or q.")
                return x0
            Hf = H_BFGS(Hf, p[-1], q[-1])
            if (not np.all(np.isfinite(Hf))):
                print("Else.")
                print("Nan value in matrix Hf.")
                return x0
            for i in range(0, 10):
                if (np.linalg.det(Hf) == 0):
                    Hf = Hf + np.random.normal(-0.01*np.linalg.norm(Hf), 0.01*np.linalg.norm(Hf), Hf.shape)
            if (np.linalg.det(Hf) == 0):
                print("Else.")
                print("Matrix Hf is singular.")
                return x0
            Hf_inv = np.linalg.inv(Hf)
            if (not np.all(np.isfinite(Hf_inv))):
                print("Else.")
                print("Nan value in matrix Hf_inv.")
                return x0
            d = -1 * np.matmul(Hf_inv, df(x0))
            t = Busca_Armijo(f, df, x0, d)
            p = np.concatenate((p, [t*d]), axis=0)
            p = p[1:]
            q = np.concatenate((q, [df(x0 + t*d) - df(x0)]), axis=0)
            q = q[1:]
            x0 = x0 + t*d
            if (np.linalg.norm(df(x0+t*d) -df(x0)) / np.linalg.norm(t*d) < epsilon):
                return x0
    return x0