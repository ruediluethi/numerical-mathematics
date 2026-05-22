import numpy as np

def PCA(A, d=2):
    n = A.shape[0]
    # d = A.shape[1]
    ATA = 1/n * A.T @ A
    lambdas, V = np.linalg.eig(ATA)

    # project the data onto the new 2D basis
    A_d = np.zeros((n,d))
    for i in range(0,d):
        A_d[:,i] = A @ V[:,i]

    return A_d