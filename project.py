from time import time
import numpy as np
from numpy import linalg as LA

class Grid:
    """
    Grid class which given a grid-size N can produce the corresponding 2D-Laplacian for the mapping (i,j) -> n = j + N*i
    """
    def __init__(self, N : int):
        self.N = N
        self.dx = 1 / (N+1)
    
    def n(self, i : int, j : int) -> int:
        return self.N*i + j

    # "Intuitive" creation of Laplacian
    def slow_laplacian(self) -> np.matrix:
        L = np.zeros((self.N**2, self.N**2), dtype=int)
        for i in range(self.N):
            for j in range(self.N):
                L[self.n(i,j)][self.n(i,j)] = -4
                if i != 0:
                    L[self.n(i,j)][self.n(i-1,j)] = 1
                if i != self.N-1:
                    L[self.n(i,j)][self.n(i+1,j)] = 1
                if j != 0:
                    L[self.n(i,j)][self.n(i,j-1)] = 1
                if j != self.N-1:
                    L[self.n(i,j)][self.n(i,j+1)] = 1
        return L
    
    # Creation of Laplacian using diagonals
    def laplacian(self) -> np.matrix:
        a = -4*np.ones(self.N**2, dtype=int)
        b = np.ones(self.N**2 - 1, dtype=int)
        c = np.ones(self.N**2 - self.N, dtype=int)
        L = np.diag(a, 0) + np.diag(b, 1) + np.diag(b, -1) + np.diag(c, self.N) + np.diag(c, -self.N)

        for i in range(1, self.N):
            L[i*self.N][i*self.N - 1] = 0
            L[i*self.N - 1][i*self.N] = 0

        return L
    
    def test_speeds(self):
        # Somehow, the "slow" laplacian is a lot faster than the "fast" laplacian?? I don't get it, but sure...
        start_time = time()
        self.slow_laplacian()
        print("Slow laplacian: ", time() - start_time, "s.")

        start_time = time()
        self.laplacian()
        print("Fast laplacian: ", time() - start_time, "s.")


if __name__ == '__main__':
    N = 30
    g = Grid(N)
    # np.set_printoptions(threshold=np.inf)
    # print(g.laplacian_fast())
    # print(g.laplacian())
    # print(g.laplacian_fast() - g.laplacian())
    # print(np.array_equal(g.laplacian_fast() - g.laplacian(), np.zeros((N**2, N**2), dtype=int)))
    # g.test_speeds()
    eig_vals, eig_vecs = LA.eig(g.laplacian())
    sorted_eig_vals = sorted([float(-v) for v in eig_vals])
    print(sorted_eig_vals[0], sorted_eig_vals[-1], sorted_eig_vals[-1]/sorted_eig_vals[0])
