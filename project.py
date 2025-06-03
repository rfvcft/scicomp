from collections import defaultdict
from time import time
from typing import Callable
import numpy as np
from numpy import linalg as LA
from scipy.sparse import lil_matrix

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
    
    # Laplacian implemented as sparse matrix
    # Computes self.boundary_points which maps indices n to corresponding boundary points (i, j)
    def sparse_laplacian(self) -> lil_matrix:
        def in_bounds(i, j): 
            return (0 <= i < self.N) and (0 <= j < self.N)
        
        L = lil_matrix((self.N**2, self.N**2)) # sparse zero matrix
        boundary_points = defaultdict(list) # keep track of where boundary values need to land
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for i in range(self.N):
            for j in range(self.N):
                L[self.n(i, j), self.n(i, j)] = -4 
                for dx, dy in directions:
                    i_, j_ = i + dx, j + dy
                    if in_bounds(i_, j_):
                        L[self.n(i, j), self.n(i_, j_)] = 1
                    else:
                        boundary_points[self.n(i, j)].append((i_, j_))
        self.boundary_points = boundary_points
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

        start_time = time()
        self.sparse_laplacian()
        print("Sparse laplacian: ", time() - start_time, "s.")

class TimeStepper:
    
    def __init__(self, N : int, M : int, final_time : int, u_init : np.ndarray, f : Callable[[np.ndarray, float], np.ndarray], g : Callable):
        """
        N - Grid Size
        M - Number of time steps
        final_time - Final time
        u_init - initial condition for u
        f - source term function, depending on x, y and t
        g - boundary value [term/function], should be a function that returns values on the boundary for Dirichlet BC:s
        """
        self.N = N
        self.M = M
        self.T = final_time
        self.u0 = u_init
        self.f = f
        self.g = g

        self.dx = 1 / (N+1)
        self.dt = 1 / (M-1)
        self.g = Grid(self.N)
        self.A = np.eye(N**2) - self.dt * self.g.laplacian()
        self.inner_points = self.generate_inner_points()
        self.boundary_points = self.generate_boundary_points()
    
    def generate_boundary_points(self):
        # TODO: Create this method which returns the points that are on the boundary, in an array which corresponds to where the boundary values need to land 
        # (does this make any sense?, should it be implemented in some other way perhaps?)
        pass

    def generate_inner_points(self) -> np.ndarray:
        inner_points = np.zeros(self.N ** 2)
        for i in range(N):
            for j in range(N):
                # Translate the inner point into array form, but keep the x- and y- values
                inner_points[self.g.n(i, j)] = ((i+1)*self.dx, (j+1)*self.dx)
        return inner_points

    def solve(self, u_old : np.ndarray, t : float) -> np.ndarray:
        # REPLACE WITH OWN IMPLEMENTATION!
        return np.linalg.solve(self.A, self.b(u_old, t))
        
    def b(self, u_old : np.ndarray, t : float) -> np.ndarray:
        """Generates a right hand side for the problem, given a time t, which takes in to consideration the source term and the boundary terms"""
        return u_old + self.f(self.inner_points, t) + self.g(self.boundary_points, t)

    def step(self) -> None:
        times = np.linspace(0, self.T, num=self.M)
        u = self.u0
        for t in times:
            # Visualize here if that is wanted, probably something like this:
            # if visualize:
            #     v.visualize(u)
            # or save all u:s and visualize in the end
            u = self.solve(u, t)


if __name__ == '__main__':
    N = 4
    M = 10
    T = 5
    u0 = np.zeros(N**2)

    def f(xs : np.ndarray, t : float) -> np.ndarray:
        def f_internal(x_ : float, y_: float, t_ : float) -> bool:
            return (x_-0.5)**2 + (y_-0.5)**2 < 0.1
        out = np.zeros(xs.size)
        for n, (x, y) in enumerate(xs):
           if f_internal(x, y, t):
               out[n] = 1

    def g(boundary : np.ndarray, t : float) -> np.ndarray:
        # TODO: Write a reasonable way to make the BC:s work!
        return np.zeros(N**2)

    g = Grid(N)
  
    # np.set_printoptions(threshold=np.inf)
    

    # print(g.laplacian_fast() - g.laplacian())
    # print(np.array_equal(g.laplacian_fast() - g.laplacian(), np.zeros((N**2, N**2), dtype=int)))
    # g.test_speeds()
    # eig_vals, eig_vecs = LA.eig(g.laplacian())
    # sorted_eig_vals = sorted([float(-v) for v in eig_vals])
    # print(sorted_eig_vals[0], sorted_eig_vals[-1], sorted_eig_vals[-1]/sorted_eig_vals[0])


    #ts = TimeStepper(N, M, T, u0, f, g)

    print(g.laplacian())
    print()
    sparse_laplacian = g.sparse_laplacian()
    dense_laplacian = sparse_laplacian.toarray()
    print(dense_laplacian)
    print()
    v = np.zeros(N**2)
    v[1] = 1.0
    w = sparse_laplacian @ v
    print(w)
