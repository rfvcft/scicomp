from collections import defaultdict
from time import time
from typing import Callable, Optional
import numpy as np
from numpy import linalg as LA
from scipy.sparse import lil_matrix, csr_matrix, identity
from matplotlib import pyplot as plt
from matplotlib import animation
import cg

class Grid:
    """
    Grid class which given a grid-size N can produce the corresponding 2D-Laplacian for the mapping (i,j) -> n = j + N*i
    """
    def __init__(self, N : int):
        self.N = N
        self.dx = 1 / (N+1)
        self.boundary_points = defaultdict(list)
    
    def n(self, i : int, j : int) -> int:
        return self.N * i + j
    
    def inverse_n(self, n: int) -> tuple[int, int]:
        return n // self.N, n % self.N

    def to_grid(self, u: np.ndarray) -> np.matrix:
        """ Takes a u-vector, and returns grid of u-values """
        u_grid  = np.zeros((self.N, self.N))
        for n, u_n in enumerate(u):
            u_grid[self.inverse_n(n)] = u_n
        return u_grid

    def generate_inner_points_components(self) -> tuple[np.ndarray, np.ndarray]:
        ''' Generates x_1, x_2 coordinates for inner points. '''
        x_1 = np.zeros(self.N**2)
        x_2 = np.zeros(self.N**2)
        for i in range(self.N):
            for j in range(self.N):
                n = self.n(i, j)
                x_1[n] = (i + 1) * self.dx
                x_2[n] = (j + 1) * self.dx
        return x_1, x_2
    

    def sparse_laplacian(self) -> csr_matrix:
        '''
        Laplacian implemented as sparse matrix
        Computes self.boundary_points which maps indices n to corresponding boundary points 
        '''
        def in_bounds(i, j): 
            return (0 <= i < self.N) and (0 <= j < self.N)
        
        L = lil_matrix((self.N**2, self.N**2)) # list-sparse matrix
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
                        boundary_points[self.n(i, j)].append(((i_+1)*self.dx, (j_+1)*self.dx)) 
        self.boundary_points = boundary_points
        return (1/self.dx)**2 * csr_matrix(L) # convert to sparse-row matrix
    
    def sparse_identity(self) -> csr_matrix:
        return identity(self.N**2, format="csr")


class TimeStepper:

    def __init__(
            self, 
            N : int, 
            M : int, 
            final_time : int, 
            u_init : Callable[[np.ndarray, np.ndarray], np.ndarray], 
            f : Callable[[np.ndarray, np.ndarray, float], np.ndarray], 
            g : Callable[[np.ndarray, np.ndarray, float], np.ndarray],
            preconditioned: Optional[bool]=False,
        ):
        """
        Class for solving the heat equation.

        N - Grid Size
        M - Number of time steps
        final_time - Final time
        u_init - initial condition for u, depending on x_1, x_2
        f - source term function, depending on x_1, x_2 and t
        g - boundary value function, depending on x_1, x_2 and t
        preconditioned - use a preconditioner in the solver
        """
        self.N = N
        self.M = M
        self.T = final_time
        self.u_init = u_init
        self.f = f  
        self.g = g

        self.dx = 1 / (N+1)
        self.dt = self.T / (M-1) 
        self.grid = Grid(self.N)
        self.laplacian = self.grid.sparse_laplacian()
        self.A = self.grid.sparse_identity() - self.dt * self.laplacian
        self.x_1, self.x_2 = self.grid.generate_inner_points_components()
        self.u0 = self.evaluate_u_init()
        self.solver = cg.Solver(self.A, preconditioned=preconditioned)

    def solve(self, u_old : np.ndarray, t : float, testing=False) -> tuple[np.ndarray, int]:
        result = self.solver.solve(self.b(u_old, t), initial_guess=u_old, testing=testing)
        x = result[0]
        if x is not None:
            return result
        else:
            raise RuntimeError("CG-method did not converge.")

        
    def b(self, u_old : np.ndarray, t : float) -> np.ndarray:
        """Generates a right hand side for the problem, given a time t, which takes into consideration the source term and the boundary terms"""
        return u_old + self.dt * self.evaluate_f(t + self.dt) + self.dt * (1 / self.dx)**2 * self.evaluate_g(t + self.dt)
    
    def evaluate_u_init(self) -> np.ndarray:
        return self.u_init(self.x_1, self.x_2)

    def evaluate_f(self, t) -> np.ndarray:
        return self.f(self.x_1, self.x_2, t)
    
    def evaluate_g(self, t) -> np.ndarray:
        evaluated_g = np.zeros(self.N**2)
        for n, boundary_points in self.grid.boundary_points.items():
            evaluated_g[n] = sum(self.g(x_1, x_2, t) for x_1, x_2 in boundary_points)
        return evaluated_g

    def step(self) -> tuple[list[float], list[np.matrix]]:
        """ Solves the heat equation incrementally """
        t_list = np.linspace(0, self.T, num=self.M)
        u = self.u0
        u_list = [self.grid.to_grid(u)]
        for t in t_list:
            u, _ = self.solve(u, t)
            u_list.append(self.grid.to_grid(u))
        return t_list, u_list


class Visualizer:
    '''
    Animate heat flow.

    t_list - list of timesteps.
    u_list - list of matrices corresponding to timesteps.
    '''
    def __init__(self, t_list: list[np.ndarray], u_list: list[np.matrix]):
        self.t_list = t_list
        self.u_list = u_list
        self.T = t_list[-1]
        self.N = len(t_list)
        self.interval = (self.T / self.N) * 1000 # milliseconds

    def animate(self, output_filename: Optional[str] = None) -> None:
        # Set up figure and initial image
        fig, ax = plt.subplots()
        img = ax.imshow(self.u_list[0], cmap='viridis', vmin=np.min(self.u_list), vmax=np.max(self.u_list))
        title = ax.set_title(f"t = {self.t_list[0]:.2f}")
        cbar = fig.colorbar(img, ax=ax)

        # Animation function
        def update(frame):
            img.set_data(self.u_list[frame])
            title.set_text(f"t = {self.t_list[frame]:.2f}")
            return img, title

        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=self.N, interval=self.interval, blit=False)

        # Save/display animation
        if output_filename is not None:
            ani.save(output_filename, writer="pillow", fps=self.N / self.T)
        else:
            plt.show()


if __name__ == '__main__':
    def u_init(x_1, x_2):
        return 0 * x_1

    def f(x_1, x_2, t):
        r = 0.3 # set to zero for time independent f 
        return 10 * ((x_1 - 0.5 + r * np.cos(2*np.pi*t))**2 + (x_2 - 0.5 + r * np.sin(2 * np.pi * t))**2 < 0.1).astype(int)

    def moving_boundary_point(t):
        v = 0.3 # speed of point moving on boundary
        distance = v * t 
        circumference = 4.0
        distance %= circumference

        if distance < 1.0:
            y_1, y_2 = distance, 0.0
        elif 1.0 <= distance < 2.0:
            y_1, y_2 = 1.0, distance - 1.0
        elif 2.0 <= distance < 3.0:
            y_1, y_2 = 1.0 - (distance - 2.0), 1.0
        else:
            y_1, y_2 = 0.0, 1.0 - (distance - 3.0)

        return y_1, y_2

    def g(x_1, x_2, t):
        y_1, y_2 = moving_boundary_point(t)
        r = 0.1
        return 0.5 * int((x_1 - y_1)**2 + (x_2 - y_2)**2 < r**2)

    N = 80
    M = 40
    T = 10.0

    # test = Test(N, M, T, 4, 124)
    # test.test_iterations()
    # test.test_residual()

    ts = TimeStepper(N, M, T, u_init, f, g, preconditioned=True)
    t_list, u_list = ts.step()

    # vis = Visualizer(t_list, u_list)
    # vis.animate(output_filename="animation.gif")

