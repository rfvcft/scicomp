import time
import numpy as np
from matplotlib import pyplot as plt
from typing import Optional
from project import TimeStepper

class Test:
    def __init__(self, N : int, M : int, T : int, min_size : int, max_size : int, step_size : int):
        self.N = N
        self.M = M
        self.T = T
        self.min_size = min_size
        self.max_size = max_size
        self.step_size = step_size 
        self.u_init = lambda x_1, x_2: 0*x_1
        self.f = lambda x_1, x_2, t: 10 * ((x_1 - 0.5 + 0.3 * np.cos(2*np.pi*t))**2 + (x_2 - 0.5 + 0.3 * np.sin(2 * np.pi * t))**2 < 0.1).astype(int)
        self.g = lambda x_1, x_2, t: 0

    def test_iterations(self, output_filename : Optional[str] = None):
        iterations_precond = []
        iterations_non_precond = []
        ns = []
        for n in range(self.min_size, self.max_size+1, self.step_size):
            ts_precond = TimeStepper(n, self.M, self.T, self.u_init, self.f, self.g, preconditioned=True)
            _, i = ts_precond.solve(ts_precond.u0, t=0.0)
            iterations_precond.append(i)
            ts_non_precond = TimeStepper(n, self.M, self.T, self.u_init, self.f, self.g, preconditioned=False)
            _, j = ts_non_precond.solve(ts_non_precond.u0, t=0.0)
            iterations_non_precond.append(j)
            ns.append(n**2)
        
        #TODO: Does this match up with what theory predicts??
        fig, ax = plt.subplots()
        ax.scatter(ns, iterations_precond, label="Preconditioned")
        ax.scatter(ns, iterations_non_precond, label="Non-preconditioned")
        ax.set_xlabel("n")
        ax.set_ylabel("Iterations")
        ax.set_title("Iterations vs size of matrix, with and without precond")
        ax.legend()
        
        if output_filename is not None:
            fig.savefig(output_filename)
            plt.close()
        else:
            plt.show()


    def test_residual(self, output_filename : Optional[str] = None):
        ts_precond = TimeStepper(self.N, self.M, self.T, self.u_init, self.f, self.g, preconditioned=True)
        _, i, precond_res = ts_precond.solve(ts_precond.u0, t=0.0, testing=True)
        ts_non_precond = TimeStepper(self.N, self.M, self.T, self.u_init, self.f, self.g, preconditioned=False)
        _, j, non_precond_res = ts_non_precond.solve(ts_precond.u0, t=0.0, testing=True)

        #TODO: Does this match up with what theory predicts??
        fig, ax = plt.subplots()
        ax.scatter(list(range(i+1)), precond_res[1:], label="Preconditioned")
        ax.scatter(list(range(j+1)), non_precond_res[1:], label="Non-preconditioned")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual")
        ax.set_title(f"Residual vs iterations, with and without precond (n={self.N})")
        ax.legend()
        ax.semilogy()
        if output_filename is not None:
            fig.savefig(output_filename)
            plt.close()
        else:
            plt.show()
        plt.show()

    def test_condition_number(self, output_filename : Optional[str] = None):
        condition_number_precond = []
        condition_number_non_precond = []
        ns = []
        for n in range(self.min_size, self.max_size+1, self.step_size):
            ts = TimeStepper(n, self.M, self.T, self.u_init, self.f, self.g, preconditioned=True)
            condition_number_non_precond.append(np.linalg.cond(ts.A.todense()))
            B = np.linalg.inv(ts.solver.L.todense())@ts.A.todense()@np.linalg.inv(ts.solver.L_T.todense())
            condition_number_precond.append(np.linalg.cond(B))
            ns.append(n**2)
            print(n)
        
        fig, ax = plt.subplots()
        ax.scatter(ns, condition_number_precond, label="Preconditioned")
        ax.scatter(ns, condition_number_non_precond, label="Non-preconditioned")
        ax.set_xlabel("n")
        ax.set_ylabel("Conditioning number")
        ax.set_title("Conditioning number vs size of matrix, with and without precond")
        ax.legend()
        
        if output_filename is not None:
            fig.savefig(output_filename)
            plt.close()
        else:
            plt.show()
        plt.show()
    
    def test_time(self, output_filename : Optional[str] = None):
        time_precond = []
        time_non_precond = []
        ns = []
        for n in range(self.min_size, self.max_size+1, self.step_size):
            ts_precond = TimeStepper(n, self.M, self.T, self.u_init, self.f, self.g, preconditioned=True)
            start_time = time.time()
            ts_precond.step()
            time_precond.append(time.time() - start_time)
            ts_non_precond = TimeStepper(n, self.M, self.T, self.u_init, self.f, self.g, preconditioned=False)
            start_time = time.time()
            ts_non_precond.step()
            time_non_precond.append(time.time() - start_time)
            ns.append(n**2)
            print(n)

        fig, ax = plt.subplots()
        ax.scatter(ns, time_precond, label="Preconditioned")
        ax.scatter(ns, time_non_precond, label="Non-preconditioned")
        ax.set_xlabel("n")
        ax.set_ylabel("Time")
        ax.set_title("Time to step vs size of matrix, with and without precond")
        ax.legend()
        
        if output_filename is not None:
            fig.savefig(output_filename)
            plt.close()
        else:
            plt.show()
        
if __name__ == '__main__':
    N = 150
    M = 50
    T = 10.0
    min_size = 2
    max_size = 102
    step_size = 10
    t = Test(N, M, T, min_size, max_size, step_size)
    # t.test_iterations("iterations_test.png")
    # t.test_condition_number("conditioning_test.png")
    # t.test_residual("residual_test.png")
    t.test_time("time_test.png")
