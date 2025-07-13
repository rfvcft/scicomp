import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse import csr_matrix, isspmatrix_csr, identity
from scipy.sparse.linalg import spsolve_triangular, spilu, spsolve
from typing import Optional

from line_profiler import profile

class Solver:
    def __init__(self, A: csr_matrix, max_iterations=1000, tol=10e-5, preconditioned=False) -> None:
        """
        A - symmetric positive definite matrix
        max_iterations - Maximum number of iterations
        tol - Method terminates if the residual error is smaller than tol. 
            If the condition is not met after max_iterations, None is returned.
        preconditioned - set to True for preconditioned CG method.
        """
        self.A = A
        self.max_iterations = max_iterations
        self.tol = tol
        self.preconditioned = preconditioned

        if preconditioned:
            self.L = self.incomplete_cholesky() # approximates A as L L^T
            self.L_T = self.L.transpose()    

    @profile
    def incomplete_cholesky(self) -> np.matrix:
        if not (self.A.shape[0] == self.A.shape[1]):
            raise ValueError("Matrix must be square")
        if not (self.A != self.A.T).nnz == 0:
            raise ValueError("Matrix must be symmetric")

        #A = A_csr.tocsr()
        n = self.A.shape[0]
        L = lil_matrix((n, n))  # row-wise efficient

        for i in range(n):
            # Get nonzero column indices in row i
            row_start, row_end = self.A.indptr[i], self.A.indptr[i + 1]
            cols = self.A.indices[row_start:row_end]

            for j in cols:
                if j > i:
                    continue  # Skip upper triangle

                sum_ = 0.0
                for k in L.rows[i]:
                    if k < j and k in L.rows[j]:
                        sum_ += L[i, k] * L[j, k]

                if i == j:
                    val = self.A[i, i] - sum_
                    if val <= 0:
                        raise np.linalg.LinAlgError("Matrix is not positive definite.")
                    L[i, j] = np.sqrt(val)
                else:
                    L[i, j] = (self.A[i, j] - sum_) / L[j, j]

        return L.tocsr()
    
    @profile
    def solve(self, b: np.ndarray, initial_guess: Optional[np.ndarray] = None, testing=False) -> tuple[Optional[np.ndarray], int]:
        '''
        Solves linear system A x = b using the CG-method and returns x (or None if method does not
        converge), and i: the number of iterations taken.

        b - Right-hand side of linear system
        initial_guess - An initial guess for the solution x. 
        '''
        if self.preconditioned:
            return self.preconditioned_solve(b, initial_guess, testing=testing)
        else:
            return self.unpreconditioned_solve(b, initial_guess, testing=testing)
    
    @profile
    def unpreconditioned_solve(self, b: np.ndarray, initial_guess: Optional[np.ndarray], testing=False) -> tuple[Optional[np.ndarray], int]:
        # Initialize variables
        if initial_guess is not None:
            x = initial_guess
        else:
            x = np.zeros(b.shape[0]) 
        r = p = b - self.A@x # Residual and search direction
        alpha = np.dot(r, r) # ||residual||^2

        if testing:
            residuals = []
            residuals.append(alpha)

        # CG method
        for i in range(self.max_iterations):
            # Compute auxilliary variables
            v = self.A@p
            _lambda = alpha / np.dot(v, p)

            # Update variables
            x = x + _lambda * p
            r = r - _lambda * v
            alpha_new = np.dot(r, r)
            p = r + (alpha_new / alpha) * p
            alpha = alpha_new
            
            if testing:
                residuals.append(alpha)

            # Convergence control
            if alpha < self.tol**2:
                if testing:
                    return x, i, residuals
                else:
                    print(f"Unpreconditioned CG-method converged in {i + 1} iterations.")
                    return x, i
            
        # Method did not converge
        return None
    
    @profile
    def preconditioned_solve(self, b: np.ndarray, initial_guess: Optional[np.ndarray], testing=False) -> tuple[Optional[np.ndarray], int]:
        # Initialize variables
        if initial_guess is not None:
            x = initial_guess
        else:
            x = np.zeros(b.shape[0]) 
        r = b - self.A@x
        r_hat = spsolve_triangular(self.L, r, lower=True) # r^{hat} = L^{-1} r
        p = spsolve_triangular(self.L_T, r_hat, lower=False) # p = L^{-T} r
        alpha = np.dot(r_hat, r_hat)
        if testing:
            residuals = []
            residuals.append(np.dot(r, r))

        # CG method
        for i in range(self.max_iterations):
            # Compute auxilliary variables
            v = self.A@p
            _lambda = alpha / np.dot(v, p)

            # Update variables
            x = x + _lambda * p
            r = r - _lambda * v 
            r_hat = spsolve_triangular(self.L, r, lower=True)
            # r_hat = spsolve(self.L, r)
            alpha_new = np.dot(r_hat, r_hat)
            p = spsolve_triangular(self.L_T, r_hat, lower=False) + (alpha_new / alpha) * p
            # p = spsolve(self.L_T, r_hat) + (alpha_new / alpha) * p
            alpha = alpha_new

            if testing:
                residuals.append(np.dot(r, r))

            # Convergence control
            if np.dot(r, r) < self.tol**2:
                if testing:
                    return x, i, residuals
                else: 
                    print(f"Preconditioned CG-method converged in {i + 1} iterations.")
                    return x, i

        # Method did not converge
        return None
    

def test() -> None:
    A_dense = np.matrix([
        [2, 1, 0],
        [1, 2, 0],
        [0, 0, 2],
    ])
    A_sparse = csr_matrix(A_dense)
    b = np.array([3, 0, 0])


    preconditioned_solver = Solver(A_sparse, preconditioned=True)
    x, _ = preconditioned_solver.solve(b)
    
    print(f"x: {x}")
    print(f"A x: {A_sparse@x}")
    

if __name__ == '__main__':
    test()

