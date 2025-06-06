import numpy as np
from scipy.sparse import csr_matrix
from typing import Optional

def solve(A: csr_matrix, b: np.ndarray, initial_guess: Optional[np.ndarray] = None, max_iterations=10, tol=10e-5) -> np.ndarray:
    '''
    Solves linear system A x = b using the CG-method and returns x (or None if method does not
    converge).

    A - Symmetric positive definite matrix
    b - Right-hand side of linear system
    initial_guess - An initial guess for the solution x. 
    max_iterations - Maximum number of iterations
    tol - Method terminates if the residual error is smaller than tol. 
          If the condition is not met after max_iterations, None is returned.
    '''
    
    tol = tol**2 # We will compare ||residual||**2 with tol**2

    # Initialize variables
    if initial_guess is not None:
        x = initial_guess
    else:
        x = np.zeros(b.shape[0]) 
    r = p = b - A@x # Residual and search direction
    alpha = np.dot(r, r) # ||residual||^2

    # CG method
    for _ in range(max_iterations):
        # Compute auxilliary variables
        v = A@p
        _lambda = alpha / np.dot(v, p)

        # Update variables
        x = x + _lambda * p
        r = r - _lambda * v
        alpha_new = np.dot(r, r)
        p = r + (alpha_new / alpha) * p
        alpha = alpha_new

        # Convergence control
        if alpha < tol:
            return x
        
    # Method did not converge
    return None


if __name__ == '__main__':
    A_dense = np.matrix([
        [2, -1, 0],
        [-1, 2, -1],
        [0, -1, 2],
    ])
    A_sparse = csr_matrix(A_dense)
    b = np.array([-2.6, 100.2, -40.2])
    x = solve(A_sparse, b)

    print(f"A: {A_dense}")
    print(f"b: {b}")
    print(f"x: {x}")
    print(f"A x: {A_sparse@x}")

