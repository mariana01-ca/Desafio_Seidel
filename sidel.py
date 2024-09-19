import numpy as np
A = np.array([[52, 20, 25],
              [30, 50, 20],
              [18, 30, 55]])
b = np.array([4800, 5810, 5690])
x0 = np.zeros(len(b))
max_iterations = 100
tol = 1e-6
def seidel(A, b, x0, tol, max_iterations):
    n = len(b)
    x = x0.copy()
    for k in range(max_iterations):
        x_new = np.copy(x)
        for i in range(n):
            sum1 = sum(A[i][j] * x_new[j] for j in range(i))
            sum2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - sum1 - sum2) / A[i][i]
        
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        x = x_new
        print(f"Iteración {k+1}: {x}")
    return x
solution = seidel(A, b, x0, tol, max_iterations)
print("Solución final:", solution)
