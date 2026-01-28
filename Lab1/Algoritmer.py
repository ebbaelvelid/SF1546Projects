import numpy as np
import matplotlib.pyplot as plt

# Modul 1: Iterativa Metoder

def bisection(f, a, b, tol=1e-10, max_iter=100): # Intervallhalveringsmetoden
    # Kolla att f(a) och f(b) har olika tecken
    if f(a) * f(b) > 0:
        raise ValueError("f(a) och f(b) måste ha olika tecken")
    
    for n in range(max_iter):
        c = (a + b) / 2   # Mittpunkt
        fc = f(c)
        
        print(f"Iteration {n}: x = {c:.10f}, f(x) = {fc:.2e}")
        
        # Kolla konvergens
        if abs(fc) < tol or (b - a)/2 < tol:
            return c
        
        # Avgör vilket delintervall
        if f(a) * fc < 0:
            b = c  # Nollstället i [a, c]
        else:
            a = c  # Nollstället i [c, b]
    
    raise ValueError("Konvergerade inte")

def fixpoint_iteration(g, x0, tol=1e-10, max_iter=100):
    x = x0
    
    for n in range(max_iter):
        x_new = g(x)
        error = abs(x_new - x)
        
        print(f"Iteration {n}: x = {x_new:.10f}, error = {error:.2e}")
        
        if error < tol:
            print(f"Konvergerat efter {n+1} iterationer")
            return x_new
        
        x = x_new
    
    raise ValueError("Konvergerade inte")

def newtons_method(f, df, x0, tol=1e-10, max_iter=100):
    # f: funktionen
    # df: derivatan av f
    x = x0
    
    for n in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        # Kolla att derivatan inte är noll
        if abs(dfx) < 1e-14:
            raise ValueError("Derivatan är nära noll")
        
        x_new = x - fx/dfx  # Newtons formel
        error = abs(x_new - x)
        
        print(f"Iteration {n}: x = {x_new:.12f}, f(x) = {fx:.2e}")
        
        if error < tol:
            print(f"Konvergerat efter {n+1} iterationer")
            return x_new
        
        x = x_new
    
    raise ValueError("Konvergerade inte")

def secant_method(f, x0, x1, tol=1e-10, max_iter=100):
    for n in range(max_iter):
        f0 = f(x0)
        f1 = f(x1)
        
        # Kolla att f(x1) ≠ f(x0)
        if abs(f1 - f0) < 1e-14:
            raise ValueError("f(x1) - f(x0) nära noll")
        
        # Sekantformeln
        x_new = x1 - f1 * (x1 - x0)/(f1 - f0)
        error = abs(x_new - x1)
        
        print(f"Iteration {n}: x = {x_new:.12f}, f(x) = {f1:.2e}")
        
        if error < tol:
            print(f"Konvergerat efter {n+1} iterationer")
            return x_new
        
        # Uppdatera för nästa iteration
        x0, x1 = x1, x_new
    
    raise ValueError("Konvergerade inte")


# Modul 2: Linjära Ekvationssystem

def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n)
    
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i,j] * y[j]
        y[i] /= L[i,i]
    
    return y

def backward_substitution(U, y):
    n = len(y)
    x = np.zeros(n)
    
    for i in range(n-1, -1, -1):
        x[i] = y[i]
        for j in range(i+1, n):
            x[i] -= U[i,j] * x[j]
        x[i] /= U[i,i]
    
    return x

def lu_factorization(A): 
    n = A.shape[0]
    L = np.eye(n)  # Börja med identitetsmatris
    U = A.copy()   # U kommer modifieras
    
    for k in range(n-1):  # För varje kolonn
        for i in range(k+1, n):  # För varje rad under
            # Beräkna multiplikator
            L[i,k] = U[i,k] / U[k,k]
            
            # Eliminera
            for j in range(k, n):
                U[i,j] -= L[i,k] * U[k,j]
    
    return L, U

def solve_with_lu(A, b): # Lös Ax = b med LU-faktorisering

    # Steg 1: Faktorisera A = LU
    L, U = lu_factorization(A)
    
    # Steg 2: Lös Ly = b (framåtsubstitution)
    y = forward_substitution(L, b)
    
    # Steg 3: Lös Ux = y (bakåtsubstitution)
    x = backward_substitution(U, y)
    
    return x

import scipy.linalg as la # pivotering

A = np.array([[0, 1], [1, 1]])
b = np.array([1, 2])

P, L, U = la.lu(A)

print("P =\n", P)
print("L =\n", L)
print("U =\n", U)

# Lös systemet
Pb = P @ b
y = la.solve_triangular(L, Pb, lower=True)
x = la.solve_triangular(U, y, lower=False)

print("x =", x)

def solve_tridiagonal(l, d, u, b):
    """
    Lös tridiagonalt system
    l: nedre diagonal
    d: huvuddiagonal  
    u: övre diagonal
    """
    n = len(d)
    
    # Framåt
    for i in range(1, n):
        w = l[i-1] / d[i-1]
        d[i] -= w * u[i-1]
        b[i] -= w * b[i-1]
    
    # Bakåt
    x = np.zeros(n)
    x[-1] = b[-1] / d[-1]
    for i in range(n-2, -1, -1):
        x[i] = (b[i] - u[i]*x[i+1]) / d[i]
    
    return x

def hilbert(n):
    """Skapa Hilbert-matris av storlek n"""
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i,j] = 1 / (i + j + 1)
    return H

# Testa
for n in [3, 5, 8, 10]:
    H = hilbert(n)
    kappa = np.linalg.cond(H)
    print(f"n = {n}: κ(H) = {kappa:.2e}")

# Output:
# n = 3:  κ(H) = 5.24e+02
# n = 5:  κ(H) = 9.44e+05
# n = 8:  κ(H) = 1.53e+10
# n = 10: κ(H) = 1.60e+13  (MYCKET dåligt!)

from scipy.sparse import csr_matrix # Glesa system
from scipy.sparse.linalg import spsolve

A_sparse = csr_matrix(A)
x = spsolve(A_sparse, b)

from scipy.linalg import solve_banded # Specifik algoritm för bandomatriser


# Modul 3: Interpolation

def vandermonde_interpolation(x_data, y_data):
    n = len(x_data)
    
    # Skapa Vandermonde-matris
    V = np.vander(x_data, increasing=True)
    
    # Lös Vc = y
    c = np.linalg.solve(V, y_data)
    
    return c  # Koefficienter [c₀, c₁, ..., cₙ₋₁]

def eval_polynomial(c, x):
    """
    Evaluera polynom c₀ + c₁x + c₂x² + ...
    """
    n = len(c)
    result = np.zeros_like(x)
    
    for i in range(n):
        result += c[i] * x**i
    
    return result

# Eller använd NumPys polyval:
# result = np.polyval(c[::-1], x)  # OBS: omvänd ordning!

def centered_interpolation(x_data, y_data):
    """
    Centrerad Vandermonde-ansats
    """
    n = len(x_data)
    xm = np.mean(x_data)
    
    # Centrera x-värdena
    x_centered = x_data - xm
    
    # Vandermonde med centrerade värden
    V = np.vander(x_centered, increasing=True)
    
    # Lös
    c = np.linalg.solve(V, y_data)
    
    return c, xm

def eval_centered_polynomial(c, xm, x):
    """
    Evaluera centrerat polynom
    """
    x_centered = x - xm
    result = np.zeros_like(x)
    
    for i in range(len(c)):
        result += c[i] * x_centered**i
    
    return result

def divided_differences(x, y):
    """
    Beräkna dividerade differenser
    Returnerar koefficienterna för Newtons form
    """
    n = len(x)
    # Skapa tabell
    F = np.zeros((n, n))
    F[:, 0] = y  # Första kolonnen = y-värdena
    
    # Fyll tabellen
    for j in range(1, n):
        for i in range(n - j):
            F[i, j] = (F[i+1, j-1] - F[i, j-1]) / (x[i+j] - x[i])
    
    # Koefficienterna är första raden
    return F[0, :]

def eval_newton_polynomial(c, x_nodes, x):
    """
    Evaluera Newtons interpolationspolynom
    
    p(x) = c₀ + c₁(x-x₀) + c₂(x-x₀)(x-x₁) + ...
    """
    n = len(c)
    p = np.ones_like(x, dtype=float) * c[0]
    
    for i in range(1, n):
        # Beräkna produkt (x-x₀)(x-x₁)...(x-xᵢ₋₁)
        term = c[i]
        for j in range(i):
            term *= (x - x_nodes[j])
        p += term
    
    return p

# Newtons matris
def newton_matrix(x):
    n = len(x)
    N = np.zeros((n, n))
    
    for i in range(n):
        N[i, 0] = 1
        for j in range(1, n):
            term = 1
            for k in range(j):
                term *= (x[i] - x[k])
            N[i, j] = term
    
    return N

# Test
x = np.linspace(0, 1, 12)
N = newton_matrix(x)

print(f"Newton: κ = {np.linalg.cond(N):.2e}")
# Output: κ ≈ 10³-10⁴ (MYCKET bättre än Vandermonde!)


def chebyshev_nodes(n, a=-1, b=1):
    """Skapa Chebyshev-punkter på [a, b]"""
    k = np.arange(1, n+1)
    x_cheb = np.cos((2*k - 1) * np.pi / (2*n))
    # Transformera till [a, b]
    x_cheb = (a + b)/2 + (b - a)/2 * x_cheb
    return np.sort(x_cheb)  # Sortera stigande

# Jämför
for n in [10, 15, 20]:
    # Jämnt fördelade
    x_eq = np.linspace(-1, 1, n)
    y_eq = runge(x_eq)
    c_eq = divided_differences(x_eq, y_eq)
    
    # Chebyshev
    x_cheb = chebyshev_nodes(n)
    y_cheb = runge(x_cheb)
    c_cheb = divided_differences(x_cheb, y_cheb)
    
    # Test
    x_test = np.linspace(-1, 1, 1000)
    y_exact = runge(x_test)
    y_eq_interp = eval_newton_polynomial(c_eq, x_eq, x_test)
    y_cheb_interp = eval_newton_polynomial(c_cheb, x_cheb, x_test)
    
    err_eq = np.max(np.abs(y_exact - y_eq_interp))
    err_cheb = np.max(np.abs(y_exact - y_cheb_interp))
    
    print(f"n = {n}: Jämnt: {err_eq:.6f}, Chebyshev: {err_cheb:.6f}")

def piecewise_linear(x_data, y_data, x):
    return np.interp(x, x_data, y_data)  # NumPys inbyggda

# Exempel
x_data = np.array([0, 1, 2, 3])
y_data = np.array([1, 3, 2, 5])

x_plot = np.linspace(0, 3, 100)
y_plot = piecewise_linear(x_data, y_data, x_plot)

plt.plot(x_plot, y_plot, 'b-', linewidth=2)
plt.plot(x_data, y_data, 'ro', markersize=8)
plt.show()

