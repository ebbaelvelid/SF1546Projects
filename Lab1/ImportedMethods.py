import numpy as np
import matplotlib.pyplot as plt
import os

# Uppgift 1

def fixpunkt_whileloop(g, x0, tol, max_iter):
    # Från exfpiPrincipenWithcosApp.py
    x = x0
    DeltaX = tol + 1.0
    n = 0
    
    print(f"{'n':>3} {'x_n':>15} {'|Δx|':>12}")
    
    while DeltaX > tol:
        n += 1
        xold = x
        x = g(xold)
        DeltaX = np.abs(x - xold)
        print(f"{n:3d} {x:15.10f} {DeltaX:12.2e}")
        
        if n > max_iter:
            raise RuntimeError("Fixpunkt konvergerade inte")
    
    return x, n

def newton_whileloop(func, x0, tol, max_iter): # Kolla derivata 0 (felhantering)
    # Från exNewtonApp.py
    x = x0
    DeltaX = tol + 1.0
    n = 0
    
    print(f"{'n':>3} {'x_n':>15} {'|Δx|':>12}")
    
    while DeltaX > tol:
        n += 1
        fx, fprim = func(x)
        xnew = x - fx/fprim
        DeltaX = np.abs(xnew - x)
        print(f"{n:3d} {xnew:15.10f} {DeltaX:12.2e}") 
        x = xnew
        
        if n > max_iter:
            raise RuntimeError("Newton konvergerade inte")
    
    return x, n

# Uppgift 2

def divided_differences(x, y): # För Newtons ansats i a)
    # Från py3.py
    n = len(x)
    coef = np.copy(y).astype(float)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x[j:n] - x[0:n-j])
    return coef

def eval_newton(coef, x_data, x_eval): # Också för Newtons anstats i a)
    # Från py3.py
    n = len(coef)
    p = np.full_like(x_eval, coef[-1], dtype=float)
    for i in range(n-2, -1, -1):
        p = p * (x_eval - x_data[i]) + coef[i]
    return p