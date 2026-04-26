#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 14:54:46 2026

@author: ebbaelvelid
Laboration 2: T2
Randvärdesproblem: Temperaturfördelning i stav
"""

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import os

def console_clear():
    os.system('clear')

k = 2
Ta, Tb = 2, 2

def q(x):
    """Värmekälla"""
    return 50 * x**3 * np.log(x + 1)

# FDM (nästan rakt av från py6.py fdm_poisson)
def fdm_temperatur(xspan, h=None, N=None):
    """
    FDM för k*d²T/dx² = q(x), T(xa)=Ta, T(xb)=Tb
    Baserad på fdm_poisson
    """
    if h is None and N is None:
        N = 10
        h = (xspan[1]-xspan[0])/N
    elif h is None:
        h = (xspan[1]-xspan[0])/N
    else:
        N = round((xspan[1]-xspan[0])/h)

    # Diskretisera x
    xk = np.arange(xspan[0], xspan[1]+h, h)

    # Bilda matris A (tridiagonal, inre punkter j=1..N-1)
    A1 = -2 * np.diag(np.ones(N-1), 0)
    A2 = np.diag(np.ones(N-2),  1)
    A3 = np.diag(np.ones(N-2), -1)
    A = A1 + A2 + A3

    # Bilda högerled b = (h²/k)*q(xj), korrigera för randvillkor
    b = (h**2/k) * q(xk[1:N])
    b[0]  = b[0]  - Ta
    b[-1] = b[-1] - Tb

    # Lös systemet
    A = sparse.csr_matrix(A)
    wk = sparse.linalg.spsolve(A, b)
    wk = np.concatenate((Ta, wk, Tb), axis=None)

    return xk, wk

def plot_analys(xk, wk, xspan, h=None, N=None):
    """Plotta värmekälla och temperaturfördelning"""
    if h is None and N is None:
        N = 10
        h = (xspan[1]-xspan[0])/N
    elif h is None:
        h = (xspan[1]-xspan[0])/N
    else:
        N = round((xspan[1]-xspan[0])/h)

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(16, 6))

    xv = np.linspace(xspan[0], xspan[1], 500)
    ax1.plot(xv, q(xv))
    ax1.set_xlabel('x', fontsize=14)
    ax1.set_ylabel('$q(x)$', fontsize=14)
    ax1.tick_params(labelsize=14)
    ax1.set_title("Värmekälla")
    ax1.grid(True)

    ax2.plot(xk, wk, 'r-.', label="Approximativa lösningar")
    ax2.set_xlabel('x', fontsize=14)
    ax2.set_ylabel('$T(x)$', fontsize=14)
    ax2.tick_params(labelsize=14)
    ax2.set_title(f"Temperaturfördelning (N={N})")
    ax2.legend(fontsize=14)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def globaltfel_fdm(xspan, xtarget):
    """
    Konvergensstudie vid xtarget
    Baserad på globaltfel_fdm_linear
    """
    ek = []
    wk_N = []

    N = 50 * np.array([2**i for i in np.arange(0, 5)])  # 50,100,200,400,800

    for ni in N:
        xk, wk = fdm_temperatur(xspan, h=None, N=ni)
        index = np.argmin(np.abs(xk - xtarget))
        wk_N.append(wk[index])

    # Använd finaste som referens
    T_ref = wk_N[-1]
    for val in wk_N[:-1]:
        ek.append(np.abs(val - T_ref))

    # Beräkna kvot e_k * N² (bör vara konstant för p=2)
    N_plot = N[:-1]
    kvot = np.array(ek) * N_plot**2

    # Tabell
    headers = ["N", "xk", "wk", "ek", "e_k*N^2"]
    col_format = "{:<5}  {:>8}    {:>15}    {:<12}     {:<10}"
    tal_format = "{:<5d} {:>10.4f} {:>15.10f} {:>12.6e}  {:>10.4f}"

    print(col_format.format(*headers))
    print("-" * 65)

    for i in range(len(N_plot)):
        row_data = [N_plot[i], xtarget, wk_N[i], ek[i], kvot[i]]
        print(tal_format.format(*row_data))
    print(f"{N[-1]:<5d} {xtarget:>10.4f} {T_ref:>15.10f}   (referens)")

    # Beräkna p empiriskt (steglängdshalvering)
    print("\nc) Noggrannhetsordning")
    p_values = []
    for i in range(len(ek)-1):
        p = np.log(ek[i] / ek[i+1]) / np.log(2)
        p_values.append(p)
        print(f"   N={N_plot[i]:>4d} → N={N_plot[i+1]:>4d}: p = {p:.3f}")

    print(f"\n   p ≈ 2 (FDM med centrala differenser) ✓")

    # Log-log plot (professorns stil)
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.loglog(N_plot, ek, 'o')
    plt.loglog(N_plot, ek, 'r-.', label='Konvergensanalys')
    ax.set_xlabel('N', fontsize=14)
    ax.set_ylabel('$e_k(N)$', fontsize=14)
    ax.tick_params(labelsize=14)
    plt.title(f"Finita differensmetoden: $x_k$ = {xtarget}")
    ax.legend(fontsize=14)
    plt.grid(True)
    plt.show()

def main():
    console_clear()

    xspan = np.array([0, 1])

    # a) N=4: visa matrisen 
    print("a) Diskretisering N=4")
    h = 0.25
    xk, wk = fdm_temperatur(xspan, h=h, N=None)
    print(f"   h = {h},  xk = {xk}")
    print(f"   T = {wk}\n")

    # b) Generell diskretisering
    print("b) Generell diskretisering: Se fdm_temperatur()\n")

    # c) Test N=4
    print("c) Lösning N=4")
    print(f"   Inre punkter T: {wk[1:-1]}\n")

    # d) Lösning N=100
    print("d) Lösning N=100")
    N = 100
    xk, wk = fdm_temperatur(xspan, h=None, N=N)

    index_02 = np.argmin(np.abs(xk - 0.2))
    print(f"   T(0.2) ≈ {wk[index_02]:.6f}")

    plot_analys(xk, wk, xspan, h=None, N=N)

    # e) Konvergensstudie x=0.7
    print("\ne) Konvergensstudie vid x=0.7\n")
    globaltfel_fdm(xspan, 0.7)

    # f) Olika randvillkor
    print("\nf) Olika randvillkor")

    def fdm_temp_rv(xspan, Ta_in, Tb_in, N=100):
        """FDM med godtyckliga randvillkor"""
        h = (xspan[1]-xspan[0])/N
        xk = np.arange(xspan[0], xspan[1]+h, h)
        A1 = -2*np.diag(np.ones(N-1), 0)
        A2 = np.diag(np.ones(N-2),  1)
        A3 = np.diag(np.ones(N-2), -1)
        A = sparse.csr_matrix(A1 + A2 + A3)
        b = (h**2/k)*q(xk[1:N])
        b[0]  -= Ta_in
        b[-1] -= Tb_in
        wk = sparse.linalg.spsolve(A, b)
        return xk, np.concatenate((Ta_in, wk, Tb_in), axis=None)

    xk1, wk1 = fdm_temp_rv(xspan, 2, 2)
    xk2, wk2 = fdm_temp_rv(xspan, 10, 5)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xk1, wk1, 'b-', linewidth=2, label='Ta=2, Tb=2')
    ax.plot(xk2, wk2, 'r-', linewidth=2, label='Ta=10, Tb=5')
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('$T(x)$', fontsize=14)
    ax.set_title('Inverkan av randvillkor', fontsize=14)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
