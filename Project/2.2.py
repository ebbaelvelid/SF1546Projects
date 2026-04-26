#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 21:35:56 2026

@author: ebbaelvelid
Del 2.2
"""

import numpy as np
import matplotlib.pyplot as plt
import os
 
 
def clear_console():
    os.system('clear')
 
 
def solve_fin_bvp(N, L, alpha1, alpha2, Ts, TL, Tinf):
    # Löser T'' = alpha1*(T-Tinf) + alpha2*(T^4-Tinf^4) med T(0)=Ts, T(L)=TL via FDM + Newtons metod
    h = L / N
    x = np.linspace(0, L, N+1)
 
    def F(w):
        T   = np.concatenate(([Ts], w, [TL]))
        res = np.zeros(N-1)
        for j in range(1, N):
            T2       = (T[j+1] - 2*T[j] + T[j-1]) / h**2
            rhs      = alpha1*(T[j] - Tinf) + alpha2*(T[j]**4 - Tinf**4)
            res[j-1] = T2 - rhs
        return res
 
    def J(w):
        T    = np.concatenate(([Ts], w, [TL]))
        n    = N - 1
        Jmat = np.zeros((n, n))
        for j in range(1, N):
            i          = j - 1
            Jmat[i, i] = -2/h**2 - alpha1 - 4*alpha2*T[j]**3
            if i > 0:
                Jmat[i, i-1] = 1/h**2
            if i < n-1:
                Jmat[i, i+1] = 1/h**2
        return Jmat
 
    # Startgissning: linjär interpolation
    w = np.linspace(Ts, TL, N+1)[1:N]
 
    # Newtons metod
    for _ in range(100):
        s = -np.linalg.solve(J(w), F(w))
        w = w + s
        if np.linalg.norm(s) < 1e-10:
            break
 
    return x, np.concatenate(([Ts], w, [TL]))
 
 
def diskret_2_norm(w_num, w_exakt): # Diskret 2-norm (formel 16)
    r = w_num - w_exakt
    return np.sqrt(np.sum(r**2) / len(r))
 
 
def main():
    clear_console()
 
    # Uppgift 8.3a
    hc     = 40.0
    K      = 240.0
    D      = 4.13e-3
    Ts     = 450.0
    Tinf   = 293.0
    L      = 2.5
    TL     = Tinf
    alpha1 = 4*hc / (D*K)
    alpha2 = 0.0 # ingen strålning?
 
    N = 400
    x_num, T_num = solve_fin_bvp(N, L, alpha1, alpha2, Ts, TL, Tinf)
    T_exakt = lambda x: Tinf + (Ts - Tinf) * np.exp(-np.sqrt(alpha1) * x)
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(x_num, T_num,          'b-',  label='FDM (N=400)')
    ax1.plot(x_num, T_exakt(x_num), 'r--', label='Analytisk')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('T (K)')
    ax1.set_title('Temperaturfördelning')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(x_num, np.abs(T_num - T_exakt(x_num)), 'g-')
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('|Fel| (K)')
    ax2.set_title('Fel: FDM vs analytisk')
    ax2.grid(True)
    plt.tight_layout()
    plt.show()
 
    # Uppgift 8.3b
    Nvals   = [50, 100, 200, 400, 800]
    eN_vals = []
    for Ni in Nvals:
        xi, Ti = solve_fin_bvp(Ni, L, alpha1, alpha2, Ts, TL, Tinf)
        eN = diskret_2_norm(Ti[1:Ni], T_exakt(xi[1:Ni]))
        eN_vals.append(eN)
 
    print(f"\n{'N':>6}  {'eN':>14}  {'p':>6}")
    for i, (Ni, ei) in enumerate(zip(Nvals, eN_vals)):
        if i == 0:
            print(f"{Ni:>6}  {ei:>14.6e}  {'—':>6}")
        else:
            p = np.log(eN_vals[i-1] / ei) / np.log(2)
            print(f"{Ni:>6}  {ei:>14.6e}  {p:>6.3f}")
 
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(Nvals, eN_vals, 'bo-')
    ax.set_xlabel('N')
    ax.set_ylabel('$e_N$')
    ax.set_title('Diskret 2-fel vs N')
    ax.grid(True)
    plt.tight_layout()
    plt.show()
 
    # Uppgift 8.4a
    sigma   = 5.67e-8
    L_84    = 0.30
    D_84    = 5e-3
    Tinf_84 = 293.15
    Ts_84   = 373.15
    TL_84   = Tinf_84
 
    material = {
        'SS AISI 316':  {'K': 14.0,  'hc': 100.0, 'eps': 0.17},
        'Aluminium Al': {'K': 180.0, 'hc': 100.0, 'eps': 0.82},
        'Koppar Cu':    {'K': 398.0, 'hc': 100.0, 'eps': 0.03},
    }
 
    fig, ax = plt.subplots(figsize=(9, 5))
    for namn, par in material.items():
        a1 = 4 * par['hc'] / (D_84 * par['K'])
        a2 = 4 * par['eps'] * sigma / (D_84 * par['K'])
        x_m, T_m = solve_fin_bvp(400, L_84, a1, a2, Ts_84, TL_84, Tinf_84)
        ax.plot(x_m, T_m, label=namn)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('T (K)')
    ax.set_title('Temperaturfördelning längs fläns (konvektion + strålning)')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
 
    # Uppgift 8.4b
    print("\nMinsta fläns-längd L_min:")
    for namn, par in material.items():
        a1    = 4 * par['hc'] / (D_84 * par['K'])
        L_min = 4.605 / np.sqrt(a1)
        print(f"  {namn:<18}: L_min = {L_min:.4f} m")
 
 
if __name__ == "__main__":
    main()
