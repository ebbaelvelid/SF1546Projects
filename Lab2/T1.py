#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 14:54:46 2026

@author: ebbaelvelid
Laboration 2 - T1
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import os

def clear_console():
    os.system('clear')

# b) F(t,y) för RLC-systemet
def F_RLC(t, y, L, C, R):
    q, i = y
    dqdt = i
    didt = -(1/(L*C))*q - (R/L)*i
    return np.array([dqdt, didt])

def framatEulerSystem(F, tspan, U0, h):
    
    """ 
    Implementerar Framåt Euler för ODE-System
    
    Parameters:
        F       : Vektorvärd function F(t, U)
        tspan   : Tidsintervall
        U0      : initial state U0
        h       : steglängd, eller tidssteg
    
    Returns:
        tk     : numpy array of time points
        Uk     : numpy array of state values
    """
    
    n_steps = round(np.abs(tspan[1]-tspan[0])/h)
    tk = np.zeros(n_steps+1)
    Uk = np.zeros((n_steps+1, len(U0)))

    tk[0] = tspan[0]
    Uk[0] = U0

    for k in np.arange(n_steps):
        Uk[k+1] = Uk[k] + h * F(tk[k], Uk[k])
        tk[k+1] = tk[k] + h

    return tk, Uk  

def plot_rk45_compare(sol_damped, sol_undamped, title='c) RK45 - dämpad vs odämpad'):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(sol_damped.t, sol_damped.y[0], 'r-', label='q(t) dämpad (R=1)')
    ax1.plot(sol_undamped.t, sol_undamped.y[0], 'k--', label='q(t) odämpad (R=0)')
    ax1.set_ylabel('q(t)')
    ax1.set_title(title)
    ax1.grid(True)
    ax1.legend(loc=3)

    ax2.plot(sol_damped.t, sol_damped.y[1], 'b-', label='i(t) dämpad (R=1)')
    ax2.plot(sol_undamped.t, sol_undamped.y[1], 'k--', label='i(t) odämpad (R=0)')
    ax2.set_xlabel('t')
    ax2.set_ylabel('i(t)')
    ax2.grid(True)
    ax2.legend(loc=3)

    plt.tight_layout()
    plt.show()

def plot_euler_vs_rk45(tk, Uk, sol, h, title=''):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(sol.t, sol.y[0], 'k--', linewidth=1.6, label='RK45 (referens)')
    ax1.plot(tk, Uk[:,0], 'r-', linewidth=1.3, label='Euler')
    ax1.set_ylabel('q(t)')
    ax1.set_xlim(tk[0], tk[-1])
    ax1.set_title(title)
    ax1.grid(True)
    ax1.legend(loc=3)

    q_ref = sol.y[0]
    qmin, qmax = np.min(q_ref), np.max(q_ref)
    qmargin = 0.2*(qmax - qmin + 1e-12)
    ax1.set_ylim(qmin - qmargin, qmax + qmargin)

    qmax_euler = np.max(np.abs(Uk[:,0]))
    qmax_ref = np.max(np.abs(q_ref)) + 1e-12
    if qmax_euler > 3*qmax_ref:
        axins1 = ax1.inset_axes([0.65, 0.55, 0.32, 0.35])
        axins1.plot(sol.t, sol.y[0], 'k--', linewidth=1.0)
        axins1.plot(tk, Uk[:,0], 'r-', linewidth=1.0)
        axins1.set_title('zoom-out', fontsize=9)
        axins1.grid(True, alpha=0.3)
        axins1.set_ylim(-1.1*qmax_euler, 1.1*qmax_euler)

    ax2.plot(sol.t, sol.y[1], 'k--', linewidth=1.6, label='RK45 (referens)')
    ax2.plot(tk, Uk[:,1], 'b-', linewidth=1.3, label='Euler')
    ax2.set_xlabel('t')
    ax2.set_ylabel('i(t)')
    ax2.set_xlim(tk[0], tk[-1])
    ax2.grid(True)
    ax2.legend(loc=3)

    i_ref = sol.y[1]
    imin, imax = np.min(i_ref), np.max(i_ref)
    imargin = 0.2*(imax - imin + 1e-12)
    ax2.set_ylim(imin - imargin, imax + imargin)

    imax_euler = np.max(np.abs(Uk[:,1]))
    imax_ref = np.max(np.abs(i_ref)) + 1e-12
    if imax_euler > 3*imax_ref:
        axins2 = ax2.inset_axes([0.65, 0.55, 0.32, 0.35])
        axins2.plot(sol.t, sol.y[1], 'k--', linewidth=1.0)
        axins2.plot(tk, Uk[:,1], 'b-', linewidth=1.0)
        axins2.set_title('zoom-out', fontsize=9)
        axins2.grid(True, alpha=0.3)
        axins2.set_ylim(-1.1*imax_euler, 1.1*imax_euler)

    plt.tight_layout()
    plt.show()

def main():
    clear_console()

    tspan = np.array([0, 20])
    T = tspan[1]
    L, C = 2, 0.5
    U0 = np.array([1, 0])

    # c) solve_ivp RK45
    t_eval = np.linspace(tspan[0], tspan[1], 2000)

    R = 1 # dämpad: används som referens i d) och e) också
    sol_dampad = integrate.solve_ivp(
        lambda t, y: F_RLC(t, y, L, C, R),
        tspan, U0, method='RK45', t_eval=t_eval
    )
    # referens vid sluttiden T (för e)
    q_ref_T = sol_dampad.y[0, -1]
    i_ref_T = sol_dampad.y[1, -1]

    R = 0 # odämpad
    sol_odampad = integrate.solve_ivp(
        lambda t, y: F_RLC(t, y, L, C, R),
        tspan, U0, method='RK45', t_eval=t_eval
    )
    plot_rk45_compare(sol_dampad, sol_odampad)

    # d) Euler framåt, N = 20, 40, 80, 160 (dämpad svängning R=1)
    R = 1
    print("d) Euler framåt, dämpad svängning (L=2, C=0.5, R=1)")
    for N in [20, 40, 80, 160]:
        h = (tspan[1]-tspan[0]) / N
        tk, Uk = framatEulerSystem(lambda t, y: F_RLC(t, y, L, C, R), tspan, U0, h)

        # stabilitetsindikator
        status = 'Instabil' if np.max(np.abs(Uk[:,0])) > 10 else 'Stabil'
        print(f"   N={N:3d}, h={h:.4f}: {status}")

        # plot: Euler + RK45
        plot_euler_vs_rk45(
            tk, Uk, sol_dampad, h,
            title=f'd) Euler vs RK45, N={N}, h={h:.4f} [{status}]'
        )

    # e) Konvergensstudie p = log(e_h/e_{h/2}) / log(2)
    print("\ne) Konvergensstudie: Euler framåt vs RK45 (fel vid T=20)\n")
    headers    = ["N", "h", "e_q", "e_i", "p_q", "p_i"]
    col_format = "{:<6} {:>8} {:>12} {:>12} {:>8} {:>8}"
    tal_format = "{:<6d} {:>8.4f} {:>12.4e} {:>12.4e} {:>8.3f} {:>8.3f}"
    print(col_format.format(*headers))

    N_values = [80, 160, 320, 640]
    ek_q, ek_i = [], []
    for N in N_values:
        h = (tspan[1]-tspan[0]) / N
        tk, Uk = framatEulerSystem(lambda t, y: F_RLC(t, y, L, C, R), tspan, U0, h)

        # fel vid sluttiden T jämfört med RK45-referens
        ek_q.append(np.abs(Uk[-1, 0] - q_ref_T))
        ek_i.append(np.abs(Uk[-1, 1] - i_ref_T))

        if len(ek_q) > 1:
            p_q = np.log(ek_q[-2] / ek_q[-1]) / np.log(2)
            p_i = np.log(ek_i[-2] / ek_i[-1]) / np.log(2)
            print(tal_format.format(N, h, ek_q[-1], ek_i[-1], p_q, p_i))
        else:
            print(f"{N:<6d} {h:>8.4f} {ek_q[-1]:>12.4e} {ek_i[-1]:>12.4e}        -        -")

if __name__ == "__main__":
    main()
