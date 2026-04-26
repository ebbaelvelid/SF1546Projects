#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 21:28:46 2026

@author: ebbaelvelid
Del 2.1
"""

import numpy as np
import matplotlib.pyplot as plt
import os
 
 
def clear_console():
    os.system('clear')
 
 
def f(x):
    return np.sin(np.exp(x))
 
 
def df_exakt(x):
    # Kedjeregeln (d/dx[sin(e^x)] = cos(e^x) * e^x)
    return np.cos(np.exp(x)) * np.exp(x)
 
 
def main():
    clear_console()
 
    x0      = 0.75
    df_sann = df_exakt(x0)
    hs      = 2.0**(-np.arange(1, 9)) # h = 2^-1, ..., 2^-8
 
    fel8, fel12, fel13 = [], [], []
 
    for h in hs:
        # Formel (8): centrerad differens
        approx8  = (f(x0 + h) - f(x0 - h)) / (2*h)
        # Formel (12): framåt, andra ordningen
        approx12 = (-f(x0 + 2*h) + 4*f(x0 + h) - 3*f(x0)) / (2*h)
        # Formel (13): bakåt, andra ordningen
        approx13 = (3*f(x0) - 4*f(x0 - h) + f(x0 - 2*h)) / (2*h)
 
        fel8.append(abs(approx8  - df_sann))
        fel12.append(abs(approx12 - df_sann))
        fel13.append(abs(approx13 - df_sann))
 
    fel8  = np.array(fel8)
    fel12 = np.array(fel12)
    fel13 = np.array(fel13)
 
    # Referenskurva Ch^2, anpassad till mitten av datasetet
    idx = 3
    C   = fel8[idx] / hs[idx]**2
    ref = C * hs**2
 
    # Plot
    inv_h = 1.0 / hs
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(inv_h, fel8,  'bo-', label='Centrerad (8)')
    ax.loglog(inv_h, fel12, 'rs-', label='Framåt (12)')
    ax.loglog(inv_h, fel13, 'g^-', label='Bakåt (13)')
    ax.loglog(inv_h, ref,   'k--', label='Referens $Ch^2$')
    ax.set_xlabel('1/h')
    ax.set_ylabel('|fel|')
    ax.set_title('Fel i derivataapproximationer för $f(x)=sin(e^x)$ vid $x=0.75$')
    ax.legend()
    ax.grid(True, which='major', alpha=0.4)
    plt.tight_layout()
    plt.show()
 
 
if __name__ == "__main__":
    main()
