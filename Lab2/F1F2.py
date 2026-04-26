def clear_console():
    os.system('clear')

import numpy as np
import matplotlib.pyplot as plt
import os

# Definera funktionen f(t,y) 
def f(t, y):
    return 1 + t - y

# Den exakta lösningen
def y_exakt(t):
    return np.exp(-t) + t

def framatEuler(f, tspan, y0, h): # Från framatEulerlectures.py
    """
    Framåt Euler för ODE-skalär
    f: står för funktionen f(t,y(t))
    tspan: tidsintervall
    y0: beggynnelsevärdet y(t0) = y0
    h: steglängd (tidsteglängd)
    """
    
    a, b = tspan[0], tspan[1]
    # Diskretiseringsteg: Generate n+1 griddpunkter
    n = round(np.abs(b-a)/h)
    t = np.linspace(a, b, n+1)
    
    #Skapa arrayen y[k] med värden noll
    y = np.zeros(n+1)

    #Begynnelsevillkor
    y[0] = y0

    # Iterate med Framåt Euler
    for k in np.arange(n):
        y[k+1] = y[k] + h*f(t[k], y[k])
        
    return t, y

#F1a: Ritar riktningsfältet (från Rexempel61PlotQuiverApp.py)
def direction_field(f, tmin, tmax, ymin, ymax, density, scale):
    
    xs = np.linspace(tmin, tmax, density)
    ys = np.linspace(ymin, ymax, density)
    X, Y = np.meshgrid(xs, ys)

    # Vektorer (1, f(x,y)) normaliserade till enhetlängd
    S = f(X, Y)
    U = np.ones_like(S)
    V = S
    L = np.hypot(U, V)
    U /= L
    V /= L

    fig, ax = plt.subplots(figsize=(8,6))
    plt.quiver(X, Y, U, V, scale=scale, color='steelblue', alpha=0.7)
 
    #Plotta analytisk lösning i samma figur som riktningsfältetet
    t_vec = np.linspace(tmin, tmax, 200)
    y_vec = y_exakt(t_vec)
    plt.plot(t_vec, y_vec, color='red', linewidth=2, label='Exakt: $y=e^{-t}+t$')
    
    plt.xlim(tmin, tmax)
    plt.ylim(ymin, ymax)
    plt.xlabel("t")
    plt.ylabel("y")
    plt.title("F1a: Riktningsfält för dy/dt = 1+t-y")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

def F1a():
    direction_field(f, 0, 1.2, 0.5, 1.8, density=20, scale=20)

# F1b och F1c: Euler framåt med h=0.1 (från Rexemple66EulerForwardApp.py)
def F1b_F1c():
    tspan = np.array([0, 1.2])
    y0 = 1.0
    h = 0.1
    
    tk, yk = framatEuler(f, tspan, y0, h)
    
    # F1c: Beräknar fel vid T=1.2
    ek = np.abs(yk[-1] - y_exakt(tspan[1]))
    print(" ")
    print("F1c: Fel vid T=1.2")
    print(f"ek = |yk(T) - y_exakt(T)| = {ek:.4f}")
    print("Förväntat ek: ca 0.0188")
    
    # Plottn
    fig, ax = plt.subplots(figsize=(8,6))
    #Plot approximativa lösningarna
    ax.plot(tk, yk, 'o', color='blue', label='Euler framåt', markersize=5)
    ax.plot(tk, yk, 'b-.')
    
    #Plot analystiska lösningarna
    t_fine = np.linspace(0, 1.2, 200)
    ax.plot(t_fine, y_exakt(t_fine), 'r-', linewidth=2, label='Exakt lösning')
    
    ax.set_xlabel('t', fontsize=12)
    ax.set_ylabel('y(t)', fontsize=12)
    ax.set_title('F1b: Euler framåt, h=0.1')
    ax.tick_params(labelsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

# F2) Konvergensstudie (från Rexeample68LokaltGlobaltfel.py)
def F2_konvergens():
    tspan = np.array([0, 1.2])
    y0 = 1.0
    
    # F2a-b: Beräknar fel för olika h
    h_vec = [0.2, 0.1, 0.05, 0.025, 0.0125]
    ek = []
    yk_end = []
    
    print(" ")
    print("F2: Konvergensstudie")
    print("-" * 50)
    
    # Definerar kolumnrubrikerna
    headers = ["h", "yk(T)", "y_exakt(T)", "ek"]
    col_format = "{:<8} {:>12} {:>12} {:>12}"
    tal_format = "{:<8.4f} {:>12.6f} {:>12.6f} {:>12.2e}"
    
    # Printar rubrik
    print(col_format.format(*headers))
    print("-" * 50)
    
    for h in h_vec:
        tk, yk = framatEuler(f, tspan, y0, h)
        y_exakt_T = y_exakt(tspan[1])
        fel = np.abs(yk[-1] - y_exakt_T)
        ek.append(fel)
        yk_end.append(yk[-1])
        
        # Printar rader
        row_data = [h, yk[-1], y_exakt_T, fel]
        print(tal_format.format(*row_data))
    
    print(" ")
    print("Förväntat ek: [3.91e-02, 1.88e-02, 9.21e-03, 4.56e-03, 2.27e-03]")
    
    # F2c: Beräkna noggrannhetsordning
    print(" ")
    print("F2c: Noggrannhetsordning p")
    print("-" * 35)
    headers_p = ["h", "p"]
    col_format_p = "{:<8} {:>12}"
    tal_format_p = "{:<8.4f} {:>12.3f}"
    
    print(col_format_p.format(*headers_p))
    print("-" * 35)
    
    p_values = []
    for i in np.arange(len(h_vec)-1):
        p = np.log(ek[i] / ek[i+1]) / np.log(2)
        p_values.append(p)
        print(tal_format_p.format(h_vec[i], p))
    
    print(" ")
    print("Förväntad nogranhetsordning: p ≈ [1.057, 1.028, 1.013, 1.007]")
    
    # Plot konvergens 
    fig, ax = plt.subplots(figsize=(8,6))
    ax.loglog(h_vec, ek, 'o', color='blue', markersize=7)
    ax.loglog(h_vec, ek, 'b-.', label='Fel $e_k$')
    
    # Referenslinje för O(h)
    h_arr = np.array(h_vec)
    ax.loglog(h_arr, h_arr * (ek[0] / h_vec[0]), 'r--', linewidth=2, label='Referens $O(h)$')
    
    ax.set_xlabel('h', fontsize=12)
    ax.set_ylabel('$e_k$', fontsize=12)
    ax.set_title('F2: Konvergensstudie: Euler framåt')
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)
    plt.grid(True, which='both')
    plt.tight_layout()

def main():
   clear_console()
       
   # F1a: Riktningsfältet
   F1a()
   # F1b och F1c: Euler framåt med h=0.1
   F1b_F1c()
   # F2: Konvergensstudie
   F2_konvergens()

if __name__ == "__main__":
    main()
