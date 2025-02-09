import math
import matplotlib.pyplot as plt

# ----------------------
# 1. Parameters
# ----------------------
N = 5000       # number of oscillators
kB = 1.0       # Boltzmann's constant (set to 1 for "dimensionless" convenience)
epsilon = 1.0  # quantum of energy (set to 1 => U = q * epsilon = q)
q_max = 102    # we'll go from q=0 up to q=102

# ----------------------
# 2. Arrays to store results
# ----------------------
q_vals = range(q_max + 1)
Omega_vals = [0]*(q_max + 1)
S_vals     = [0.0]*(q_max + 1)   # will hold S in units of k_B
T_vals     = [None]*(q_max + 1)  # will hold T in dimensionless form (kT / epsilon)
C_vals     = [None]*(q_max + 1)  # will hold heat capacity in dimensionless form

# ----------------------
# 3. Multiplicity function
# ----------------------
def multiplicity(n_osc, q_energy):
    # Ω = C(q + n_osc - 1, q)
    return math.comb(n_osc + q_energy - 1, q_energy)

# ----------------------
# 4. First pass: Compute Omega(q) and S(q) = kB ln(Omega).
#    (Here kB=1 => S_vals is actually "S/kB".)
# ----------------------
for q in q_vals:
    Om = multiplicity(N, q)
    Omega_vals[q] = Om
    S_vals[q] = math.log(Om) if Om > 0 else 0.0

# ----------------------
# 5. Second pass: "Centered" difference for T(q):
#    T(q) ~ [2 * epsilon] / [S(q+1) - S(q-1)].
#    We'll do this only where q-1 >= 0 and q+1 <= q_max,
#    i.e. for q in [1, ..., q_max-1].
# ----------------------
for q in range(1, q_max):
    delta_S = S_vals[q+1] - S_vals[q-1]
    if abs(delta_S) > 1e-14:
        T_vals[q] = 2.0 / delta_S  # dimensionless T = kT/epsilon

# ----------------------
# 6. Third pass: "Centered" difference for C(q):
#    C(q) ~ [2 * epsilon] / [T(q+1) - T(q-1)] 
#    Then divide by N to get dimensionless C/(N k_B) (but kB=1).
# ----------------------
for q in range(1, q_max):
    # We need T(q+1) and T(q-1) to exist
    if T_vals[q+1] is not None and T_vals[q-1] is not None:
        delta_T = T_vals[q+1] - T_vals[q-1]
        if abs(delta_T) > 1e-14:
            C_vals[q] = (2.0 / delta_T) / N

# ----------------------
# 7. Print table
# ----------------------
print(f"{'q':>4} {'Omega':>15} {'S/kB':>12} {'kT/ε':>10} {'C/(NkB)':>12}")
for q in q_vals:
    Om_str = f"{Omega_vals[q]:15d}" if Omega_vals[q] < 1e15 else f"{Omega_vals[q]:15.4g}"
    S_str  = f"{S_vals[q]:12.4f}"
    T_str  = "       -   " if (T_vals[q] is None) else f"{T_vals[q]:10.4f}"
    C_str  = "       -   " if (C_vals[q] is None) else f"{C_vals[q]:12.4f}"
    print(f"{q:4d} {Om_str} {S_str} {T_str} {C_str}")

# ----------------------
# 8. Prepare data for plotting
# ----------------------
U_data = [q * epsilon for q in q_vals]  # U = q * epsilon (but epsilon=1, so just q)
S_data = S_vals                         # S/kB
# For T vs. C, we need to filter out 'None' values:
T_plot = []
C_plot = []
for q in q_vals:
    if (T_vals[q] is not None) and (C_vals[q] is not None):
        T_plot.append(T_vals[q])  # dimensionless temperature (kT/epsilon)
        C_plot.append(C_vals[q])  # dimensionless C/(N kB)

# ----------------------
# 9. Plot: Entropy vs. Energy
# ----------------------
plt.figure(figsize=(6,4))
plt.plot(U_data, S_data, 'o-', label='S/kB')
plt.xlabel('Energy U (in units of ε)')
plt.ylabel('Entropy (S/kB)')
plt.title('Entropy vs. Energy (Einstein Solid, N=5000)')
plt.grid(True)
plt.legend()

# ----------------------
# 10. Plot: Heat Capacity vs. Temperature
# ----------------------
plt.figure(figsize=(6,4))
plt.plot(T_plot, C_plot, 'o-', color='red', label='C/(N k_B)')
plt.xlabel('kT/ε')
plt.ylabel('C/(N k_B)')
plt.title('Heat Capacity vs. Temperature (Einstein Solid, N=5000)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

