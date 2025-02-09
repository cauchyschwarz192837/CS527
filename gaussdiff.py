import numpy as np
import matplotlib.pyplot as plt

def gaussian(sigma, trunc_multiple=3.5):
    h = np.ceil(sigma * trunc_multiple)
    x = np.arange(-h, h + 1)
    g = np.exp(-0.5 * (x / sigma) ** 2)
    g = g / np.sum(g)
    return g, x

def d_gaussian(sigma, trunc_multiple=3.5):
    h = np.ceil(sigma * trunc_multiple)
    x = np.arange(-h, h + 1)
    g = np.exp(-0.5 * (x / sigma) ** 2)
    d = -x * g

    normalization_array = -(x ** 2) * g
    d = d / (-np.sum(normalization_array))
    return d, x

def s(t, T):
    return np.sin((np.pi * t / T) ** 2)

def s_deriv(t, T):
    return 2 * (np.pi**2 * t / T**2) * np.cos((np.pi * t / T) ** 2)

def plotfunc(x, toplt, title):
    plt.plot(x, toplt, '-')
    plt.axhline(0, color = 'red', linewidth=1.0)
    plt.grid()
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    sigma = 3
    g, gx = gaussian(sigma)
    d, dx = d_gaussian(sigma)

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.plot(gx, g, '.', label='Gaussian')
    plt.grid()
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(dx, d, '.', label='Gaussian Derivative')
    plt.grid()
    plt.legend()
    plt.show()

    T = 50
    t = np.arange(-T, T + 1)
    s_t = s(t, T)
    s_t_deriv = s_deriv(t, T)
    
    s_t_smooth = np.convolve(s_t, g, mode='valid') # 2
    s_t_nderiv = np.convolve(s_t, d, mode='valid') # 3
    s_trunc = s_t[len(g)//2:-(len(g)//2)] # 1
    diff = s_t_nderiv - s_t_deriv[len(g)//2:-(len(g)//2)] # 4

    t_trunc = t[len(g)//2:-(len(g)//2)]

    plotfunc(t_trunc, s_trunc, "s(t) truncated") # plot 1
    plotfunc(t_trunc, s_t_smooth, "s(t) smoothed with g") # plot 2
    plotfunc(t_trunc, s_t_nderiv, "ds(t)") # plot 3
    plotfunc(t_trunc, diff, "difference between ds(t) and s'(t)") # plot 4




