import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
import time as time


# 定义参数和变量
beta = 0.5
gamma =5.0/3.0
eqm = 0
alpha_2= np.arange(0.1, 1.0, 0.05)
w=np.arange(0.1, 1.0, 0.05)

for i in np.arange(0, 18, 1):
    l = 3
    alpha=alpha_2[i]
    Rm = 8
    L =10
    dx =0.2
    x = np.arange(-L, L , dx)
    d2x = 2 * dx
    dx2 = dx * dx

    nx = len(x) - 2

    Bz0 = 1
    RT = 1
    Bm = Bz0
    pm = beta * Bm ** 2 / 2
    rhom = 1
    p0 = pm + Bm ** 2 / 2

    if eqm == 1:
        Bz = (np.abs(x) >= l) * Bz0 * x / np.abs(x + np.finfo(float).eps) + \
             (np.abs(x) < l) * Bz0 * x / l
        p = p0 - Bz ** 2 / 2
        rho = p / RT
    else:
        Bz = Bz0 * np.tanh(x / l)
        p = p0 + Bz0*(1/np.cosh(x/l))**2/ 2
        rho = p / RT

    rhopj = (np.diff(rho[1:nx + 1]) + np.diff(rho[0:nx])) / 2 / dx
    ppj = (np.diff(p[1:nx + 1]) + np.diff(p[0:nx])) / 2 / dx
    Bzpj = (np.diff(Bz[1:nx + 1]) + np.diff(Bz[0:nx])) / 2 / dx

    # 去除边界值
    xj = x[1:nx]
    rhoj = rho[1:nx]
    pj = p[1:nx]
    Bzj = Bz[1:nx]

    # 构建稀疏矩阵
    # rho1
    M12 = np.diag(-rhopj, 0) + np.diag(rhoj[1:nx - 1] / d2x, -1) + np.diag(-rhoj[0:nx - 2] / d2x, 1)
    M13 = np.diag(-1j * alpha * rhoj, 0)
    # ux1
    M24 = np.diag(1j * alpha * Bzj / rhoj, 0)
    M25 = np.diag(-Bzpj / rhoj, 0) + np.diag(Bzj[1:nx - 1] / rhoj[1:nx - 1] / d2x, -1) + np.diag(
        -Bzj[0:nx - 2] / rhoj[0:nx - 2] / d2x, 1)
    M26 = np.diag(beta / (2 * rhoj[1:nx - 1]) / d2x, -1) + np.diag(-beta / (2 * rhoj[1:nx - 1]) / d2x, 1)
    # uz1
    M34 = np.diag(Bzpj / rhoj, 0)
    M36 = np.diag(-1j * alpha * beta / (2 * rhoj), 0)
    # Bx1
    M42 = np.diag(1j * alpha * Bzj, 0)
    M44 = np.diag(-(alpha ** 2 + 2 / dx2) / Rm * np.ones(nx - 1), 0) + np.diag(1 / dx2 / Rm * np.ones(nx - 2),
                                                                               -1) + np.diag(
        1 / dx2 / Rm * np.ones(nx - 2), 1)
    # Bz1
    M52 = np.diag(-Bzpj, 0) + np.diag(Bzj[1:nx - 1] / d2x, -1) + np.diag(-Bzj[0:nx - 2] / d2x, 1)
    M55 = np.diag(-(alpha ** 2 + 2 / dx2) / Rm * np.ones(nx - 1), 0) + np.diag(1 / dx2 / Rm * np.ones(nx - 2),
                                                                               -1) + np.diag(
        1 / dx2 / Rm * np.ones(nx - 2), 1)
    # p1
    M62 = np.diag(-ppj, 0) + np.diag(gamma * pj[1:nx - 1] / d2x, -1) + np.diag(-gamma * pj[0:nx - 2] / d2x, 1)
    M63 = np.diag(-1j * alpha * gamma * pj, 0)

    # 稀疏矩阵
    M = np.vstack((
        np.hstack((np.zeros((nx - 1, nx - 1)), M12, M13, np.zeros((nx - 1, nx - 1)), np.zeros((nx - 1, nx - 1)),
                   np.zeros((nx - 1, nx - 1)))),
        np.hstack((np.zeros((nx - 1, nx - 1)), np.zeros((nx - 1, nx - 1)), np.zeros((nx - 1, nx - 1)), M24, M25, M26)),
        np.hstack((np.zeros((nx - 1, nx - 1)), np.zeros((nx - 1, nx - 1)), np.zeros((nx - 1, nx - 1)), M34,
                   np.zeros((nx - 1, nx - 1)), M36)),
        np.hstack((np.zeros((nx - 1, nx - 1)), M42, np.zeros((nx - 1, nx - 1)), M44, np.zeros((nx - 1, nx - 1)),
                   np.zeros((nx - 1, nx - 1)))),
        np.hstack((np.zeros((nx - 1, nx - 1)), M52, np.zeros((nx - 1, nx - 1)), np.zeros((nx - 1, nx - 1)), M55,
                   np.zeros((nx - 1, nx - 1)))),
        np.hstack((np.zeros((nx - 1, nx - 1)), M62, M63, np.zeros((nx - 1, nx - 1)), np.zeros((nx - 1, nx - 1)),
                   np.zeros((nx - 1, nx - 1))))
    ))
    M = M.astype('complex128')



    sigma = 0.1  # search eigenvalues around this value

    D, V = eigs(M, k=6, sigma=sigma)

    idx = np.argmax(np.real(D))
    w[i] = np.imag(1j * D[idx])
    i=i+1

#print(D[idx])
cnt=18
for cnt in np.arange(0,18,1):
    print("%.3f"%alpha_2[cnt],w[cnt])
    cnt = cnt +1
#绘图

h = plt.figure(figsize=(8, 5), dpi=120)
plt.plot(alpha_2, w, 'g' , markersize=3)
plt.ylabel(r"$\omega_i$")
plt.xlabel(r'$\alpha$')
plt.legend([r'$\omega_{i}$'], loc=0)
plt.title(r'$\alpha$---$\omega$')
plt.show()
