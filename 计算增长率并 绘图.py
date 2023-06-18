import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
import time as time


# 定义参数和变量
beta = 0.2
alpha = 0.3
Rm = 8
gamma = 3
bc = 1
eqm = 0

l = 1
L =10*l
dx = 0.1*l
x = np.arange(-L, L+dx, dx)
d2x = 2*dx
dx2 = dx*dx

nx = len(x) -2

Bz0 = 1
RT = 1
Bm = Bz0
pm = beta * Bm**2 / 2
rhom = pm / RT
p0 = pm + Bm**2 / 2

if eqm == 1:
    Bz = (np.abs(x) >= l) * Bz0 * x / np.abs(x + np.finfo(float).eps) + \
         (np.abs(x) < l) * Bz0 * x / l
    p = p0 - Bz**2 / 2
    rho = p / RT
else:
    Bz = Bz0 * np.tanh(x / l)
    p = p0 + (1/np.cosh(x / l))**2 * Bz0**2 / 2
    rho = p / RT

rhopj = (np.diff(rho[1:nx+1]) + np.diff(rho[0:nx])) / 2 / dx
ppj = (np.diff(p[1:nx+1]) + np.diff(p[0:nx])) / 2 / dx
Bzpj = (np.diff(Bz[1:nx+1]) + np.diff(Bz[0:nx])) / 2 / dx

# 去除边界值
xj = x[1:nx]
rhoj = rho[1:nx]
pj = p[1:nx]
Bzj = Bz[1:nx]



# 构建稀疏矩阵
# rho1
M12 = np.diag(-rhopj, 0) + np.diag(rhoj[1:nx-1]/d2x, -1) + np.diag(-rhoj[0:nx-2]/d2x, 1)
M13 = np.diag(-1j*alpha*rhoj, 0)
# ux1
M24 = np.diag(1j*alpha*Bzj/rhoj, 0)
M25 = np.diag(-Bzpj/rhoj, 0) + np.diag(Bzj[1:nx-1]/rhoj[1:nx-1]/d2x, -1) + np.diag(-Bzj[0:nx-2]/rhoj[0:nx-2]/d2x, 1)
M26 = np.diag(beta/(2*rhoj[1:nx-1])/d2x, -1) + np.diag(-beta/(2*rhoj[1:nx-1])/d2x, 1)
# uz1
M34 = np.diag(Bzpj/rhoj, 0)
M36 = np.diag(-1j*alpha*beta/(2*rhoj), 0)
# Bx1
M42 = np.diag(1j*alpha*Bzj, 0)
M44 = np.diag(-(alpha**2+2/dx2)/Rm*np.ones(nx-1), 0) + np.diag(1/dx2/Rm*np.ones(nx-2), -1) + np.diag(1/dx2/Rm*np.ones(nx-2), 1)
# Bz1
M52 = np.diag(-Bzpj, 0) + np.diag(Bzj[1:nx-1]/d2x, -1) + np.diag(-Bzj[0:nx-2]/d2x, 1)
M55 = np.diag(-(alpha**2+2/dx2)/Rm*np.ones(nx-1), 0) + np.diag(1/dx2/Rm*np.ones(nx-2), -1) + np.diag(1/dx2/Rm*np.ones(nx-2), 1)
# p1
M62 = np.diag(-ppj, 0) + np.diag(gamma*pj[1:nx-1]/d2x, -1) + np.diag(-gamma*pj[0:nx-2]/d2x, 1)
M63 = np.diag(-1j*alpha*gamma*pj, 0)

# 稀疏矩阵
M = np.vstack((
    np.hstack((np.zeros((nx-1, nx-1)), M12, M13, np.zeros((nx-1, nx-1)), np.zeros((nx-1, nx-1)), np.zeros((nx-1, nx-1)))),
    np.hstack((np.zeros((nx-1, nx-1)), np.zeros((nx-1, nx-1)), np.zeros((nx-1, nx-1)), M24, M25, M26)),
    np.hstack((np.zeros((nx-1, nx-1)), np.zeros((nx-1, nx-1)), np.zeros((nx-1, nx-1)), M34, np.zeros((nx-1, nx-1)), M36)),
    np.hstack((np.zeros((nx-1, nx-1)), M42, np.zeros((nx-1, nx-1)), M44, np.zeros((nx-1, nx-1)), np.zeros((nx-1, nx-1)))),
    np.hstack((np.zeros((nx-1, nx-1)), M52, np.zeros((nx-1, nx-1)), np.zeros((nx-1, nx-1)), M55, np.zeros((nx-1, nx-1)))),
    np.hstack((np.zeros((nx-1, nx-1)), M62, M63, np.zeros((nx-1, nx-1)), np.zeros((nx-1, nx-1)), np.zeros((nx-1, nx-1))))
))
M = M.astype('complex128')

print(M.shape)

sigma = 0.1  # search eigenvalues around this value

D,V= eigs(M,k=6,sigma=sigma)

idx = np.argmax(np.real(D))
w = 1j * D[idx]

#print(D[idx])
#print(w)

rho1 = V[0:nx-1, idx]
ux1 = V[nx-1:2*nx-2, idx]
uz1 = V[2*nx-2:3*nx-3, idx]
Bx1 = V[3*nx-3:4*nx-4, idx]
Bz1 = V[4*nx-4:5*nx-5, idx]
p1 = V[5*nx-5:6*nx-6, idx]


h = plt.figure(figsize=(13, 9), dpi=80)
plt.subplot(2, 3, 1)
plt.plot(x, rho, 'g', x, Bz, 'b', x, p, 'r--', linewidth=2)
plt.xlabel('x')
plt.xlim(-L, L)
plt.legend(['rho_0', 'Bz_0', 'p_0'], loc=4)
plt.title(f'(a) alpha={alpha}, beta={beta}, Rm={Rm}')

plt.subplot(2, 3, 2)
if nx <= 2 ** 9:
    # FM=np.array(M)
    d = np.linalg.eigvals(M)
    wtmp = 1j * d
    ind = np.where(np.imag(wtmp) > 0)
    xmax = 1.1 * np.max(np.abs(np.real(wtmp)))
    ymax = np.max(np.imag(wtmp))
    ymin = np.min(np.imag(wtmp))
    plt.plot(np.real(wtmp), np.imag(wtmp), 'm.', np.real(wtmp[ind]), np.imag(wtmp[ind]), 'r+', [-xmax, xmax], [0, 0], 'g--', linewidth=2)
    plt.xlabel('Re(omega)')

plt.ylabel('Im(omega)')
plt.title('(b) eigenvalues')

runtime = time.process_time()

plt.subplot(2, 3, 3)
plt.plot(xj, np.real(Bx1), xj, np.imag(Bx1), 'r--', linewidth=2)
plt.xlabel('x')
plt.legend(['Re(Bx_1)', 'Im(Bx_1)'])
plt.title(f'(c) nx={nx}, runtime={runtime:.3f}s')
plt.subplot(2, 3, 4)
plt.plot(xj, np.real(ux1), xj, np.imag(ux1), 'r--', linewidth=2)
plt.xlabel('x')
plt.legend(['Re(ux1)', 'Im(ux1)'])
plt.title("{:.4e}".format(w))
plt.subplot(2, 3, 5)
plt.plot(xj, np.real(p1), xj, np.imag(p1), 'r--', linewidth=2)
plt.xlabel('x')
plt.legend(['Re(p_1)', 'Im(p_1)'])
plt.subplot(2, 3, 6)
plt.plot(xj, np.real(Bz1), xj, np.imag(Bz1), 'r--', linewidth=2)
plt.xlabel('x')
plt.legend(['Re(Bz_1)', 'Im(Bz_1)'])

plt.savefig(f'tearing_eig,alpha={alpha},beta={beta},Rm={Rm},nx={nx}.png', dpi=300)
plt.show()
