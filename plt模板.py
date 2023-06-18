#=====================================
# This program is used for ploting
# the growth rate and frequency
# dependent on toroidal mode number
# Author: Yawei (Arvay) Hou
# Email: arvayhou@ustc.edu.cn;
#        arvay.hou@aliyun.com;
# Time: 2018/02/27
# Last updated:
#=====================================
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si
# from scipy import interpolate
from matplotlib.pyplot import savefig
from matplotlib.ticker import ScalarFormatter
from matplotlib.path import Path
from scipy import interpolate
#from scipy.interpolate import spline
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


#--Plot growth rate vs. q_0
def plot_gma_q0(gamma,x,y_sg):

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #ax1.plot(x1,y1_sg,linewidth='6',linestyle=":", color = 'm')
    #ax1.scatter(RDG[:,0],np.abs(RDG[:,1]), marker = '*', color = 'm', s = 160)
    #ax1.set_xlim([0, np.e])
    #ax1.set_ylabel(r"$|\nabla n_f|/n_f$ [$m^{-1}$]", fontsize = 28, color = 'm')
    #ax1.set_xlabel("$\sqrt{\psi/\psi_{LCFS}}", fontsize = 24)

    ax1.plot(x,y_sg/1000,linewidth='3',c='b')
    ax1.scatter(gamma[:,0], gamma[:,1]/1000, marker = 'D', color = 'b', s = 25)
    ax1.set_ylabel(r"$\omega_i$  $[\times 10^3$]rad/s", fontsize = 20, color = 'b')
    ax1.set_xlabel(r"$\alpha$($R_m=80$)", fontsize = 20)


    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 15)


    #ax1=plt.gca()
    #ax1.yaxis.get_major_formatter().set_powerlimits((1,3))
    #plt.legend(loc=9)
    #plt.show()
    plt.savefig('gr-q0.png',format='png',bbox_inches='tight',pad_inches=0,dpi = 300)
    plt.savefig('gr-q0.eps',format='eps',bbox_inches='tight',pad_inches=0,dpi = 300)
    #plt.grid()
    plt.show()
    return()

#--Control parameter: number of q0
nn=150

#--Read data from file
gamma = np.fromfile('alpha.txt', dtype = float,
                     count = 2*nn, sep = ' ')
#--Tf,Gamma
gamma.shape = nn, 2
print(gamma)
#print(gamma)
# Tf = gamma[:,0]
# Gamma = gamma[:,1]

x = np.linspace(gamma[0,0],gamma[nn-1,0], 201)
# interpolate + smooth
itp = interp1d(gamma[:,0], gamma[:,1], kind='linear')
window_size, poly_order = 101, 3
y_sg = savgol_filter(itp(x), window_size, poly_order)


#--Plot growth rate vs. q_0
plot_gma_q0(gamma,x,y_sg)
plt.show()
