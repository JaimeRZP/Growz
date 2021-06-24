import numpy as np
import pymc3 as pm
import classy
import theano
import theano.tensor as tt
import os
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from pandas import read_table
import pickle

class utils():
    def __init__(self):
        self.c = 299792458.0
        self.z_new = np.arange(0.0, 3+0.1, 0.01)[:, None]
    
    def read_light_curve_parameters(self, path):
        with open(path, 'r') as text:
            clean_first_line = text.readline()[1:].strip()
            names = [e.strip().replace('3rd', 'third')
                     for e in clean_first_line.split()]

        lc_parameters = read_table(
            path, sep=' ', names=names, header=0, index_col=False)
        return lc_parameters
    
    def get_signal(self, z_arr):
        ℓ_true = 0.2
        η_true = 6
        cov_func = η_true ** 2 * pm.gp.cov.Matern52(1, ℓ_true)
        mean_func = pm.gp.mean.Zero()
        return np.random.multivariate_normal(mean_func(z_arr[:, None]).eval(),
               cov_func(z_arr[:, None]).eval()+1e-8*np.eye(len(z_arr[:, None])),
               1).flatten()
    
    def get_preds(self, z_arr, mode='Planck'):
        if mode == 'Planck':
            params = {'h': 0.67,
              'Omega_cdm': 0.26,
              'Omega_b': 0.050,
              'n_s': 0.9665,
              'ln10^{10}A_s': 3.040}
        if mode == 'Riess':
            params = {'h': 0.74,
              'Omega_cdm': 0.21,
              'Omega_b': 0.050,
              'n_s': 0.9665,
              'ln10^{10}A_s': 3.040}
        if mode == 'Panth':
            params = {'h': 0.724,
              'Omega_cdm': 0.262,
              'Omega_b': 0.037,
              'n_s': 0.837,
              'ln10^{10}A_s': 3.138}
        self.cosmo = classy.Class()
        self.cosmo.set({ 'output':'mPk', 'P_k_max_h/Mpc': 20, 'z_max_pk': 1085})
        self.cosmo.set(params)
        self.cosmo.compute()

        self.H0 = self.cosmo.Hubble(0)
        self.h = self.c*self.H0/100000
        self.Wm0 = params['Omega_cdm']+params['Omega_b']
        H_arr = np.array([])
        dA_arr = np.array([])
        dL_arr = np.array([])
        f_arr = np.array([])
        s8_arr = np.array([])
        for z in z_arr: 
            H = self.cosmo.Hubble(z)
            dA = self.cosmo.angular_distance(z)
            dL = self.cosmo.luminosity_distance(z)
            f = self.cosmo.scale_independent_growth_factor_f(z)
            s8 = self.cosmo.sigma(8./self.cosmo.h(),z)
            s8_arr = np.append(s8_arr, s8)
            f_arr = np.append(f_arr, f)
            dL_arr = np.append(dL_arr, dL)
            dA_arr = np.append(dA_arr, dA)
            H_arr = np.append(H_arr, H)
         
        dM_arr = dA_arr*(1+z_arr)
        Hkms_arr = H_arr *self.c/1000
        preds ={'H_arr': H_arr, 'dA_arr': dA_arr,
               'dL_arr': dL_arr, 'dM_arr': dM_arr,
               'Hkms_arr': Hkms_arr, 'f_arr': f_arr, 
               's8_arr': s8_arr}
        return preds
    
    def make_fs8(self, H, x_arr, wm0, s80):
        z_arr = np.exp(x_arr)-1
        a_arr = 1./(1+z_arr) 
        dx = np.mean(np.diff(x_arr))
        
        Wm0 =  wm0*(100/H[0])**2
        E = H/H[0]
        
        d = np.zeros(len(z_arr))
        y = np.zeros(len(z_arr))
        d[-1]= a_arr[-1]
        y[-1]= E[0]*a_arr[-1]**3
        for i in np.arange(1, len(z_arr)):
            A0 = -1.5*Wm0/(a_arr[-i]*E[-i])
            B0 = -1./(a_arr[-i]**2*E[-i])
            A1 = -1.5*Wm0/(a_arr[-(i+1)]*E[-(i+1)])
            B1 = -1./(a_arr[-(i+1)]**2*E[-(i+1)])
            y[-(i+1)] = (1-0.5*dx**2*A0*B0)*y[-i]-0.5*(A0+A1)*dx*d[-i]
            d[-(i+1)] = -0.5*(B0+B1)*dx*y[-i]+(1-0.5*dx**2*A0*B0)*d[-i]
        
        fs8 = s80*y/(a_arr**2*E*d[0])
        s8 = s80*d/d[0]
        
        return s8, fs8
    
    def make_dM(self, H, x_arr):
        z_arr = np.exp(x_arr)-1
        a_arr = 1./(1+z_arr) 
        dx = np.mean(np.diff(x_arr))
        dM = np.zeros_like(z_arr)
        dM[1:] = dx*np.cumsum((1+z_arr)/H)[:-1]
        dM = 0.5*(dM[1:]+dM[:-1])-0.5*dM[1]
        return dM