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
        self.cosmo = classy.Class()
        self.cosmo.set({ 'output':'mPk', 'P_k_max_h/Mpc': 20, 'z_max_pk': 1000})
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

    def make_f(self, H, z_arr, Wm0, Df):  
        dz_f = z_arr[1]
        a_arr = 1/(1+z_arr) 
        x_arr = np.log(a_arr)
        Wm =  Wm0*(H[0]/H)**2*(1+z_arr)**3
        comf_H = a_arr*H
        diff_comf_H = np.zeros(len(z_arr))
        diff_comf_H[0] = (comf_H[1]-comf_H[0])/(x_arr[1]-x_arr[0])
        diff_comf_H[1:-1] = (comf_H[2:]-comf_H[:-2])/(x_arr[2:]-x_arr[:-2])
        diff_comf_H[-1] = (comf_H[-1]-comf_H[-2])/(x_arr[-1]-x_arr[-2])
        q =  1+(diff_comf_H/comf_H)

        f0 = 1-Df
        f = np.zeros(len(z_arr))
        f[-1] = f0
        for i in np.arange(1, len(z_arr)):
            k0 = (-1/(1+z_arr[-i]))*((3/2)*Wm[-i]-f[-i]**2-q[-i]*f[-i])
            f1 = f[-i]-dz_f*k0
            k1 = (-1/(1+z_arr[-(i+1)]))*((3/2)*Wm[-(i+1)]-f1**2-q[-(i+1)]*f1)
            f[-(i+1)] =  f[-i] - dz_f*(k1+k0)/2
        return np.array(f)
    
    def make_sigma8(self, f, z_arr, sigma80): 
        dz_f = z_arr[1]
        s8 = np.ones(len(z_arr))
        s8[0] = sigma80
        for i in np.arange(1, len(z_arr)):
            k0 = -1*(f[i-1]*s8[i-1])/(1+z_arr[i-1])
            #s1 = s8[i-1]+dz_f*k0
            #k1 = -1*(f[i]*s1)/(1+z_arr_f[i])
            s8[i] = s8[i-1] + dz_f*(k0)  #+ dz_f*(k0+k1)/2

        return  np.array(s8)
    
    def make_dM(self, H, z_arr):
        dz_f = z_arr[1]
        dM = np.zeros_like(z_arr)
        dM[1:] = dz_f*np.cumsum(1/H)[:-1]
        return dM

    def get_pred_samples(self, trace, model, gp):
        with model:
            DHkms_pred_f = gp.conditional("DHkms_pred", self.z_new)
        with model:
            pred_samples = pm.sample_posterior_predictive(trace, samples=1000, var_names=["DHkms_pred"])

        return pred_samples
    
    def H_model(self, z, coeffs):
        return coeffs[0] + coeffs[1]*z  + (1/2)*coeffs[2]*z**2

    def _loss(self, coeffs, data, cov, z):
        inv_cov = np.linalg.inv(cov)
        diff = data - self.H_model(z, coeffs)
        xi2 = np.dot(np.dot(diff, inv_cov), diff)
        return np.sqrt(xi2)
    
    def get_H_fit(self, z_output, z_data, data, cov):
        x0 = [70, 0.675, 0.003]
        x = least_squares(self._loss, x0, args=(data, cov, z_data))
        best_coeffs = x['x']
        print(best_coeffs)
        return self.H_model(z_output, best_coeffs)
        

    def R_check(self, trace, model, gp,  path):
        R_stat = pm.summary(trace)['r_hat'][["ℓ","η"]]
        print(R_stat, np.all(np.array(R_stat)))
        with open(os.path.join(path, 'trace.pkl'), 'wb') as buff:
            pickle.dump({'trace': trace}, buff)      
        if np.all(np.array(R_stat)<1.01):
            print('Sampling process finsihed')
            pred_samples = self._get_pred_samples(trace, model, gp)
            with open(os.path.join(path, 'prediction.pkl'), 'wb') as buff:
                pickle.dump({'pred_samples': pred_samples,
                         'z_new': self.z_new}, buff)
            raise KeyboardInterrupt    
