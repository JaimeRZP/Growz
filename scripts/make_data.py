import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import classy
import utils
from pandas import read_table

class MakeData():
    def __init__(self, z_max, res, path, cosmo_mode='Planck', cosmo_path=None):
        np.random.seed(1998)
        self.c = 299792458.0
        self.path = path
        self.res = res
        self.z_max = z_max
        self.x_arr = np.linspace(0, np.log(1+self.z_max), self.res)
        self.z_arr = np.exp(self.x_arr)-1
        self.a_arr = 1./(1+self.z_arr) 
        self.dx = np.mean(np.diff(self.x_arr))
        self.dz = np.diff(self.z_arr)
        self.sigma8 = None
        self.cosmo = self.get_cosmo(mode=cosmo_mode, path=cosmo_path)
            
        if self.z_max > 1085:
            self.z_planck = self.z_arr[self.z_arr<1085]
        else:
            self.z_planck = self.z_arr
        self.Planck = utils.get_preds(self.z_arr[self.z_arr<1085], self.cosmo)
        
        bg = self.cosmo.get_background()
        self.Omega_nu = bg['(.)rho_ur'][-1]/(bg['H [1/Mpc]'][-1])**2
        self.Wm0 = self.cosmo.Omega_m() 
        self.wm0 = self.cosmo.Omega_m() * self.cosmo.h()**2
        self.wb0 = self.cosmo.Omega_b() * self.cosmo.h()**2
        self.Wr0 = (self.cosmo.Omega_g()+self.Omega_nu) 
        self.wr0 = (self.cosmo.Omega_g()+self.Omega_nu) * self.cosmo.h()**2
        self.WL0 = self.cosmo.Omega_Lambda()
        self.wL0 = self.cosmo.Omega_Lambda() * self.cosmo.h()**2
        
        self.H_arr = 100*np.sqrt(self.wm0*(1+self.z_arr)**3+(self.wr0)*(1+self.z_arr)**4+self.wL0)
        self.H0 = self.H_arr[0]
        self.dM_arr = utils.make_dM((1000/self.c)*self.H_arr, self.x_arr)
        self.dA_arr = self.dM_arr/(1+self.z_arr)
        self.s8_arr, self.fs8_arr = utils.make_fs8(self.H_arr, self.x_arr,
                                                   self.wm0, self.cosmo.sigma8())

    def get_cosmo(self, mode='Planck', path=None):
        if mode=='Planck':
            print('Using Planck mean')
            params = {'h': 0.6727,
                      'Omega_cdm': 0.265621, #0.237153,
                      'Omega_b': 0.0494116,
                      'Omega_Lambda': 0.6834,
                      'sigma8': 0.812}
                      #'n_s': 0.9649,
                      #'ln10^{10}A_s': 3.045}
        elif mode=='best_fit':
            print('Using best fit mean')
            params = {'h': 0.6833,
                      'Omega_cdm': 0.250763, #0.237153,
                      'Omega_b': 0.0479757,
                      'Omega_Lambda': 0.6996939,
                      'sigma8': 0.786}
        elif mode=='other':
            print('Using LCDM mean from file')
            samples = self._get_params_from_file(path)
            H0 = np.mean(samples['H0_gp'])
            Wm0 = np.mean(samples['Omega_m'])
            Wb0 = np.mean(samples['omega_b']/(samples['H0_gp']/100)**2)
            sigma8 = np.mean(samples['s80'])
            Wc0 = Wm0 - Wb0
            WL0 = 1-Wm0-0.0015674
            params = {'h': H0/100,
                      'Omega_cdm': Wc0, 
                      'Omega_b': Wb0,
                      'Omega_Lambda': WL0,
                      'sigma8': sigma8}
        else:
            print('Not recognized option')
        cosmo = classy.Class()
        cosmo.set({'output':'mPk', 'P_k_max_h/Mpc': 20, 'z_max_pk': 1085})
        cosmo.set(params)
        cosmo.compute()
        self.cosmo = cosmo
        self.sigma8 = self.get_sigma8()
        return self.cosmo

    def get_sigma8(self):
        if self.sigma8 is None:
            self.sigma8 = self.cosmo.sigma8()
        return self.sigma8
        
    def make_idx(self, z_data, z_arr):
        idxs = np.array([])
        for z in z_data:
            closest_z = min(z_arr, key=lambda x:abs(x-z))
            idx = np.array([int(i) for i in range(len(z_arr)) if z_arr[i]==closest_z])
            if closest_z >= z:
                idx += -1
            idxs = np.append(idxs, idx)
        return np.array(idxs).astype(int)
            
    def make_U(self, z_data, z_arr, idxs):
        dz = np.diff(z_arr)[idxs]
        return (z_data - z_arr[idxs])/dz
    
    def get_H_mean(self, path=None, mode='LCDM'):
        z_arr = self.z_arr
        if mode=='Planck':
            H_mean = self.H_arr
        elif mode=='LCDM':
            params = self._get_params_from_file(path)
            H0 = np.mean(params['H0_gp'])
            Wm0 = np.mean(params['Omega_m'])
            Wr0 = np.mean(self.wr0/(self.H0/100)**2)
            WL0 = 1-Wm0-Wr0
            H_mean = H0**2*np.sqrt(Wm0*(1+z_arr)**3+Wr0*(1+z_arr)**4+WL0)
        return H_mean

    def get_Wm_mean(self, path=None, mode='LCDM'):
        if mode=='Planck':
            Wm0_mean = self.cosmo.Omega_m
            wm0_mean = self.wm0
        elif mode=='LCDM':
            params = self._get_params_from_file(path)
            Wm0_mean = np.mean(params['Omega_m'])
            H0_mean = np.mean(params['H0_gp'])
            wm0_mean = Wm0_mean*(H0_mean/100)**2
        return Wm0_mean, wm0_mean
        
    def _get_params_from_file(self, path):
        path += '/samples.npz'
        return np.load(path)
        
    def get_synthetic(self, dataset_name, new='False'):
        filepath = os.path.join(self.path, dataset_name+'.npz')
        if (os.path.exists(filepath)) and (new is False):
            print('Found file for '+ dataset_name)
            pass
        else:
            print('Making new '+ dataset_name)
            covs = np.load('/mnt/zfsusers/jaimerz/PhD/Growz/data/raw/'+dataset_name+'_covs.npz')
            rels_dA  = covs['da_err']
            rels_H = covs['h_err'] 
            rels_fs8 = covs['fs8_err']
            z_arr = covs['z_arr']
            idx_arr =  self.make_idx(z_arr, self.z_arr)
            U_arr = self.make_U(z_arr, self.z_arr, idx_arr)

            H = self.H_arr[idx_arr]+(self.H_arr[idx_arr+1]-self.H_arr[idx_arr])*U_arr
            dA = self.dA_arr[idx_arr]+(self.dA_arr[idx_arr+1]-self.dA_arr[idx_arr])*U_arr
            fs8 = self.fs8_arr[idx_arr]+(self.fs8_arr[idx_arr+1]-self.fs8_arr[idx_arr])*U_arr
            
            H_err = covs['h_err']
            dA_err = covs['da_err']
            fs8_err = covs['fs8_err']
            err = np.concatenate([H_err, dA_err, fs8_err])
            np.random.seed(10)
            random = np.random.randn(len(z_arr))
            H_data = H + random*H_err
            dA_data = dA + random*dA_err
            fs8_data = fs8 + random*fs8_err
            data = np.concatenate([H_data, dA_data, fs8_data])

            H_cov = covs['hh_cov']
            dA_cov = covs['dada_cov']
            fs8_cov = covs['fs8fs8_cov']
            cov = covs['total_cov']
            
            np.savez(os.path.join(self.path, dataset_name), 
                     data = data,
                     fs8_data = fs8_data,
                     dA_data = dA_data,
                     H_data = H_data,
                     fs8_err = fs8_err,
                     dA_err = dA_err,
                     H_err = H_err,
                     z = z_arr,
                     cov = cov,
                     err = err, 
                     idx = idx_arr,
                     U = U_arr)
        
        return np.load(filepath)
    
    def get_DESI(self, z_arr=None, new='False', mode=None, improv=1):
        if z_arr is None:
            z_arr = self.z_arr
        if mode is None:
            dataset_name = 'DESI'
        else:
            dataset_name = mode + 'DESI'
        filepath = os.path.join(self.path, dataset_name+'.npz')
        if (os.path.exists(filepath)) and (new is False):
            print('Found file for '+ dataset_name)
            pass
        else:
            print('Making new '+ dataset_name)
            DESI_rels_dA  = [2.78, 1.87, 1.45, 1.19, 1.01, 0.87, 0.77, 
             0.76, 0.88, 0.91, 0.91, 0.91, 1.00, 1.17, 
             1.50, 2.36, 3.62, 4.79]
            DESI_rels_H = [5.34, 3.51, 2.69, 2.20, 1.85, 1.60, 1.41, 
                     1.35, 1.42, 1.41, 1.38, 1.36, 1.46, 1.66, 
                     2.04, 3.15, 4.87, 6.55]
            DESI_rels_fs8 = [3.60, 2.55, 2.17, 1.91, 1.60, 1.24, 
                     1.18, 1.11, 1.13, 1.12, 1.12, 1.14, 1.26, 
                     1.47, 1.89, 3.06, 5.14, 7.66]

            z_DESI = np.arange(0.15, 1.85+0.1, 0.1)
            DESI_idx =  self.make_idx(z_DESI, z_arr)
            DESI_U = self.make_U(z_DESI, z_arr, DESI_idx)

            H_DESI = self.H_arr[DESI_idx]+(self.H_arr[DESI_idx+1]-self.H_arr[DESI_idx])*DESI_U
            dA_DESI = self.dA_arr[DESI_idx]+(self.dA_arr[DESI_idx+1]-self.dA_arr[DESI_idx])*DESI_U
            s8_DESI = self.s8_arr[DESI_idx]+(self.s8_arr[DESI_idx+1]-self.s8_arr[DESI_idx])*DESI_U
            fs8_DESI = self.fs8_arr[DESI_idx]+(self.fs8_arr[DESI_idx+1]-self.fs8_arr[DESI_idx])*DESI_U
            
            print('Multiplying error bars by ', improv)
            DESI_H_err = improv*H_DESI*DESI_rels_H/100
            DESI_dA_err = improv*dA_DESI*DESI_rels_dA/100
            DESI_fs8_err = improv*fs8_DESI*DESI_rels_fs8/100
            DESI_err = np.concatenate([DESI_H_err, DESI_dA_err, DESI_fs8_err])
            
            np.random.seed(10)
            random = np.random.randn(len(z_DESI))
            DESI_H_data = H_DESI + random*DESI_H_err
            DESI_dA_data = dA_DESI + random*DESI_dA_err
            DESI_fs8_data = fs8_DESI + random*DESI_fs8_err
            DESI_data = np.concatenate([DESI_H_data, DESI_dA_data, DESI_fs8_data])
            DESI_geo_data = np.concatenate([DESI_H_data, DESI_dA_data])

            DESI_H_cov = np.zeros([len(z_DESI), len(z_DESI)])
            DESI_dA_cov = np.zeros([len(z_DESI), len(z_DESI)])
            DESI_fs8_cov = np.zeros([len(z_DESI), len(z_DESI)])
            for i in np.arange(len(z_DESI)):
                DESI_H_cov[i,i] = DESI_H_err[i]**2
                DESI_dA_cov[i,i] = DESI_dA_err[i]**2
                DESI_fs8_cov[i,i] = DESI_fs8_err[i]**2

            DESI_cov = np.block([[DESI_H_cov, np.zeros_like(DESI_H_cov), np.zeros_like(DESI_H_cov)],
                             [np.zeros_like(DESI_H_cov), DESI_dA_cov, np.zeros_like(DESI_H_cov)],
                             [np.zeros_like(DESI_H_cov), np.zeros_like(DESI_H_cov), DESI_fs8_cov]])
            
            DESI_geo_cov = np.block([[DESI_H_cov, np.zeros_like(DESI_H_cov)],
                             [np.zeros_like(DESI_H_cov), DESI_dA_cov]])

            if mode is None:
                np.savez(os.path.join(self.path, dataset_name), 
                 data = DESI_data,
                 fs8_data = DESI_fs8_data,
                 dA_data = DESI_dA_data,
                 H_data = DESI_H_data,
                 fs8_err = DESI_fs8_err,
                 dA_err = DESI_dA_err,
                 H_err = DESI_H_err,
                 z=z_DESI,
                 cov=DESI_cov,
                 err=DESI_err, 
                 idx = DESI_idx,
                 U=DESI_U)

            elif mode=='geo':
                np.savez(os.path.join(self.path, dataset_name), 
                 data = DESI_geo_data,
                 z=z_DESI,
                 cov=DESI_geo_cov,
                 idx = DESI_idx,
                 U=DESI_U)

            elif mode=='gro':
                np.savez(os.path.join(self.path, dataset_name), 
                 data = DESI_fs8_data,
                 z=z_DESI,
                 cov=DESI_fs8_cov,
                 idx = DESI_idx,
                 U=DESI_U)
        
        return np.load(filepath)
    
    def get_DS17(self, z_arr=None, new='False'):
        if z_arr is None:
            z_arr = self.z_arr
        dataset_name = 'DS17'
        filepath = os.path.join(self.path, dataset_name+'.npz')
        if (os.path.exists(filepath)) and (new is False):
            print('Found file for '+ dataset_name)
            pass
        else:
            print('Making new '+ dataset_name)
            SN = utils.read_light_curve_parameters('/mnt/zfsusers/jaimerz/PhD/Growz/data/raw/PantheonDS17/lcparam_DS17f.txt')
            SN_data = np.array(SN.mb)
            z_SN = np.array(SN.zcmb)
            SN_idx =  self.make_idx(z_SN, z_arr) 
            SN_U = self.make_U(z_SN, z_arr, SN_idx)
            SN_cov = np.genfromtxt('/mnt/zfsusers/jaimerz/PhD/Growz/data/raw/PantheonDS17/syscov_panth.txt') + np.diag(SN.dmb**2)
            SN_err = np.sqrt(np.diag(SN_cov))

            np.savez(os.path.join(self.path, dataset_name),  
                     data = SN_data,
                     z=z_SN,
                     cov=SN_cov,
                     err=SN_err, 
                     idx = SN_idx,
                     U=SN_U)
        
        return np.load(filepath)
    
    def get_CC(self, z_arr=None, new='False'):
        if z_arr is None:
            z_arr = self.z_arr
        dataset_name = 'CC'
        filepath = os.path.join(self.path, dataset_name+'.npz')
        if (os.path.exists(filepath)) and (new is False):
            print('Found file for '+ dataset_name)
            return np.load(filepath)
        else:
            print('Making new '+ dataset_name)
            CC_data = np.array([69. ,  69. ,  68.6,  83. ,  75. ,  75. ,  72.9,  77. ,
                    88.8,  83. ,  81.5,  83. ,  95. ,  77. ,  87.1,  82.6,  92.8,
                    89. ,  80.9,  97. ,  90.4, 104. ,  87.9,  97.3,  92. ,  97.3,
                   105. , 125. ,  90. , 117. , 154. , 168. , 160. , 177. , 140. ,
                   202. , 186.5])
            z_CC = np.array([0.07  , 0.09  , 0.12  , 0.17  , 0.179 , 0.199 , 0.2   ,
                   0.27  , 0.28  , 0.352 , 0.38  , 0.3802, 0.4   , 0.4004, 0.4247,
                   0.44  , 0.4497, 0.47  , 0.4783, 0.48  , 0.51  , 0.593 , 0.6   ,
                   0.61  , 0.68  , 0.73  , 0.781 , 0.875 , 0.88  , 0.9   , 1.037 ,
                   1.3   , 1.363 , 1.43  , 1.53  , 1.75  , 1.965])
            CC_err= np.array([19.6,  12. ,  26.2,   8. ,   4. ,   5. ,  29.6,  14. ,
                    36.6,  14. ,   1.9,  13.5,  17. ,  10.2,  11.2,   7.8,  12.9,
                    23. ,   9. ,  62. ,   1.9,  13. ,   6.1,   2.1,   8. ,   7. ,
                    12. ,  17. ,  40. ,  23. ,  20. ,  17. ,  33.6,  18. ,  14. ,
                    40. ,  50.4])
            CC_idx =  self.make_idx(z_CC, z_arr) 
            CC_U = self.make_U(z_CC, z_arr, CC_idx)
            CC_cov = np.zeros([len(z_CC),len(z_CC)])
            for i in np.arange(len(z_CC)):
                CC_cov[i,i] = CC_err[i]**2
            np.savez(os.path.join(self.path, dataset_name),  
                     data = CC_data,
                     z=z_CC,
                     cov=CC_cov,
                     err=CC_err, 
                     idx=CC_idx, 
                     U=CC_U)
        
        return np.load(filepath)

    def get_false_CC(self, z_arr=None, new='False'):
        if z_arr is None:
            z_arr = self.z_arr
        dataset_name = 'false_CC'
        filepath = os.path.join(self.path, dataset_name+'.npz')
        if (os.path.exists(filepath)) and (new is False):
            print('Found file for '+ dataset_name)
            return np.load(filepath)
        else:
            print('Making new '+ dataset_name)
            CC_data = np.array([222. , 226. ])
            z_CC = np.array([2.34  , 2.36  ])
            CC_err= np.array([7. ,   8. ])
            CC_idx =  self.make_idx(z_CC, z_arr) 
            CC_U = self.make_U(z_CC, z_arr, CC_idx)
            CC_cov = np.zeros([len(z_CC),len(z_CC)])
            for i in np.arange(len(z_CC)):
                CC_cov[i,i] = CC_err[i]**2
            np.savez(os.path.join(self.path, dataset_name),  
                     data = CC_data,
                     z=z_CC,
                     cov=CC_cov,
                     err=CC_err, 
                     idx=CC_idx, 
                     U=CC_U)
        
        return np.load(filepath)

    def get_Wigglez(self, z_arr=None, new='False'):
        if z_arr is None:
            z_arr = self.z_arr
        dataset_name = 'Wigglez'
        filepath = os.path.join(self.path, dataset_name+'.npz')
        if (os.path.exists(filepath)) and (new is False):
            print('Found file for '+ dataset_name)
            pass
        else:
            print('Making new '+ dataset_name)
            z_Wigglez = np.array([0.44, 0.60, 0.73])
            fs8_Wigglez = np.array([0.413, 0.390, 0.437])
            data_Wigglez = np.concatenate([fs8_Wigglez])
            Wigglez_idx =  self.make_idx(z_Wigglez, z_arr)
            Wigglez_U = self.make_U(z_Wigglez, z_arr, Wigglez_idx)
            Wigglez_cov = 10**(-3)*np.array([[6.4, 2.57, 0], 
                                            [2.57, 3.969, 2.54], 
                                            [0, 2.54, 5.184]])
            Wigglez_fs8_err = np.sqrt(np.diag(Wigglez_cov))
            np.savez(os.path.join(self.path, dataset_name),  
             data = fs8_Wigglez,
             z=z_Wigglez,
             cov=Wigglez_cov,
             err=Wigglez_fs8_err, 
             idx=Wigglez_idx,
             U=Wigglez_U)
        
        return np.load(filepath)
    
    def get_DSS(self, z_arr=None, new='False'):
        if z_arr is None:
            z_arr = self.z_arr
        dataset_name = 'DSS'
        filepath = os.path.join(self.path, dataset_name+'.npz')
        if (os.path.exists(filepath)) and (new is False):
            print('Found file for '+ dataset_name)
            pass
        else:
            print('Making new '+ dataset_name)
            z_DSS = np.array([0])
            fs8_DSS = np.array([0.39])
            DSS_cov = np.array([[0.022**2]])
            DSS_err = np.array([0.022])
            DSS_idx = [0]
            DSS_U = self.make_U(z_DSS, z_arr, DSS_idx)
            np.savez(os.path.join(self.path, dataset_name),  
             data = fs8_DSS,
             z=z_DSS,
             cov=DSS_cov,
             err=DSS_err, 
             idx=DSS_idx, 
             U=DSS_U)
        
        return np.load(filepath)
    
    def get_CMB(self, z_arr=None, new='False'):
        if z_arr is None:
            z_arr = self.z_arr
        dataset_name = 'CMB'
        filepath = os.path.join(self.path, dataset_name+'.npz')
        if (os.path.exists(filepath)) and (new is False):
            print('Found file for '+ dataset_name)
            pass
        else:
            print('Making new '+ dataset_name)
            z_CMB = np.array([1089.95]) 
            CMB_idx =  self.make_idx(z_CMB, z_arr)
            CMB_U = self.make_U(z_CMB, z_arr, CMB_idx)
            #dM_CMB = self.dM_arr[CMB_idx]+(self.dM_arr[CMB_idx+1]-self.dM_arr[CMB_idx])*CMB_U
            #perp_CMB = 100*(CMB_rd/dM_CMB)
            CMB_rd = 144.46 
            perp_CMB = np.array([1.04109]) 
            CMB_cov = np.array([[0.00030**2]])
            CMB_err = np.array([0.00030])
            np.savez(os.path.join(self.path, dataset_name),  
             data = perp_CMB,
             rd=CMB_rd,
             z=z_CMB,
             cov=CMB_cov,
             err=CMB_err, 
             idx=CMB_idx,
             U=CMB_U)
        
        return np.load(filepath)
    
    def get_FCMB(self, z_arr=None, new='False'):
        if z_arr is None:
            z_arr = self.z_arr
        dataset_name = 'FCMB'
        filepath = os.path.join(self.path, dataset_name+'.npz')
        if (os.path.exists(filepath)) and (new is False):
            print('Found file for '+ dataset_name)
            pass
        else:
            print('Making new '+ dataset_name)
            z_FCMB = np.array([1090.30])
            FCMB_idx =  self.make_idx(z_FCMB, z_arr)
            FCMB_U = self.make_U(z_FCMB, z_arr, FCMB_idx)
            dM_CMB = self.dM_arr[FCMB_idx]+(self.dM_arr[FCMB_idx+1]-self.dM_arr[FCMB_idx])*FCMB_U
            
            #perp_CMB = np.array([1.04097])
            
            FCMB_err = np.array([dM_CMB[0]/2000])
            FCMB_cov = np.array([FCMB_err**2])
            #FCMB_err = np.array([0.00046])
            np.savez(os.path.join(self.path, dataset_name),  
             data = dM_CMB,
             z=z_FCMB,
             cov=FCMB_cov,
             err=FCMB_err, 
             idx=FCMB_idx,
             U=FCMB_U)
        
        return np.load(filepath)
    
    def get_eBOSS(self, z_arr=None, mode=None, new='False'):
        if z_arr is None:
            z_arr = self.z_arr
        if mode is None:
            dataset_name = 'eBOSS'
        else:
            dataset_name = mode + 'eBOSS'
        filepath = os.path.join(self.path, dataset_name+'.npz')
        if (os.path.exists(filepath)) and (new is False):
            print('Found file for '+ dataset_name)
            pass
        else:
            print('Making new '+ dataset_name)
            z_eBOSS = np.array([1.48])
            para_eBOSS = 13.23
            perp_eBOSS = 30.21
            fs8_eBOSS = 0.462
            rd_eBOSS = 147.3 
            eBOSS_idx =  self.make_idx(z_eBOSS, z_arr) 
            eBOSS_U = self.make_U(z_eBOSS, z_arr, eBOSS_idx)
            
            data_eBOSS = np.array([13.11, 30.66, 0.439])
            data_geo_eBOSS = np.array([13.11, 30.66])
            data_fs8_eBOSS = np.array([0.439])
            
            eBOSS_cov = np.array([[0.7709, -0.05656, 0.01750],
                                  [-0.05656, 0.2640, -0.006204],
                                  [0.01750, -0.006204, 0.002308]])
            eBOSS_geo_cov = np.array([[0.7709, -0.05656],
                                      [-0.05656, 0.2640]])
            eBOSS_fs8_cov = np.array([[0.002308]])

            eBOSS_H_err = np.sqrt(np.diag(eBOSS_cov))[0]
            eBOSS_dA_err = np.sqrt(np.diag(eBOSS_cov))[1]
            eBOSS_fs8_err = np.sqrt(np.diag(eBOSS_cov))[2]
            
            if mode is None:
                np.savez(os.path.join(self.path, dataset_name),  
                 data = data_eBOSS,
                 z=z_eBOSS,
                 cov=eBOSS_cov,
                 para_data=para_eBOSS,
                 perp_data=perp_eBOSS,
                 fs8_data=fs8_eBOSS,
                 para_err=eBOSS_H_err,
                 perp_err=eBOSS_dA_err,
                 fs8_err=eBOSS_fs8_err, 
                 rd=rd_eBOSS, 
                 idx=eBOSS_idx,
                 U=eBOSS_U)
            elif mode=='geo':
                np.savez(os.path.join(self.path, dataset_name),  
                 data = data_geo_eBOSS,
                 z=z_eBOSS,
                 cov=eBOSS_geo_cov,
                 rd=rd_eBOSS, 
                 idx=eBOSS_idx,
                 U=eBOSS_U)
            elif mode=='gro':
                np.savez(os.path.join(self.path, dataset_name),  
                 data = data_fs8_eBOSS,
                 z=z_eBOSS,
                 cov=eBOSS_fs8_cov,
                 rd=rd_eBOSS, 
                 idx=eBOSS_idx,
                 U=eBOSS_U)
        
        return np.load(filepath)
    
    def get_BOSS(self, z_arr=None, mode=None, new='False'):
        if z_arr is None:
            z_arr = self.z_arr
        if mode is None:
            dataset_name = 'BOSS'
        else:
            dataset_name = mode + 'BOSS'
        filepath = os.path.join(self.path, dataset_name+'.npz')
        if (os.path.exists(filepath)) and (new is False):
            print('Found file for '+ dataset_name)
            pass
        else:
            print('Making new '+ dataset_name)
            z_BOSS = np.array([0.38, 0.51, 0.61])
            perp_BOSS = np.array([1512.39, 1975.22, 2306.68])
            para_BOSS = np.array([81.2087, 90.9029, 98.9647])
            fs8_BOSS = np.array([0.49749, 0.457523, 0.436148])
            data_BOSS = np.concatenate([para_BOSS, perp_BOSS, fs8_BOSS])
            data_geo_BOSS = np.concatenate([para_BOSS, perp_BOSS])
            data_fs8_BOSS = np.concatenate([fs8_BOSS])
            BOSS_idx =  self.make_idx(z_BOSS, z_arr) 
            BOSS_U = self.make_U(z_BOSS, z_arr, BOSS_idx)
            BOSS_cov = np.array([
                   [3.63049e+00, 1.80306e+00, 9.19842e-01, 9.71342e+00, 7.75546e+00,
                    5.97897e+00, 2.79185e-02, 1.24050e-02, 4.75548e-03],
                   [1.80306e+00, 3.77146e+00, 2.21471e+00, 4.85105e+00, 1.19729e+01,
                    9.73184e+00, 9.28354e-03, 2.22588e-02, 1.05956e-02],
                   [9.19842e-01, 2.21471e+00, 4.37982e+00, 2.43394e+00, 6.71715e+00,
                    1.60709e+01, 1.01870e-03, 9.71991e-03, 2.14133e-02],
                   [9.71342e+00, 4.85105e+00, 2.43394e+00, 5.00049e+02, 2.94536e+02,
                    1.42011e+02, 3.91498e-01, 1.51597e-01, 4.36366e-02],
                   [7.75546e+00, 1.19729e+01, 6.71715e+00, 2.94536e+02, 7.02299e+02,
                    4.32750e+02, 1.95890e-01, 3.88996e-01, 1.81786e-01],
                   [5.97897e+00, 9.73184e+00, 1.60709e+01, 1.42011e+02, 4.32750e+02,
                    1.01718e+03, 3.40803e-02, 2.46111e-01, 4.78570e-01],
                   [2.79185e-02, 9.28354e-03, 1.01870e-03, 3.91498e-01, 1.95890e-01,
                    3.40803e-02, 2.03355e-03, 8.11829e-04, 2.64615e-04],
                   [1.24050e-02, 2.22588e-02, 9.71991e-03, 1.51597e-01, 3.88996e-01,
                    2.46111e-01, 8.11829e-04, 1.42289e-03, 6.62824e-04],
                   [4.75548e-03, 1.05956e-02, 2.14133e-02, 4.36366e-02, 1.81786e-01,
                    4.78570e-01, 2.64615e-04, 6.62824e-04, 1.18576e-03]])
            BOSS_geo_cov = np.array([
                   [3.63049e+00, 1.80306e+00, 9.19842e-01, 9.71342e+00, 7.75546e+00, 5.97897e+00],
                   [1.80306e+00, 3.77146e+00, 2.21471e+00, 4.85105e+00, 1.19729e+01, 9.73184e+00],
                   [9.19842e-01, 2.21471e+00, 4.37982e+00, 2.43394e+00, 6.71715e+00, 1.60709e+01],
                   [9.71342e+00, 4.85105e+00, 2.43394e+00, 5.00049e+02, 2.94536e+02, 1.42011e+02],
                   [7.75546e+00, 1.19729e+01, 6.71715e+00, 2.94536e+02, 7.02299e+02, 4.32750e+02],
                   [5.97897e+00, 9.73184e+00, 1.60709e+01, 1.42011e+02, 4.32750e+02, 1.01718e+03]])
            BOSS_fs8_cov = np.array([
                   [2.03355e-03, 8.11829e-04, 2.64615e-04],
                   [8.11829e-04, 1.42289e-03, 6.62824e-04],
                   [2.64615e-04, 6.62824e-04, 1.18576e-03]])

            BOSS_err = np.sqrt(np.diag(BOSS_cov))
            BOSS_para_err = np.array([BOSS_err[0], BOSS_err[1], BOSS_err[2]])
            BOSS_perp_err = np.array([BOSS_err[3], BOSS_err[4], BOSS_err[5]])
            BOSS_fs8_err = np.array([BOSS_err[6], BOSS_err[7], BOSS_err[8]])

            rd_BOSS = 147.78
            if mode is None:
                np.savez(os.path.join(self.path, dataset_name),  
                 data = data_BOSS,
                 z=z_BOSS,
                 cov=BOSS_cov,
                 para_data=para_BOSS,
                 perp_data=perp_BOSS,
                 fs8_data=fs8_BOSS,
                 para_err=BOSS_para_err,
                 perp_err=BOSS_perp_err,
                 fs8_err=BOSS_fs8_err,
                 rd=rd_BOSS, 
                 idx=BOSS_idx, 
                 U=BOSS_U)
            elif mode=='geo':
                np.savez(os.path.join(self.path, dataset_name),  
                 data = data_geo_BOSS,
                 z=z_BOSS,
                 cov=BOSS_geo_cov,
                 rd=rd_BOSS, 
                 idx=BOSS_idx, 
                 U=BOSS_U)
            elif mode=='gro':
                np.savez(os.path.join(self.path, dataset_name),  
                 data = data_fs8_BOSS,
                 z=z_BOSS,
                 cov=BOSS_fs8_cov,
                 rd=rd_BOSS, 
                 idx=BOSS_idx, 
                 U=BOSS_U)
                
        return np.load(filepath)
    
