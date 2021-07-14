import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import utils
from pandas import read_table

class make_data():
    def __init__(self, z_max, res, path):
        self.path = path
        self.tools = utils.utils()
        self.res = res
        self.z_max = z_max
        self.x_arr = np.linspace(0, np.log(1+self.z_max), self.res)
        self.z_arr = np.exp(self.x_arr)-1
        self.a_arr = 1./(1+self.z_arr) 
        self.dx = np.mean(np.diff(self.x_arr))
        self.dz = np.diff(self.z_arr)
        if self.z_max > 1085:
            self.z_planck = self.z_arr[self.z_arr<1085]
        else:
            self.z_planck = self.z_arr
        self.Planck = self.tools.get_preds(self.z_arr[self.z_arr<1085], mode='Planck')

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
                
    def get_WFIRST(self, z_arr=None, new='False'):
        if z_arr is None:
            z_arr = self.z_arr
        dataset_name = 'WFIRST'
        filepath = os.path.join(self.path, dataset_name+'.npz')
        if (os.path.exists(filepath)) and (new is False):
            print('Found file for '+ dataset_name)
            pass
        else:
            WFIRST_rels_E  = np.array([1.3, 1.1, 1.5, 
                                1.5, 2.0, 2.3, 
                                2.6, 3.4, 8.9])
            z_WFIRST = np.array([0.07, 0.2, 0.35, 
                              0.6, 0.8, 1.0,
                              1.3, 1.7, 2.5])
            WFIRST_idx =  self.make_idx(z_WFIRST, z_arr) 
            WFIRST_U = self.make_U(z_WFIRST, z_arr, WFIRST_idx)
            E_arr = self.Planck['Hkms_arr']/self.Planck['Hkms_arr'][0]
            WFIRST_err = E_arr[WFIRST_idx]*WFIRST_rels_E/100
            WFIRST_data = E_arr[WFIRST_idx]+ np.random.randn(len(z_WFIRST))*WFIRST_err
            WFIRST_cov = np.zeros([len(z_WFIRST), len(z_WFIRST)])
            for i in np.arange(len(z_WFIRST)):
                WFIRST_cov[i,i] = WFIRST_err[i]**2
                
            np.savez(os.path.join(self.path, dataset_name), 
                     data = WFIRST_data,
                     z=z_WFIRST,
                     cov=WFIRST_cov,
                     err=WFIRST_err, 
                     idx=WFIRST_idx, 
                     U=WFIRST_U)
            
        return np.load(filepath)
        
    def get_DESI(self, z_arr=None, new='False', mode=None):
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

            H_arr = self.Planck['Hkms_arr']
            x_arr = self.x_arr[self.z_arr<1085]
            z_arr = self.z_arr[self.z_arr<1085]
            dA_arr = self.tools.make_dM((1000/self.tools.c)*H_arr, x_arr)
            dA_arr /= (1+z_arr)
            s8_arr, fs8_arr = self.tools.make_fs8(H_arr, x_arr, 0.1422, 0.812, mode='david')
            
            
            H_DESI = H_arr[DESI_idx]+(H_arr[DESI_idx+1]-H_arr[DESI_idx])*DESI_U
            dA_DESI = dA_arr[DESI_idx]+(dA_arr[DESI_idx+1]-dA_arr[DESI_idx])*DESI_U
            s8_DESI = s8_arr[DESI_idx]+(s8_arr[DESI_idx+1]-s8_arr[DESI_idx])*DESI_U
            fs8_DESI = fs8_arr[DESI_idx]+(fs8_arr[DESI_idx+1]-fs8_arr[DESI_idx])*DESI_U
            
            DESI_H_err = H_DESI*DESI_rels_H/100
            DESI_dA_err = dA_DESI*DESI_rels_dA/100
            DESI_fs8_err = fs8_DESI*DESI_rels_fs8/100

            DESI_err = np.concatenate([DESI_H_err, DESI_dA_err, DESI_fs8_err])

            DESI_H_data = H_DESI + np.random.randn(len(z_DESI))*DESI_H_err
            DESI_dA_data = dA_DESI + np.random.randn(len(z_DESI))*DESI_dA_err
            DESI_fs8_data = fs8_DESI + np.random.randn(len(z_DESI))*DESI_fs8_err
            DESI_data = np.concatenate([DESI_H_data, DESI_dA_data, DESI_fs8_data])

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

            
            if mode is None:
                np.savez(os.path.join(self.path, dataset_name), 
                 data = DESI_data,
                 z=z_DESI,
                 cov=DESI_cov,
                 err=DESI_err, 
                 idx = DESI_idx,
                 U=DESI_U)
                
            elif mode=='H':
                np.savez(os.path.join(self.path, dataset_name), 
                 data = DESI_H_data,
                 z=z_DESI,
                 cov=DESI_H_cov,
                 err=DESI_H_err, 
                 idx = DESI_idx,
                 U=DESI_U)
                
            elif mode=='dA':
                np.savez(os.path.join(self.path, dataset_name), 
                 data = DESI_dA_data,
                 z=z_DESI,
                 cov=DESI_dA_cov,
                 err=DESI_dA_err,
                 idx = DESI_idx,
                 U=DESI_U)
            
            elif mode=='fs8':
                np.savez(os.path.join(self.path, dataset_name), 
                 data = DESI_fs8_data,
                 z=z_DESI,
                 cov=DESI_fs8_cov,
                 err=DESI_fs8_err,
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
            SN = self.tools.read_light_curve_parameters('/home/jaimerz/PhD/Growz/data/raw/PantheonDS17/lcparam_DS17f.txt')
            SN_data = np.array(SN.mb)
            z_SN = np.array(SN.zcmb)
            SN_idx =  self.make_idx(z_SN, z_arr) 
            SN_U = self.make_U(z_SN, z_arr, SN_idx)
            SN_cov = np.genfromtxt('/home/jaimerz/PhD/Growz/data/raw/PantheonDS17/syscov_panth.txt') + np.diag(SN.dmb**2)
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
            CC_data = np.array([69. ,  69. ,  68.6,  83. ,  75. ,  75. ,  72.9,  77. ,
                    88.8,  83. ,  81.5,  83. ,  95. ,  77. ,  87.1,  82.6,  92.8,
                    89. ,  80.9,  97. ,  90.4, 104. ,  87.9,  97.3,  92. ,  97.3,
                   105. , 125. ,  90. , 117. , 154. , 168. , 160. , 177. , 140. ,
                   202. , 186.5, 222. , 226. ])
            z_CC = np.array([0.07  , 0.09  , 0.12  , 0.17  , 0.179 , 0.199 , 0.2   ,
                   0.27  , 0.28  , 0.352 , 0.38  , 0.3802, 0.4   , 0.4004, 0.4247,
                   0.44  , 0.4497, 0.47  , 0.4783, 0.48  , 0.51  , 0.593 , 0.6   ,
                   0.61  , 0.68  , 0.73  , 0.781 , 0.875 , 0.88  , 0.9   , 1.037 ,
                   1.3   , 1.363 , 1.43  , 1.53  , 1.75  , 1.965 , 2.34  , 2.36  ])
            CC_err= np.array([19.6,  12. ,  26.2,   8. ,   4. ,   5. ,  29.6,  14. ,
                    36.6,  14. ,   1.9,  13.5,  17. ,  10.2,  11.2,   7.8,  12.9,
                    23. ,   9. ,  62. ,   1.9,  13. ,   6.1,   2.1,   8. ,   7. ,
                    12. ,  17. ,  40. ,  23. ,  20. ,  17. ,  33.6,  18. ,  14. ,
                    40. ,  50.4,   7. ,   8. ])
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
            z_CMB = np.array([1090.30])
            CMB_rd = 144.46 
            CMB_idx =  self.make_idx(z_CMB, z_arr)
            CMB_U = self.make_U(z_CMB, z_arr, CMB_idx)
            
            H_arr = 100*np.sqrt(0.1422*(1+z_arr)**3+((2.47+1.71)*10**-5)*(1+z_arr)**4+0.30)
            dM_arr = self.tools.make_dM((1000/self.tools.c)*H_arr, self.x_arr)
            dM_CMB = dM_arr[CMB_idx]+(dM_arr[CMB_idx+1]-dM_arr[CMB_idx])*CMB_U
            perp_CMB = 100*(CMB_rd/dM_CMB)
            
            #perp_CMB = np.array([1.04097])
            
            CMB_cov = np.array([[0.00046**2]])
            CMB_err = np.array([0.00046])
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
            z_FCMB = np.array([1090.30])
            FCMB_idx =  self.make_idx(z_FCMB, z_arr)
            FCMB_U = self.make_U(z_FCMB, z_arr, FCMB_idx)
            
            H_arr = 100*np.sqrt(0.1422*(1+z_arr)**3+((2.47+1.71)*10**-5)*(1+z_arr)**4+0.30)
            dM_arr = self.tools.make_dM((1000/self.tools.c)*H_arr, self.x_arr)
            dM_CMB = dM_arr[FCMB_idx]+(dM_arr[FCMB_idx+1]-dM_arr[FCMB_idx])*FCMB_U
            
            #perp_CMB = np.array([1.04097])
            
            FCMB_err = np.array([dM_CMB/2000])
            FCMB_cov = np.array([FCMB_err**2])
            FCMB_err = np.array([0.00046])
            np.savez(os.path.join(self.path, dataset_name),  
             data = dM_CMB,
             z=z_FCMB,
             cov=FCMB_cov,
             err=FCMB_err, 
             idx=FCMB_idx,
             U=FCMB_U)
        
        return np.load(filepath)
    
    def get_eBOSS(self, z_arr=None, new='False'):
        if z_arr is None:
            z_arr = self.z_arr
        dataset_name = 'eBOSS'
        filepath = os.path.join(self.path, dataset_name+'.npz')
        if (os.path.exists(filepath)) and (new is False):
            print('Found file for '+ dataset_name)
            pass
        else:
            z_eBOSS = np.array([1.48])
            para_eBOSS = 13.23
            perp_eBOSS = 30.21
            fs8_eBOSS = 0.462
            rd_eBOSS = 147.3 
            eBOSS_idx =  self.make_idx(z_eBOSS, z_arr) 
            eBOSS_U = self.make_U(z_eBOSS, z_arr, eBOSS_idx)
            data_eBOSS = np.array([13.11, 30.66, 0.439])
            eBOSS_cov = np.array([[0.7709, -0.05656, 0.01750],
                                  [-0.05656, 0.2640, -0.006204],
                                  [0.01750, -0.006204, 0.002308]])


            eBOSS_H_err = np.sqrt(np.diag(eBOSS_cov))[0]
            eBOSS_dA_err = np.sqrt(np.diag(eBOSS_cov))[1]
            eBOSS_fs8_err = np.sqrt(np.diag(eBOSS_cov))[2]
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
        
        return np.load(filepath)
    
    def get_BOSS(self, z_arr=None, new='False'):
        if z_arr is None:
            z_arr = self.z_arr
        dataset_name = 'BOSS'
        filepath = os.path.join(self.path, dataset_name+'.npz')
        if (os.path.exists(filepath)) and (new is False):
            print('Found file for '+ dataset_name)
            pass
        else:
            z_BOSS = np.array([0.38, 0.51, 0.61])
            data_BOSS = np.array([1512.39, 81.2087, 0.49749,
                             1975.22, 90.9029, 0.457523,
                             2306.68, 98.9647,  0.436148])
            perp_BOSS = np.array([1512.39, 1975.22, 2306.68])
            para_BOSS = np.array([81.2087, 90.9029, 98.9647])
            fs8_BOSS = np.array([0.49749, 0.457523, 0.436148])
            data_BOSS = np.concatenate([para_BOSS, perp_BOSS, fs8_BOSS])
            BOSS_idx =  self.make_idx(z_BOSS, z_arr) 
            BOSS_U = self.make_U(z_BOSS, z_arr, BOSS_idx)
            BOSS_cov = np.array([[3.63049e+00, 1.80306e+00, 9.19842e-01, 9.71342e+00, 7.75546e+00,
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

            BOSS_err = np.sqrt(np.diag(BOSS_cov))
            BOSS_para_err = np.array([BOSS_err[0], BOSS_err[1], BOSS_err[2]])
            BOSS_perp_err = np.array([BOSS_err[3], BOSS_err[4], BOSS_err[5]])
            BOSS_fs8_err = np.array([BOSS_err[6], BOSS_err[7], BOSS_err[8]])

            rd_BOSS = 147.78
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
        
        return np.load(filepath)
    
       