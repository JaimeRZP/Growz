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
        self.dz = 10**(-res)
        self.z_max = z_max
        self.z_arr = np.arange(0.0, self.z_max+0.1, self.dz)
        self.a_arr = 1/(1+self.z_arr) 
        self.x_arr = np.log(self.a_arr)
        self.Planck = self.tools.get_preds(self.z_arr, mode = 'Planck')
    
    def get_WFIRST(self, new='False'):
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
            WFIRST_idx = np.array([int(x) for x in z_WFIRST/(self.dz)])
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
                     idx=WFIRST_idx )
            
        return np.load(filepath)
        
    def get_DESI(self, new='False'):
        dataset_name = 'DESI'
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
            DESI_idx = np.array([int(x) for x in z_DESI/(self.dz)])

            H_arr_f = self.Planck['Hkms_arr']
            dA_arr_f = self.tools.make_dM((1000/self.tools.c)*H_arr_f, self.z_arr)
            dA_arr_f[1:] /= (1+self.z_arr)[1:]
            f_arr_f = self.tools.make_f(H_arr_f, self.z_arr, 0.31, 0.44)
            s8_arr_f = self.tools.make_sigma8(f_arr_f, self.z_arr, 0.805)
            fs8_arr = self.Planck['f_arr']*self.Planck['s8_arr']


            DESI_H_err = H_arr_f[DESI_idx]*DESI_rels_H/100
            DESI_dA_err = dA_arr_f[DESI_idx]*DESI_rels_dA/100
            DESI_fs8_err = fs8_arr[DESI_idx]*DESI_rels_fs8/100

            DESI_err = np.concatenate([DESI_H_err, DESI_dA_err, DESI_fs8_err])

            DESI_H_data = H_arr_f[DESI_idx] + np.random.randn(len(z_DESI))*DESI_H_err
            DESI_dA_data = dA_arr_f[DESI_idx] + np.random.randn(len(z_DESI))*DESI_dA_err
            DESI_fs8_data = fs8_arr[DESI_idx] + np.random.randn(len(z_DESI))*DESI_fs8_err
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

            np.savez(os.path.join(self.path, dataset_name), 
             data = DESI_data,
             z=z_DESI,
             cov=DESI_cov,
             err=DESI_err, 
             H_data = DESI_H_data, 
             dA_data = DESI_dA_data,
             fs8_data = DESI_fs8_data,
             H_err = DESI_H_err, 
             dA_err = DESI_dA_err,
             fs8_err = DESI_fs8_err, 
             idx = DESI_idx)
        
        return np.load(filepath)
    
    def get_DS17(self, new='False'):
        dataset_name = 'DS17'
        filepath = os.path.join(self.path, dataset_name+'.npz')
        if (os.path.exists(filepath)) and (new is False):
            print('Found file for '+ dataset_name)
            pass
        else:
            SN = self.tools.read_light_curve_parameters('/home/jaimerz/PhD/Growz/data/raw/PantheonDS17/lcparam_DS17f.txt')
            SN_data = np.array(SN.mb)
            z_SN = np.array(SN.zcmb)
            SN_idx = np.array([int(x) for x in z_SN/(self.dz)])
            SN_cov = np.genfromtxt('/home/jaimerz/PhD/Growz/data/raw/PantheonDS17/syscov_panth.txt') + np.diag(SN.dmb**2)
            SN_err = np.sqrt(np.diag(SN_cov))

            np.savez(os.path.join(self.path, dataset_name),  
                     data = SN_data,
                     z=z_SN,
                     cov=SN_cov,
                     err=SN_err, 
                     idx = SN_idx)
        
        return np.load(filepath)
    
    def get_CC(self, new='False'):
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
            CC_idx = np.array([int(x) for x in z_CC/(self.dz)])
            CC_cov = np.zeros([len(z_CC),len(z_CC)])
            for i in np.arange(len(z_CC)):
                CC_cov[i,i] = CC_err[i]**2
            np.savez(os.path.join(self.path, dataset_name),  
                     data = CC_data,
                     z=z_CC,
                     cov=CC_cov,
                     err=CC_err, 
                     idx=CC_idx)
        
        return np.load(filepath)
    
    def get_Wigglez(self, new='False'):
        dataset_name = 'Wigglez'
        filepath = os.path.join(self.path, dataset_name+'.npz')
        if (os.path.exists(filepath)) and (new is False):
            print('Found file for '+ dataset_name)
            pass
        else:
            z_Wigglez = np.array([0.44, 0.60, 0.73])
            fs8_Wigglez = np.array([0.413, 0.390, 0.437])
            data_Wigglez = np.concatenate([fs8_Wigglez])
            Wigglez_idx = np.array([int(x) for x in z_Wigglez/(self.dz)])
            Wigglez_cov = 10**(-3)*np.array([[6.4, 2.57, 0], 
                                            [2.57, 3.969, 2.54], 
                                            [0, 2.54, 5.184]])
            Wigglez_fs8_err = np.sqrt(np.diag(Wigglez_cov))
            np.savez(os.path.join(self.path, dataset_name),  
             data = fs8_Wigglez,
             z=z_Wigglez,
             cov=Wigglez_cov,
             err=Wigglez_fs8_err, 
             idx=Wigglez_idx)
        
        return np.load(filepath)
    
    def get_DSS(self, new='False'):
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
            np.savez(os.path.join(self.path, dataset_name),  
             data = fs8_DSS,
             z=z_DSS,
             cov=DSS_cov,
             err=DSS_err, 
             idx=DSS_idx)
        
        return np.load(filepath)
    
    def get_CMB(self, new='False'):
        dataset_name = 'CMB'
        filepath = os.path.join(self.path, dataset_name+'.npz')
        if (os.path.exists(filepath)) and (new is False):
            print('Found file for '+ dataset_name)
            pass
        else:
            z_CMB = np.array([1090.30])
            perp_CMB = np.array([1.04097])
            CMB_cov = np.array([[0.00046**2]])
            CMB_err = np.array([0.00046])
            CMB_rd = 144.46 #+- 0.48
            CMB_idx = [-2]
            np.savez(os.path.join(self.path, dataset_name),  
             data = perp_CMB,
             rd=CMB_rd,
             z=z_CMB,
             cov=CMB_cov,
             err=CMB_err, 
             idx=CMB_idx)
        
        return np.load(filepath)
    
    def get_eBOSS(self, new='False'):
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
            rd_eBOSS = 147.3 #double check
            eBOSS_idx = np.array([int(x) for x in z_eBOSS/(self.dz)])
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
             idx=eBOSS_idx)
        
        return np.load(filepath)
    
    def get_BOSS(self, new='False'):
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
            BOSS_idx = np.array([int(x) for x in z_BOSS/(self.dz)])
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
            z_BOSS_f = np.arange(0.0, 0.61+0.1, self.dz)
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
             idx=BOSS_idx)
        
        return np.load(filepath)
    
       