import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import scipy as sp
import classy
import theano
import theano.tensor as tt
import os
import utils
from make_data import MakeData
from scipy.linalg import block_diag
from pymc3.gp.util import plot_gp_dist

#Load data
z_max = 1110
res = 200
x_arr = np.linspace(0, np.log(1+z_max), res)
dx = np.mean(np.diff(x_arr))
z_arr = np.exp(x_arr)-1
a_arr = 1./(1+z_arr)

path = '/mnt/zfsusers/jaimerz/PhD/Growz/data/' 
challenge = 'cosmo41'
if challenge is not None:
    path += 'challenge/'+'cosmo{}_seed100{}'.format(challenge[-2], challenge[-1])

print('data path: ', path)
mean_path =  None #'LCDM_cosmo44_15000_15000'
mean_mode = 'Planck' #'Planck'
data_class = MakeData(z_max, res, path)
Planck = data_class.Planck
z_planck = data_class.z_planck
c = data_class.c
z_planck = data_class.z_planck
c = data_class.c

DESI = data_class.get_DESI(new=False)
WFIRST = data_class.get_WFIRST(new=False)
CC = data_class.get_CC(new=False)
DSS = data_class.get_DSS(new=False)
BOSS = data_class.get_BOSS(new=False)
eBOSS = data_class.get_eBOSS(new=False)
Wigglez = data_class.get_Wigglez(new=False)
DS17 = data_class.get_DS17(new=False)
CMB = data_class.get_CMB(new=False)

n_samples = 15000
n_tune = 15000
datadict = {'DESI': DESI,
            'WFIRST': WFIRST,
            'CC': CC,
            'DS17': DS17, 
            'BOSS': BOSS,
            'eBOSS': eBOSS,
            'Wigglez': Wigglez,
            'DSS': DSS,
            'CMB': CMB}

data_comb = 'All_CMB' # All, All_CMB, SDSS, SDSS_CMB, Add, Add_CMB
data_combs = {'All': ['CC', 'DS17', 'BOSS', 'eBOSS', 'Wigglez', 'DSS'],
             'All_CMB': ['CC', 'DS17', 'BOSS', 'eBOSS', 'Wigglez', 'DSS', 'CMB'],
             'All_CMB_NODSS': ['CC', 'DS17', 'BOSS', 'eBOSS', 'Wigglez', 'CMB'],
             'All_CMB_geo': ['CC', 'DS17', 'geo_BOSS', 'geo_eBOSS', 'CMB'],
             'All_gro': ['fs8_BOSS', 'fs8_eBOSS', 'Wigglez', 'DSS'],
             'All_CMB_gro': ['fs8_BOSS', 'fs8_eBOSS', 'Wigglez', 'DSS', 'CMB'],
             'SDSS': ['BOSS', 'eBOSS'],
             'SDSS_CMB': ['BOSS', 'eBOSS', 'CMB'],
             'Add': ['CC', 'DS17', 'Wigglez', 'DSS'],
             'Add_CMB': ['CC', 'DS17', 'Wigglez', 'DSS', 'CMB'],
             'DESI_CMB': ['DESI', 'CMB'], 
             'DESI_CMB_geo': ['geo_DESI', 'CMB'], 
             'DESI_gro': ['gro_DESI'], 
             'WFIRST_CMB': ['WFIRST', 'CMB']}
datasets = data_combs[data_comb]

need_dM = ['DESI', 'geo_DESI', 'BOSS', 'eBOSS', 'geo_BOSS', 'geo_eBOSS',
           'Wigglez', 'DS17', 'CMB', 'FCMB']
need_fs8 = ['DESI', 'gro_DESI', 'BOSS', 'eBOSS', 'fs8_BOSS', 
            'fs8_eBOSS', 'Wigglez', 'DSS']
need_rd = ['BOSS', 'eBOSS', 'geo_BOSS', 'geo_eBOSS', 'CMB']

if any(dataset in datasets for dataset in need_dM):
    get_dM=True 
else:
    get_dM=False
    
if any(dataset in datasets for dataset in need_fs8):
    get_fs8=True
else:
    get_fs8=False
    
if any(dataset in datasets for dataset in need_rd):
    get_rd = True
else:
    get_rd = False
        
#Data
data = np.array([])
data_cov = np.array([])
for dataset_name in datasets:
    dataset = datadict[dataset_name]
    data = np.concatenate([data, dataset['data']])
    data_cov = block_diag(data_cov, dataset['cov'])
data_cov = data_cov[1:]

#base model
with pm.Model() as model:
    Wm0 = pm.Uniform("Wm0", 0., 1.)
    H0 = pm.Normal("H0", mu=70, sigma=5)
    wm0 = pm.Deterministic("wm0", Wm0*(H0/100)**2)
    Wr0 = data_class.cosmo.Omega_g()+data_class.Omega_nu
    WL0 = pm.Deterministic("WL0", 1-Wm0-Wr0) 
    
    #Set up Gaussian process
    H_gp = pm.Deterministic('H_gp', H0*tt.sqrt(Wm0*(1+z_arr)**3+Wr0*(1+z_arr)**4+WL0))
    H0_gp = pm.Deterministic("H0_gp", H0)
    
    dH_gp = pm.Deterministic("dH", tt.as_tensor_variable((c/1000)/H_gp))
    dM_rec_gp = tt.zeros(len(z_arr)+1)
    dM_rec_gp = tt.inc_subtensor(dM_rec_gp[1:],
              tt.as_tensor_variable(dx*tt.cumsum(dH_gp*(1+z_arr))))
    dM_trap_gp = tt.as_tensor_variable(0.5*(dM_rec_gp[1:]+dM_rec_gp[:-1])-0.5*dM_rec_gp[1])
    dM_gp = pm.Deterministic('dM_gp', dM_trap_gp)
    #dM_gp = pm.Deterministic('dM_gp', dM_rec_gp[:-1])
    dA_gp = pm.Deterministic('dA_gp', dM_gp/(1+z_arr))
    dL_gp = pm.Deterministic('dL_gp', dM_gp*(1+z_arr))
        

    #https://arxiv.org/pdf/2106.00428.pdf
    wb0 =  pm.Uniform("wb0", 0.0, 0.45)
    a1 = 0.00785436
    a2 = 0.177084
    a3 = 0.00912388
    a4 = 0.618711
    a5 = 11.9611
    a6 = 2.81343
    a7 = 0.784719
    rd_gp = pm.Deterministic("rd_gp", 1/(a1*wb0**a2+a3*wm0**a4+a5*wb0**a6*wm0**a7)) 
        

    #s80 = data_class.s80
    s80 = pm.Normal("s80", 0.8, 0.5)
    E = H_gp/H_gp[0]
    xx = x_arr[::-1]
    ee = E[::-1]
    aa = np.exp(-xx)
    dx = np.mean(np.diff(xx))

    nz = len(aa)
    dd = tt.zeros(nz)
    yy = tt.zeros(nz)
    dd = tt.inc_subtensor(dd[0], aa[0])
    yy = tt.inc_subtensor(yy[0], aa[0]**3*E[0])

    for i in range(nz-1):
        A0 = -1.5*Wm0/(aa[i]*ee[i])
        B0 = -1./(aa[i]**2*ee[i])
        A1 = -1.5*Wm0/(aa[i+1]*ee[i+1])
        B1 = -1./(aa[i+1]**2*ee[i+1])
        yy = tt.inc_subtensor(yy[i+1], (1+0.5*dx**2*A0*B0)*yy[i]+0.5*(A0+A1)*dx*dd[i])
        dd = tt.inc_subtensor(dd[i+1],0.5*(B0+B1)*dx*yy[i]+(1+0.5*dx**2*A0*B0)*dd[i])

    y = tt.as_tensor_variable(yy[::-1])
    d = tt.as_tensor_variable(dd[::-1])

    fs8_gp = pm.Deterministic('fs8_gp', s80*y/(a_arr**2*E*d[0]))
    s8_gp = pm.Deterministic('s8_gp', s80*d/d[0])
        
    theory = tt.as_tensor_variable([])
    
    print('Adding CCs')
    CC_H = pm.Deterministic("CC_H",
           tt.as_tensor_variable(H_gp[CC['idx']]+(H_gp[CC['idx']+1]-H_gp[CC['idx']])*CC['U']))
    theory = tt.concatenate([theory, CC_H])
        
    print('Adding Pantheon')
    M = pm.Normal('M', mu=-19.0, sigma=3)
    DS17_dL = tt.as_tensor_variable(dL_gp[DS17['idx']]+(dL_gp[DS17['idx']+1]-dL_gp[DS17['idx']])*DS17['U'])
    DS17_u = pm.Deterministic("DS17_dL",
             tt.as_tensor_variable(5*tt.log10(DS17_dL)+25+M))
    theory = tt.concatenate([theory, DS17_u])
        
    print('Adding BOSS')
    B_H = tt.as_tensor_variable(H_gp[BOSS['idx']]+(H_gp[BOSS['idx']+1]-H_gp[BOSS['idx']])*BOSS['U'])
    B_dM = tt.as_tensor_variable(dM_gp[BOSS['idx']]+(dM_gp[BOSS['idx']+1]-dM_gp[BOSS['idx']])*BOSS['U'])
    B_fs8 = pm.Deterministic("B_fs8", 
               tt.as_tensor_variable(fs8_gp[BOSS['idx']]+(fs8_gp[BOSS['idx']+1]-fs8_gp[BOSS['idx']])*BOSS['U']))
    #Get alpha_perp and alpha_para 
    B_para = pm.Deterministic("B_para", B_H*rd_gp/BOSS['rd'])
    B_perp = pm.Deterministic("B_perp", B_dM*BOSS['rd']/rd_gp)
    theory = tt.concatenate([theory, B_para, B_perp, B_fs8])
        
    print('Adding eBOSS')
    eB_dH = tt.as_tensor_variable(dH_gp[eBOSS['idx']]+(dH_gp[eBOSS['idx']+1]-dH_gp[eBOSS['idx']])*eBOSS['U'])
    eB_dM = tt.as_tensor_variable(dM_gp[eBOSS['idx']]+(dM_gp[eBOSS['idx']+1]-dM_gp[eBOSS['idx']])*eBOSS['U'])
    eB_fs8 = pm.Deterministic("eB_fs8", 
               tt.as_tensor_variable(fs8_gp[eBOSS['idx']]+(fs8_gp[eBOSS['idx']+1]-fs8_gp[eBOSS['idx']])*eBOSS['U']))
    eB_para = pm.Deterministic("eB_para", eB_dH/rd_gp)
    eB_perp = pm.Deterministic("eB_perp", eB_dM/rd_gp)
    theory = tt.concatenate([theory, eB_para, eB_perp, eB_fs8])

    print('Adding Wigglez')
    Wigglez_fs8 = pm.Deterministic("Wigglez_fs8",
                tt.as_tensor_variable(fs8_gp[Wigglez['idx']]+(fs8_gp[Wigglez['idx']+1]-fs8_gp[Wigglez['idx']])*Wigglez['U']))
    theory = tt.concatenate([theory, Wigglez_fs8])

    print('Adding DSS')
    DSS_fs8 = pm.Deterministic("fs8_eBOSS", tt.as_tensor_variable(fs8_gp[DSS['idx']]))
    theory = tt.concatenate([theory, DSS_fs8])

    print('Adding CMB')
    dM_star = tt.as_tensor_variable(dM_gp[CMB['idx']]+(dM_gp[CMB['idx']+1]-dM_gp[CMB['idx']])*CMB['U'])
    t100 = pm.Deterministic('t100', 100*rd_gp/dM_star) 
    theory = tt.concatenate([theory, t100])
        
#Sampling
    lkl= pm.MvNormal("lkl", mu=theory, cov=data_cov, observed=data)
    trace = pm.sample(n_samples, return_inferencedata=True, tune=n_tune)

#print r-stat
print(pm.summary(trace)['r_hat'][["Wm0"]])
print(pm.summary(trace)['mean'][["Wm0"]])

#Save
filename = 'LCDM_freewb_'+data_comb
if mean_mode is not None:
    filename += '_'+mean_mode
if challenge is not None:
    filename += '_'+challenge
    
filename += '_{}_{}'.format(n_samples, n_tune)
print(filename)

Hz =np.array(trace.posterior["H_gp"])
Hz = Hz.reshape(-1, Hz.shape[-1])
H0_gp = np.array(trace.posterior["H0_gp"]).flatten()
Omega_m = np.array(trace.posterior["Wm0"]).flatten()

if get_dM:
    dMz = np.array(trace.posterior["dM_gp"])
    dMz = dMz.reshape(-1, dMz.shape[-1])
else:
    dMz = None

if get_rd:
    rd = np.array(trace.posterior["rd_gp"]).flatten()
    omega_b = np.array(trace.posterior["wb0"]).flatten()
else:
    omega_b = None
    rd = None
    
if get_fs8:
    s8z = np.array(trace.posterior["s8_gp"])
    s8z = s8z.reshape(-1, s8z.shape[-1])
    fs8z = np.array(trace.posterior["fs8_gp"])
    fs8z = fs8z.reshape(-1, fs8z.shape[-1])
    s80 = np.array(trace.posterior["s80"]).flatten()
    S80 = s80*np.sqrt(Omega_m/0.3)
else: 
    s8z = None 
    fs8z = None
    s80 = None
    S80 = None

if 'DS17' in datasets:
    M = np.array(trace.posterior["M"]).flatten()
else:
    M = None

os.mkdir(filename)
np.savez(os.path.join(filename,'samples.npz'), 
         z_arr = z_arr,
         Hz=Hz,
         dMz=dMz,
         s8z=s8z,
         fs8z=fs8z,
         H0_gp=H0_gp,
         Omega_m=Omega_m,
         omega_b=omega_b,
         rd=rd,
         s80=s80,
         S80=S80)
