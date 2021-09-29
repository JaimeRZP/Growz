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

challenge = None #'challenge/cosmo4_seed1004'
path = '/mnt/zfsusers/jaimerz/PhD/Growz/data/'+challenge

mean_path = None #'LCDM_cosmo44_10000_10000'
mean_mode = None #'other'
data_class = MakeData(z_max, res, path,
                      cosmo_mode=mean_mode,
                      cosmo_path=mean_path)
Planck = data_class.Planck
z_planck = data_class.z_planck
c = data_class.c

gro_DESI = data_class.get_DESI(new=True, mode='gro')
gro_BOSS = data_class.get_BOSS(new=True, mode='gro')
gro_eBOSS = data_class.get_eBOSS(new=True, mode='gro')
Wigglez = data_class.get_Wigglez(new=True)
DS17 = data_class.get_DS17(new=True)

n_samples = 10000
n_tune = 10000
datadict = {'gro_DESI': gro_DESI,
            'gro_BOSS': gro_BOSS,
            'gro_eBOSS': gro_eBOSS,
            'Wigglez': Wigglez,
            'DSS': DSS}

data_comb = 'All_gro' # All, All_CMB, SDSS, SDSS_CMB, Add, Add_CMB
data_combs = {'All_gro': ['fs8_BOSS', 'fs8_eBOSS', 'Wigglez', 'DSS'],
             'DESI_gro': ['gro_DESI']}
datasets = data_combs[data_comb]
        
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
    ℓ = pm.Uniform("ℓ", 0.001, 7) 
    η = pm.HalfNormal("η", sigma=0.5) 
    H0 = data_class.H0
    Wm0 = pm.Uniform("Wm0", 0., 1.) 
    wm0_mean = data_class.wm0 
    wr0 = data_class.wr0
    wL0 = data_class.wL0 
    gp_cov = η ** 2 * pm.gp.cov.ExpQuad(1, ℓ) + pm.gp.cov.WhiteNoise(1e-3)
    gp = pm.gp.Latent(cov_func=gp_cov)
    
    #Mean of the gp
    H = pm.Deterministic('H', 100*tt.sqrt(wm0_mean*(1+z_arr)**3+wr0*(1+z_arr)**4+wL0))

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
    

    #print('Adding DESI_gro')
    #with model:
    #    DESI_fs8 = pm.Deterministic('DESI_fs8',
    #               tt.as_tensor_variable(fs8_gp[DESI['idx']]+(fs8_gp[DESI['idx']+1]-fs8_gp[DESI['idx']])*DESI['U']))
    #    theory = tt.concatenate([theory, DESI_fs8])


    print('Adding gro_BOSS')
    B_fs8 = pm.Deterministic("B_fs8", 
               tt.as_tensor_variable(fs8_gp[BOSS['idx']]+(fs8_gp[BOSS['idx']+1]-fs8_gp[BOSS['idx']])*BOSS['U']))
    theory = tt.concatenate([theory, B_fs8])

    print('Adding gro_eBOSS')
    eB_fs8 = pm.Deterministic("eB_fs8", 
               tt.as_tensor_variable(fs8_gp[eBOSS['idx']]+(fs8_gp[eBOSS['idx']+1]-fs8_gp[eBOSS['idx']])*eBOSS['U']))
    theory = tt.concatenate([theory, eB_fs8])


    print('Adding Wigglez')
    Wigglez_fs8 = pm.Deterministic("Wigglez_fs8",
                tt.as_tensor_variable(fs8_gp[Wigglez['idx']]+(fs8_gp[Wigglez['idx']+1]-fs8_gp[Wigglez['idx']])*Wigglez['U']))
    theory = tt.concatenate([theory, Wigglez_fs8])

    if 'DSS' in datasets:
        print('Adding DSS')
        with model:
            DSS_fs8 = pm.Deterministic("fs8_eBOSS", tt.as_tensor_variable(fs8_gp[DSS['idx']]))
            theory = tt.concatenate([theory, DSS_fs8])
        
#Sampling
    lkl= pm.MvNormal("lkl", mu=theory, cov=data_cov, observed=data)
    trace = pm.sample(n_samples, return_inferencedata=True, tune=n_tune)

#print r-stat
print(pm.summary(trace)['r_hat'][["Wm0", "ℓ","η"]])
print(pm.summary(trace)['mean'][["Wm0", "ℓ","η"]])

#Save
filename = data_comb
path = filename+'_'+mean_mode+'_'+challenge+ '_{}_{}'.format(n_samples, n_tune)
print(path)

n = np.array(trace.posterior["η"]).flatten()
l = np.array(trace.posterior["ℓ"]).flatten()
DHz = np.array(trace.posterior["DH_gp"])
DHz = DHz.reshape(-1, DHz.shape[-1])
Hz =np.array(trace.posterior["H_gp"])
Hz = Hz.reshape(-1, Hz.shape[-1])
H0_gp = np.array(trace.posterior["H0_gp"]).flatten()
Omega_m = np.array(trace.posterior["Wm0"]).flatten()
s8z = np.array(trace.posterior["s8_gp"])
s8z = s8z.reshape(-1, s8z.shape[-1])
fs8z = np.array(trace.posterior["fs8_gp"])
fs8z = fs8z.reshape(-1, fs8z.shape[-1])
s80 = np.array(trace.posterior["s80"]).flatten()
S80 = s80*np.sqrt(Omega_m/0.3)

os.mkdir(path)
np.savez(os.path.join(path,'samples.npz'), 
         z_arr = z_arr,
         n=n,
         l=l,
         DHz = DHz,
         Hz=Hz,
         s8z=s8z,
         fs8z=fs8z,
         H0_gp=H0_gp,
         Omega_m=Omega_m,
         s80=s80,
         S80=S80)