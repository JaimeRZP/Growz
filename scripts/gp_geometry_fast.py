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

challenge = 'challenge/cosmo4_seed1004'
path = '/mnt/zfsusers/jaimerz/PhD/Growz/data/'+challenge

mean_path = None #'LCDM_cosmo44_10000_10000'
mean_mode = None #'other'
data_class = MakeData(z_max, res, path,
                      cosmo_mode=mean_mode,
                      cosmo_path=mean_path)
Planck = data_class.Planck
z_planck = data_class.z_planck
c = data_class.c


geo_DESI = data_class.get_DESI(new=True, mode='geo')
WFIRST = data_class.get_WFIRST(new=True)
CC = data_class.get_CC(new=True)
DSS = data_class.get_DSS(new=True)
geo_BOSS = data_class.get_BOSS(new=True, mode='geo')
geo_eBOSS = data_class.get_eBOSS(new=True, mode='geo')
CMB = data_class.get_CMB(new=True)

n_samples = 10000
n_tune = 10000
datadict = {'geo_DESI': geo_DESI,
            'WFIRST': WFIRST,
            'CC': CC,
            'DS17': DS17, 
            'geo_BOSS': geo_BOSS,
            'geo_eBOSS': geo_eBOSS,
            'CMB': CMB}

data_comb = 'DESI_gro' # All, All_CMB, SDSS, SDSS_CMB, Add, Add_CMB
data_combs = {'All_CMB_geo': ['CC', 'DS17', 'geo_BOSS', 'geo_eBOSS', 'CMB']
             'DESI_CMB_geo': ['geo_DESI', 'CMB'], 
             'WFIRST_CMB': ['WFIRST', 'CMB']}
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
    wm0_geo = data_class.wm0 
    wr0 = data_class.wr0
    wL0 = data_class.wL0 
    gp_cov = η ** 2 * pm.gp.cov.ExpQuad(1, ℓ) + pm.gp.cov.WhiteNoise(1e-3)
    gp = pm.gp.Latent(cov_func=gp_cov)
    
    #Mean of the gp
    H = pm.Deterministic('H', 100*tt.sqrt(wm0_geo*(1+z_arr)**3+wr0*(1+z_arr)**4+wL0))
    
    #Set up Gaussian process
    DH_gp = gp.prior("DH_gp", X=x_arr[:, None]) 
    H_gp = pm.Deterministic("H_gp", tt.as_tensor_variable(H*(1+DH_gp)))
    H0_gp = pm.Deterministic("H0_gp", tt.as_tensor_variable(H_gp[0]))
    
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
    wb0 =  pm.Uniform("wb0", 0.022, 0.023)
    a1 = 0.00785436
    a2 = 0.177084
    a3 = 0.00912388
    a4 = 0.618711
    a5 = 11.9611
    a6 = 2.81343
    a7 = 0.784719
    rd_gp = pm.Deterministic("rd_gp", 1/(a1*wb0**a2+a3*wm0_geo**a4+a5*wb0**a6*wm0_geo**a7)) 
        
    theory = tt.as_tensor_variable([])
    
 

    print('Adding DESI_geo')
    DESI_H = pm.Deterministic('DESI_H',
             tt.as_tensor_variable(H_gp[DESI['idx']]+(H_gp[DESI['idx']+1]-H_gp[DESI['idx']])*DESI['U']))
    DESI_dA = pm.Deterministic('DESI_dA',
              tt.as_tensor_variable(dA_gp[DESI['idx']]+(dA_gp[DESI['idx']+1]-dA_gp[DESI['idx']])*DESI['U']))
    theory = tt.concatenate([theory, DESI_H, DESI_dA])
    
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
        
    print('Adding geo_BOSS')
    B_H = tt.as_tensor_variable(H_gp[BOSS['idx']]+(H_gp[BOSS['idx']+1]-H_gp[BOSS['idx']])*BOSS['U'])
    B_dM = tt.as_tensor_variable(dM_gp[BOSS['idx']]+(dM_gp[BOSS['idx']+1]-dM_gp[BOSS['idx']])*BOSS['U'])
    #Get alpha_perp and alpha_para 
    B_para = pm.Deterministic("B_para", B_H*rd_gp/BOSS['rd'])
    B_perp = pm.Deterministic("B_perp", B_dM*BOSS['rd']/rd_gp)
    theory = tt.concatenate([theory, B_para, B_perp])
        
    print('Adding geo_eBOSS')
    eB_dH = tt.as_tensor_variable(dH_gp[eBOSS['idx']]+(dH_gp[eBOSS['idx']+1]-dH_gp[eBOSS['idx']])*eBOSS['U'])
    eB_dM = tt.as_tensor_variable(dM_gp[eBOSS['idx']]+(dM_gp[eBOSS['idx']+1]-dM_gp[eBOSS['idx']])*eBOSS['U'])
    eB_para = pm.Deterministic("eB_para", eB_dH/rd_gp)
    eB_perp = pm.Deterministic("eB_perp", eB_dM/rd_gp)
    theory = tt.concatenate([theory, eB_para, eB_perp])

    print('Adding CMB')
    dM_star = tt.as_tensor_variable(dM_gp[CMB['idx']]+(dM_gp[CMB['idx']+1]-dM_gp[CMB['idx']])*CMB['U'])
    t100 = pm.Deterministic('t100', 100*rd_gp/dM_star) 
    theory = tt.concatenate([theory, t100])
        
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
dMz = np.array(trace.posterior["dM_gp"])
dMz = dMz.reshape(-1, dMz.shape[-1])
rd = np.array(trace.posterior["rd_gp"]).flatten()
omega_b = np.array(trace.posterior["wb0"]).flatten()
M = np.array(trace.posterior["M"]).flatten()


os.mkdir(path)
np.savez(os.path.join(path,'samples.npz'), 
         z_arr = z_arr,
         n=n,
         l=l,
         DHz = DHz,
         Hz=Hz,
         dMz=dMz,
         H0_gp=H0_gp,
         Omega_m=Omega_m,
         omega_b=omega_b,
         rd=rd,
         M=M)