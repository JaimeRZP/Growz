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
nz_int = 200
x_int = np.linspace(0, np.log(1+z_max), nz_int)
z_int = np.exp(x_int)-1
a_int = 1./(1+z_int)
dx_int = np.mean(np.diff(x_int))

nz_Hgp = 150
x_Hgp = np.linspace(0, np.log(1+z_max), nz_Hgp)
z_Hgp = np.exp(x_Hgp)-1
a_Hgp = 1./(1+z_Hgp)
dx_Hgp = np.mean(np.diff(x_Hgp))

nz_Xigp = 30
x_Xigp = np.linspace(0, np.log(1+z_max), nz_Xigp)
z_Xigp = np.exp(x_Xigp)-1
a_Xigp = 1./(1+z_Xigp)
dx_Xigp = np.mean(np.diff(x_Xigp))

path = '/mnt/zfsusers/jaimerz/PhD/Growz/data/' 
challenge = None #'cosmo61'
if challenge is not None:
    path += 'challenge/'+'cosmo{}_seed100{}'.format(challenge[-2], challenge[-1])

print('data path: ', path)
mean_path =  None #'LCDM_cosmo44_10000_10000'
mean_mode = 'Planck'
data_class = MakeData(z_max, nz_int, path,
                      cosmo_mode=mean_mode,
                      cosmo_path=mean_path)
c = data_class.c

which_DESI = 'DESI'
DESI = data_class.get_synthetic(which_DESI, new=True)
CC = data_class.get_CC(new=True)
DSS = data_class.get_DSS(new=True)
BOSS = data_class.get_BOSS(new=True)
geo_BOSS = data_class.get_BOSS(new=True, mode='geo')
gro_BOSS = data_class.get_BOSS(new=True, mode='gro')
eBOSS = data_class.get_eBOSS(new=True)
geo_eBOSS = data_class.get_eBOSS(new=True, mode='geo')
gro_eBOSS = data_class.get_eBOSS(new=True, mode='gro')
Wigglez = data_class.get_Wigglez(new=True)
Vipers = data_class.get_Vipers(new=True)
sixdF = data_class.get_6dF(new=True)
FastSound = data_class.get_FastSound(new=True)
DS17 = data_class.get_DS17(new=True)
CMB = data_class.get_CMB(new=True)

n_samples = 2 #3000
n_tune = 2 #3000
datadict = {'DESI': DESI,
            'CC': CC,
            'DS17': DS17, 
            'BOSS': BOSS,
            'geo_BOSS': geo_BOSS,
            'gro_BOSS': gro_BOSS,
            'eBOSS': eBOSS,
            'geo_eBOSS': geo_eBOSS,
            'gro_eBOSS': gro_eBOSS,
            'Wigglez': Wigglez,
            'Vipers': Vipers,
            '6dF': sixdF,
            'FastSound': FastSound,
            'DSS': DSS,
            'CMB': CMB}

data_comb = 'gro' # All, All_CMB, SDSS, SDSS_CMB, Add, Add_CMB
data_combs = {'All': ['CC', 'DS17', 'BOSS', 'eBOSS', 'Wigglez', '6dF', 'FastSound', 'DSS'],
             'All_CMB': ['CC', 'DS17', 'BOSS', 'eBOSS', 'Wigglez', 'Vipers', '6dF', 'FastSound', 'DSS', 'CMB'],
             'geo': ['CC', 'DS17', 'geo_BOSS', 'geo_eBOSS', 'CMB'],
             'gro': ['gro_BOSS', 'gro_eBOSS', 'Wigglez', 'Vipers', '6dF', 'FastSound', 'DSS'],
             'DESI_CMB': ['DESI', 'CMB']}
datasets = data_combs[data_comb]

need_dM = ['DESI', 'BOSS', 'eBOSS', 'geo_BOSS', 'geo_eBOSS',
           'Wigglez', 'DS17', 'CMB']
need_fs8 = ['DESI', 'BOSS', 'eBOSS', 'gro_BOSS', 
            'gro_eBOSS', 'Wigglez', 'Vipers', '6dF', 'FastSound', 'DSS']
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

idx_Hgp = data_class.make_idx(z_int, z_Hgp)
U_Hgp = data_class.make_U(z_int, z_Hgp, idx_Hgp)

idx_Xigp = data_class.make_idx(z_int, z_Xigp)
U_Xigp = data_class.make_U(z_int, z_Xigp, idx_Xigp)

#base model
with pm.Model() as model:
    ℓ_H = pm.Uniform("ℓ_H", 0.01, 6) 
    η_H = pm.HalfNormal("η_H", sigma=0.2) 
    A0 = pm.Normal("A0", 1, 0.2)
    H0 = data_class.H0
    Wm0 = data_class.Wm0
    Wr0 = data_class.Wr0
    WL0 = data_class.WL0
    H_gp_cov = η_H ** 2 * pm.gp.cov.ExpQuad(1, ℓ_H) + pm.gp.cov.WhiteNoise(1e-5)
    H_gp = pm.gp.Latent(cov_func=H_gp_cov)
    
    #Mean of the gp
    H = pm.Deterministic('H', H0*tt.sqrt(Wm0*(1+z_Hgp)**3+Wr0*(1+z_Hgp)**4+WL0))
    DH_gp = H_gp.prior("DH_gp", X=x_Hgp[:, None]) 
    H_gp = pm.Deterministic("H_gp", tt.as_tensor_variable(H*A0*(1+DH_gp)))
    H0_gp = pm.Deterministic("H0_gp", tt.as_tensor_variable(H_gp[0]))
    
    H_int = tt.zeros(nz_int)
    H_int = tt.inc_subtensor(H_int[1:], H_gp[idx_Hgp[1:]]+(H_gp[idx_Hgp[1:]+1]-H_gp[idx_Hgp[1:]])*U_Hgp[1:])
    H_int = tt.inc_subtensor(H_int[0], H_gp[0])
    H_int = pm.Deterministic('H_int', H_int)
    
    if get_dM:
        dH_gp = pm.Deterministic("dH", tt.as_tensor_variable((c/1000)/H_int))
        dM_rec_gp = tt.zeros(len(z_int)+1)
        dM_rec_gp = tt.inc_subtensor(dM_rec_gp[1:],
                  tt.as_tensor_variable(dx_int*tt.cumsum(dH_gp*(1+z_int))))
        dM_trap_gp = tt.as_tensor_variable(0.5*(dM_rec_gp[1:]+dM_rec_gp[:-1])-0.5*dM_rec_gp[1])
        dM_gp = pm.Deterministic('dM_gp', dM_trap_gp)
        dA_gp = pm.Deterministic('dA_gp', dM_gp/(1+z_int))
        dL_gp = pm.Deterministic('dL_gp', dM_gp*(1+z_int))
        
    if get_rd:
        rd_gp = pm.Normal("rd_gp", 150, 5)
        
    if get_fs8:
        ℓ_Xi = pm.Uniform("ℓ_Xi", 0.01, 6) 
        η_Xi = pm.HalfNormal("η_Xi", sigma=0.5)
        Xi_gp_cov = η_Xi ** 2 * pm.gp.cov.ExpQuad(1, ℓ_Xi) + pm.gp.cov.WhiteNoise(1e-3)
        Xi_gp = pm.gp.Latent(cov_func=Xi_gp_cov)
        DXi_gp = Xi_gp.prior("DXi_gp", X=x_Xigp[:, None]) 
        Xi_gp = pm.Deterministic("Xi_gp", tt.as_tensor_variable(np.ones_like(z_Xigp)+DXi_gp))
        
        Xi_int = tt.zeros(nz_int)
        Xi_int = tt.inc_subtensor(Xi_int[1:], Xi_gp[idx_Xigp[1:]]+(Xi_gp[idx_Xigp[1:]+1]-Xi_gp[idx_Xigp[1:]])*U_Xigp[1:])
        Xi_int = tt.inc_subtensor(Xi_int[0], Xi_gp[0])
        Xi_int = pm.Deterministic('Xi_int', Xi_int)
        
        s80 = pm.Normal("s80", 0.8, 0.5)
        E = H_int/H_int[0]
        Om = tt.as_tensor_variable(Xi_int*Wm0)
        Omm = Om[::-1]
        xx = x_int[::-1]
        ee = E[::-1]
        aa = np.exp(-xx)
        dx = np.mean(np.diff(xx))

        dd = tt.zeros(nz_int)
        yy = tt.zeros(nz_int)
        dd = tt.inc_subtensor(dd[0], aa[0])
        yy = tt.inc_subtensor(yy[0], aa[0]**3*E[0])

        for i in range(nz_int-1):
            A0 = -1.5*Omm[i]/(aa[i]*ee[i])
            B0 = -1./(aa[i]**2*ee[i])
            A1 = -1.5*Omm[i]/(aa[i+1]*ee[i+1])
            B1 = -1./(aa[i+1]**2*ee[i+1])
            yy = tt.inc_subtensor(yy[i+1], (1+0.5*dx**2*A0*B0)*yy[i]+0.5*(A0+A1)*dx*dd[i])
            dd = tt.inc_subtensor(dd[i+1],0.5*(B0+B1)*dx*yy[i]+(1+0.5*dx**2*A0*B0)*dd[i])
        
        y = tt.as_tensor_variable(yy[::-1])
        d = tt.as_tensor_variable(dd[::-1])
        
        fs8_gp = pm.Deterministic('fs8_gp', s80*y/(a_int**2*E*d[0]))
        s8_gp = pm.Deterministic('s8_gp', s80*d/d[0])

    theory = tt.as_tensor_variable([])

#Modules
if 'DESI' in datasets:
    print('Adding DESI')
    with model:
        DESI_H = pm.Deterministic('DESI_H',
                 tt.as_tensor_variable(H_int[DESI['idx']]+(H_int[DESI['idx']+1]-H_int[DESI['idx']])*DESI['U']))
        DESI_dA = pm.Deterministic('DESI_dA',
                  tt.as_tensor_variable(dA_gp[DESI['idx']]+(dA_gp[DESI['idx']+1]-dA_gp[DESI['idx']])*DESI['U']))
        DESI_fs8 = pm.Deterministic('DESI_fs8',
                   tt.as_tensor_variable(fs8_gp[DESI['idx']]+(fs8_gp[DESI['idx']+1]-fs8_gp[DESI['idx']])*DESI['U']))
        theory = tt.concatenate([theory, DESI_H, DESI_dA, DESI_fs8])

if 'CC' in datasets:
    print('Adding CCs')
    with model:
        CC_H = pm.Deterministic("CC_H",
               tt.as_tensor_variable(H_int[CC['idx']]+(H_int[CC['idx']+1]-H_int[CC['idx']])*CC['U']))
        theory = tt.concatenate([theory, CC_H])
        
if 'DS17' in datasets:
    print('Adding Pantheon')
    with model:
        M = pm.Normal('M', mu=-19.0, sigma=1)
        DS17_dL = tt.as_tensor_variable(dL_gp[DS17['idx']]+(dL_gp[DS17['idx']+1]-dL_gp[DS17['idx']])*DS17['U'])
        DS17_u = pm.Deterministic("DS17_dL",
                 tt.as_tensor_variable(5*tt.log10(DS17_dL)+25+M))
        theory = tt.concatenate([theory, DS17_u])
        
if 'BOSS' in datasets:
    print('Adding BOSS')
    with model:
        B_H = tt.as_tensor_variable(H_int[BOSS['idx']]+(H_int[BOSS['idx']+1]-H_int[BOSS['idx']])*BOSS['U'])
        B_dM = tt.as_tensor_variable(dM_gp[BOSS['idx']]+(dM_gp[BOSS['idx']+1]-dM_gp[BOSS['idx']])*BOSS['U'])
        B_fs8 = pm.Deterministic("B_fs8", 
                   tt.as_tensor_variable(fs8_gp[BOSS['idx']]+(fs8_gp[BOSS['idx']+1]-fs8_gp[BOSS['idx']])*BOSS['U']))
        #Get alpha_perp and alpha_para 
        B_para = pm.Deterministic("B_para", B_H*rd_gp/BOSS['rd'])
        B_perp = pm.Deterministic("B_perp", B_dM*BOSS['rd']/rd_gp)
        theory = tt.concatenate([theory, B_para, B_perp, B_fs8])
        
if 'geo_BOSS' in datasets:
    print('Adding geo_BOSS')
    with model:
        B_H = tt.as_tensor_variable(H_int[BOSS['idx']]+(H_int[BOSS['idx']+1]-H_int[BOSS['idx']])*BOSS['U'])
        B_dM = tt.as_tensor_variable(dM_gp[BOSS['idx']]+(dM_gp[BOSS['idx']+1]-dM_gp[BOSS['idx']])*BOSS['U'])
        #Get alpha_perp and alpha_para 
        B_para = pm.Deterministic("B_para", B_H*rd_gp/BOSS['rd'])
        B_perp = pm.Deterministic("B_perp", B_dM*BOSS['rd']/rd_gp)
        theory = tt.concatenate([theory, B_para, B_perp])
        
if 'gro_BOSS' in datasets:
    print('Adding gro_BOSS')
    with model:
        B_fs8 = pm.Deterministic("B_fs8", 
                   tt.as_tensor_variable(fs8_gp[BOSS['idx']]+(fs8_gp[BOSS['idx']+1]-fs8_gp[BOSS['idx']])*BOSS['U']))
        theory = tt.concatenate([theory, B_fs8])
        
if 'eBOSS' in datasets:
    print('Adding eBOSS')
    with model:
        eB_dH = tt.as_tensor_variable(dH_gp[eBOSS['idx']]+(dH_gp[eBOSS['idx']+1]-dH_gp[eBOSS['idx']])*eBOSS['U'])
        eB_dM = tt.as_tensor_variable(dM_gp[eBOSS['idx']]+(dM_gp[eBOSS['idx']+1]-dM_gp[eBOSS['idx']])*eBOSS['U'])
        eB_fs8 = pm.Deterministic("eB_fs8", 
                   tt.as_tensor_variable(fs8_gp[eBOSS['idx']]+(fs8_gp[eBOSS['idx']+1]-fs8_gp[eBOSS['idx']])*eBOSS['U']))
        eB_para = pm.Deterministic("eB_para", eB_dH/rd_gp)
        eB_perp = pm.Deterministic("eB_perp", eB_dM/rd_gp)
        theory = tt.concatenate([theory, eB_para, eB_perp, eB_fs8])
        
if 'geo_eBOSS' in datasets:
    print('Adding geo_eBOSS')
    with model:
        eB_dH = tt.as_tensor_variable(dH_gp[eBOSS['idx']]+(dH_gp[eBOSS['idx']+1]-dH_gp[eBOSS['idx']])*eBOSS['U'])
        eB_dM = tt.as_tensor_variable(dM_gp[eBOSS['idx']]+(dM_gp[eBOSS['idx']+1]-dM_gp[eBOSS['idx']])*eBOSS['U'])
        eB_para = pm.Deterministic("eB_para", eB_dH/rd_gp)
        eB_perp = pm.Deterministic("eB_perp", eB_dM/rd_gp)
        theory = tt.concatenate([theory, eB_para, eB_perp])

if 'gro_eBOSS' in datasets:
    print('Adding gro_eBOSS')
    with model:
        eB_fs8 = pm.Deterministic("eB_fs8", 
                   tt.as_tensor_variable(fs8_gp[eBOSS['idx']]+(fs8_gp[eBOSS['idx']+1]-fs8_gp[eBOSS['idx']])*eBOSS['U']))
        theory = tt.concatenate([theory, eB_fs8])

if 'Wigglez' in datasets:
    print('Adding Wigglez')
    with model:
        Wigglez_fs8 = pm.Deterministic("Wigglez_fs8",
                    tt.as_tensor_variable(fs8_gp[Wigglez['idx']]+(fs8_gp[Wigglez['idx']+1]-fs8_gp[Wigglez['idx']])*Wigglez['U']))
        theory = tt.concatenate([theory, Wigglez_fs8])
        
if 'Vipers' in datasets:
    print('Adding Vipers')
    with model:
        Vipers_fs8 = pm.Deterministic("Vipers_fs8",
                    tt.as_tensor_variable(fs8_gp[Vipers['idx']]+(fs8_gp[Vipers['idx']+1]-fs8_gp[Vipers['idx']])*Vipers['U']))
        theory = tt.concatenate([theory, Vipers_fs8])
        
if '6dF' in datasets:
    print('Adding 6dF')
    with model:
        sixdF_fs8 = pm.Deterministic("6dF_fs8",
                    tt.as_tensor_variable(fs8_gp[sixdF['idx']]+(fs8_gp[sixdF['idx']+1]-fs8_gp[sixdF['idx']])*sixdF['U']))
        theory = tt.concatenate([theory, sixdF_fs8])
        
if 'FastSound' in datasets:
    print('Adding FastSound')
    with model:
        FastSound_fs8 = pm.Deterministic("FastSound_fs8",
                    tt.as_tensor_variable(fs8_gp[FastSound['idx']]+(fs8_gp[FastSound['idx']+1]-fs8_gp[FastSound['idx']])*FastSound['U']))
        theory = tt.concatenate([theory, FastSound_fs8])

if 'DSS' in datasets:
    print('Adding DSS')
    with model:
        DSS_fs8 = pm.Deterministic("fs8_eBOSS", tt.as_tensor_variable(fs8_gp[DSS['idx']]))
        theory = tt.concatenate([theory, DSS_fs8])

if 'CMB' in datasets:
    print('Adding CMB')
    with model:
        dM_star = tt.as_tensor_variable(dM_gp[CMB['idx']]+(dM_gp[CMB['idx']+1]-dM_gp[CMB['idx']])*CMB['U'])
        t100 = pm.Deterministic('t100', 100*rd_gp/dM_star) 
        theory = tt.concatenate([theory, t100])
        
#Sampling
with model:
    lkl= pm.MvNormal("lkl", mu=theory, cov=data_cov, observed=data)
    trace = pm.sample(n_samples, return_inferencedata=True, tune=n_tune, target_accept=0.97)

#print r-stat
print(pm.summary(trace)['r_hat'][["ℓ_H", "η_H", "ℓ_Xi", "η_Xi"]])
print(pm.summary(trace)['mean'][["ℓ_H", "η_H", "ℓ_Xi", "η_Xi"]])

#Save
if data_comb=="DESI_CMB":
    filename = which_DESI+"_CMB"
else:
    filename = data_comb
    
filename += '_Xi_H_lite_{}_{}'.format(n_samples, n_tune)
print(filename)
A0 = np.array(trace.posterior["A0"]).flatten()
n_H = np.array(trace.posterior["η_H"]).flatten()
l_H = np.array(trace.posterior["ℓ_H"]).flatten()
DHz = np.array(trace.posterior["DH_gp"])
DHz = DHz.reshape(-1, DHz.shape[-1])
Hz = np.array(trace.posterior["H_gp"])
Hz = Hz.reshape(-1, Hz.shape[-1])
Hz_int = np.array(trace.posterior["H_int"])
Hz_int = Hz_int.reshape(-1, Hz_int.shape[-1])
H0_gp = np.array(trace.posterior["H0_gp"]).flatten()

if get_dM:
    dMz = np.array(trace.posterior["dM_gp"])
    dMz = dMz.reshape(-1, dMz.shape[-1])
else:
    dMz = None

if get_rd:
    rd = np.array(trace.posterior["rd_gp"]).flatten()
else:
    rd = None
    
if get_fs8:
    n_Xi = np.array(trace.posterior["η_Xi"]).flatten()
    l_Xi = np.array(trace.posterior["ℓ_Xi"]).flatten()
    DXiz = np.array(trace.posterior["DXi_gp"])
    DXiz = DXiz.reshape(-1, DXiz.shape[-1])
    Xiz = np.array(trace.posterior["Xi_gp"])
    Xiz = Xiz.reshape(-1, Xiz.shape[-1])
    Xiz_int = np.array(trace.posterior["Xi_int"])
    Xiz_int = Xiz_int.reshape(-1, Xiz_int.shape[-1])
    s8z = np.array(trace.posterior["s8_gp"])
    s8z = s8z.reshape(-1, s8z.shape[-1])
    fs8z = np.array(trace.posterior["fs8_gp"])
    fs8z = fs8z.reshape(-1, fs8z.shape[-1])
    s80 = np.array(trace.posterior["s80"]).flatten()

else: 
    A0 = None
    n_Xi = None
    l_Xi = None
    DXiz = None
    Xiz = None
    Xiz_int = None
    s8z = None 
    fs8z = None
    s80 = None

if 'DS17' in datasets:
    M = np.array(trace.posterior["M"]).flatten()
else:
    M = None

os.mkdir(filename)
np.savez(os.path.join(filename,'samples.npz'), 
         z_int=z_int,
         z_Hgp=z_Hgp,
         z_Xigp=z_Xigp,
         A0=A0,
         n_Xi=n_Xi,
         l_Xi=l_Xi,
         n_H=n_H,
         l_H=l_H,
         DHz=DHz,
         DXiz=DXiz,
         Xiz=Xiz,
         Hz=Hz,
         Xiz_int=Xiz_int,
         Hz_int=Hz_int,
         dMz=dMz,
         s8z=s8z,
         fs8z=fs8z,
         H0_gp=H0_gp,
         s80=s80,
         rd=rd,
         M=M)
