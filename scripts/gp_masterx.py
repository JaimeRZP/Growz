import numpy as np
import pymc3 as pm
import classy
import theano
import theano.tensor as tt
import os
import utils
import make_data
from scipy.linalg import block_diag

#Load data
z_max = 1110
res = 200
x_arr = np.linspace(0, np.log(1+z_max), res)
dx = np.mean(np.diff(x_arr))
z_arr = np.exp(x_arr)-1
a_arr = 1./(1+z_arr)

path = '/mnt/zfsusers/jaimerz/PhD/Growz/data/products'

tools = utils.utils()
c = tools.c
data = make_data.make_data(z_max, res , path)

DESI = data.get_DESI(new=True, mode='None')
H_DESI = data.get_DESI(new=True, mode='H')
dA_DESI = data.get_DESI(new=True, mode='dA')
fs8_DESI = data.get_DESI(new=True, mode='fs8')
WFIRST = data.get_WFIRST(new=True)
CC = data.get_CC(new=True)
DSS = data.get_DSS(new=True)
BOSS = data.get_BOSS(new=True)
eBOSS = data.get_eBOSS(new=True)
Wigglez = data.get_Wigglez(new=True)
DS17 = data.get_DS17(new=True)
CMB = data.get_CMB(new=True)
FCMB = data.get_FCMB(new=True)

n_samples = 1000
n_tune = 2500

datadict = {'DESI': DESI,
            'H_DESI': H_DESI,
            'dA_DESI': dA_DESI,
            'fs8_DESI': fs8_DESI,
            'WFIRST': WFIRST,
            'CC': CC,
            'DS17': DS17, 
            'BOSS': BOSS,
            'eBOSS': eBOSS,
            'Wigglez': Wigglez,
            'DSS': DSS,
            'CMB': CMB, 
            'FCMB': FCMB}

datasets = ['H_DESI', 'dA_DESI', 'fs8_DESI', 'CMB']
#datasets = ['BOSS', 'eBOSS', 'Wigglez', 'DSS', 'CMB']
#datasets = ['BOSS', 'CMB']
#datasets = ['DESI', 'CMB']

need_dM = ['DESI', 'dA_DESI', 'BOSS', 'eBOSS', 'Wigglez', 'DS17', 'CMB']
need_fs8 = ['DESI', 'fs8_DESI', 'BOSS', 'eBOSS', 'Wigglez', 'DSS']
need_rd = ['BOSS', 'eBOSS', 'CMB']

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
    #ℓ = pm.InverseGamma("ℓ", alpha=1, beta=2) 
    ℓ = pm.Uniform("ℓ", 0.001, 7) 
    η = pm.HalfNormal("η", sigma=0.3) 
    #wm0 = 0.1422 
    #wL0 = 0.307
    wm0 = pm.Uniform("wm0", 0., 0.45) 
    wL0 = pm.Uniform("wL0", 0., 0.45) 
    wr0 = (2.47+1.71)*10**-5
    gp_cov = η ** 2 * pm.gp.cov.ExpQuad(1, ℓ) + pm.gp.cov.WhiteNoise(1e-3)
    gp = pm.gp.Latent(cov_func=gp_cov)
    
    #Mean of the gp
    H = pm.Deterministic('H', 100*tt.sqrt(wm0*(1+z_arr)**3+wr0*(1+z_arr)**4+wL0))
    
    #Set up Gaussian process
    DH_gp = gp.prior("DH_gp", X=x_arr[:, None]) 
    H_gp = pm.Deterministic("H_gp", tt.as_tensor_variable(H*(1+DH_gp)))
    H0_gp = pm.Deterministic("H0_gp", tt.as_tensor_variable(H_gp[0]))
    
    if get_dM:
        dH_gp = pm.Deterministic("dH", tt.as_tensor_variable((c/1000)*(1+z_arr)/H_gp))
        dM_rec_gp = tt.zeros(len(z_arr)+1)
        dM_rec_gp = tt.inc_subtensor(dM_rec_gp[1:],
                  tt.as_tensor_variable(dx*tt.cumsum(dH_gp)))
        dM_trap_gp = tt.as_tensor_variable(0.5*(dM_rec_gp[1:]+dM_rec_gp[:-1])-0.5*dM_rec_gp[1])
        dM_gp = pm.Deterministic('dM_gp', dM_trap_gp)
        #dM_gp = pm.Deterministic('dM_gp', dM_rec_gp[:-1])
        dA_gp = pm.Deterministic('dA_gp', dM_gp/(1+z_arr))
        dL_gp = pm.Deterministic('dL_gp', dM_gp*(1+z_arr))
        
    if get_rd:
        #https://arxiv.org/pdf/2106.00428.pdf
        #wb0 =  0.02236
        wb0 =  pm.Uniform("wb0", 0.02224, 0.02228)
        a1 = 0.00785436
        a2 = 0.177084
        a3 = 0.00912388
        a4 = 0.618711
        a5 = 11.9611
        a6 = 2.81343
        a7 = 0.784719
        rd_gp = tt.as_tensor_variable(1/(a1*wb0**a2+a3*wm0**a4+a5*wb0**a6*wm0**a7))  
        
    if get_fs8:
        #s80 = 0.812
        s80 = pm.Normal("s80", 0.8, 0.5)
        Wm0 =  pm.Deterministic('Wm0', wm0*(100/H_gp[0])**2)
        E_gp = pm.Deterministic('E_gp', H_gp/H_gp[0])
        
        d = tt.zeros(len(z_arr))
        y = tt.zeros(len(z_arr))
        d = tt.inc_subtensor(d[-1], a_arr[-1])
        y = tt.inc_subtensor(y[-1], E_gp[0]*a_arr[-1]**3)
        for i in np.arange(1, len(z_arr)):
            A0 = -1.5*Wm0/(a_arr[-i]*E_gp[-i])
            B0 = -1./(a_arr[-i]**2*E_gp[-i])
            A1 = -1.5*Wm0/(a_arr[-(i+1)]*E_gp[-(i+1)])
            B1 = -1./(a_arr[-(i+1)]**2*E_gp[-(i+1)])
            y = tt.inc_subtensor(y[-(i+1)], (1-0.5*dx**2*A0*B0)*y[-i]-0.5*(A0+A1)*dx*d[-i])
            d = tt.inc_subtensor(d[-(i+1)], -0.5*(B0+B1)*dx*y[-i]+(1-0.5*dx**2*A0*B0)*d[-i])
        
        fs8_gp = pm.Deterministic('fs8_gp', s80*y/(a_arr**2*E_gp*d[0]))
        s8_gp = pm.Deterministic('s8_gp', s80*d/d[0])
    
    theory = tt.as_tensor_variable([])
    
#Modules
if 'H_DESI' in datasets:
    print('Adding H DESI')
    with model:
        DESI_H = pm.Deterministic('DESI_H',
                 tt.as_tensor_variable(H_gp[DESI['idx']]+(H_gp[DESI['idx']+1]-H_gp[DESI['idx']])*DESI['U']))
        theory = tt.concatenate([theory, DESI_H])

if 'dA_DESI' in datasets:
    print('Adding dA DESI')
    with model:
        DESI_dA = pm.Deterministic('DESI_dA',
                  tt.as_tensor_variable(dA_gp[DESI['idx']]+(dA_gp[DESI['idx']+1]-dA_gp[DESI['idx']])*DESI['U']))
        theory = tt.concatenate([theory, DESI_dA])

if 'fs8_DESI' in datasets:
    print('Adding fs8 DESI')
    with model:
        DESI_fs8 = pm.Deterministic('DESI_fs8',
                   tt.as_tensor_variable(fs8_gp[DESI['idx']]+(fs8_gp[DESI['idx']+1]-fs8_gp[DESI['idx']])*DESI['U']))
        theory = tt.concatenate([theory, DESI_fs8])
        
if 'WFIRST' in datasets:
    print('Adding WFIRST')
    with model:
        WFIRST_E = pm.Deterministic('WFIRST_E',
                   tt.as_tensor_variable(E_gp[WFIRST['idx']]+(E_gp[WFIRST['idx']+1]-E_gp[WFIRST['idx']])*WFIRST['U']))
        theory = tt.concatenate([theory, WFIRST_E])

if 'CC' in datasets:
    print('Adding CCs')
    with model:
        CC_H = pm.Deterministic("CC_H",
               tt.as_tensor_variable(H_gp[CC['idx']]+(H_gp[CC['idx']+1]-H_gp[CC['idx']])*CC['U']))
        theory = tt.concatenate([theory, CC_H])
        
if 'DS17' in datasets:
    print('Adding Pantheon')
    with model:
        M = pm.Normal('M', mu=-19.0, sigma=3)
        u_gp = tt.zeros(len(z_arr))
        u_gp = tt.inc_subtensor(u_gp[1:], tt.as_tensor_variable(5*tt.log10(dL_gp[1:])+25+M))
        u_gp = tt.inc_subtensor(u_gp[0], tt.as_tensor_variable(5*tt.log10(dL_gp[1])+25+M))
        DS17_u = pm.Deterministic("DS17_u",
                 tt.as_tensor_variable(u_gp[DS17['idx']]+(u_gp[DS17['idx']+1]-u_gp[DS17['idx']])*DS17['U']))
        theory = tt.concatenate([theory, DS17_u])
        
if 'BOSS' in datasets:
    print('Adding BOSS')
    with model:
        #Get alpha_perp and alpha_para 
        B_para_f = pm.Deterministic("B_para_f", H_gp*rd_gp/BOSS['rd'])
        B_perp_f = pm.Deterministic("B_perp_f", dM_gp*BOSS['rd']/rd_gp)
        
        B_para = pm.Deterministic("B_para", 
                    tt.as_tensor_variable(B_para_f[BOSS['idx']]+(B_para_f[BOSS['idx']+1]-B_para_f[BOSS['idx']])*BOSS['U']))
        B_perp = pm.Deterministic("B_perp", 
                tt.as_tensor_variable(B_perp_f[BOSS['idx']]+(B_perp_f[BOSS['idx']+1]-B_perp_f[BOSS['idx']])*BOSS['U']))
        B_fs8 = pm.Deterministic("B_fs8", 
                   tt.as_tensor_variable(fs8_gp[BOSS['idx']]+(fs8_gp[BOSS['idx']+1]-fs8_gp[BOSS['idx']])*BOSS['U']))
        B_perp = tt.ones_like(B_perp)
        B_fs8 = tt.ones_like(B_fs8)
        theory = tt.concatenate([theory, B_para, B_perp, B_fs8])
        
if 'eBOSS' in datasets:
    print('Adding eBOSS')
    with model:
        eB_para_f = pm.Deterministic("eB_para_f", dH_gp/rd_gp)
        eB_perp_f = pm.Deterministic("eB_perp_f", dH_gp/rd_gp)
        
        eB_para = pm.Deterministic("eB_para", 
                    tt.as_tensor_variable(eB_para_f[eBOSS['idx']]+(eB_para_f[eBOSS['idx']+1]-eB_para_f[eBOSS['idx']])*eBOSS['U']))
        eB_perp = pm.Deterministic("eB_perp", 
                tt.as_tensor_variable(eB_perp_f[eBOSS['idx']]+(eB_perp_f[eBOSS['idx']+1]-eB_perp_f[eBOSS['idx']])*eBOSS['U']))
        eB_fs8 = pm.Deterministic("eB_fs8", 
                   tt.as_tensor_variable(fs8_gp[eBOSS['idx']]+(fs8_gp[eBOSS['idx']+1]-fs8_gp[eBOSS['idx']])*eBOSS['U']))
        
        theory = tt.concatenate([theory, eB_para, eB_perp, eB_fs8])

if 'Wigglez' in datasets:
    print('Adding Wigglez')
    with model:
        Wigglez_fs8 = pm.Deterministic("Wigglez_fs8",
                    tt.as_tensor_variable(fs8_gp[Wigglez['idx']]+(fs8_gp[Wigglez['idx']+1]-fs8_gp[Wigglez['idx']])*Wigglez['U']))
        theory = tt.concatenate([theory, Wigglez_fs8])

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
        
if 'FCMB' in datasets:
    print('Adding FCMB')
    with model:
        FCMB_dM = pm.Deterministic('FCMB_dM',
                  tt.as_tensor_variable(dM_gp[FCMB['idx']]+(dM_gp[FCMB['idx']+1]-dM_gp[FCMB['idx']])*dM_CMB['U']))
        theory = tt.concatenate([theory, FCMB_dM])
        
with model:
    lkl= pm.MvNormal("lkl", mu=theory, cov=data_cov, observed=data)
    trace = pm.sample(n_samples, return_inferencedata=True, tune=n_tune)
    
#print r-stat
print(pm.summary(trace)['r_hat'][["ℓ","η"]])
print(pm.summary(trace)['mean'][["ℓ","η"]])

#Save
filename = ''
for dataset in datasets:
    filename+=dataset+'_'
path = filename+'{}_{}_t'.format(n_samples, n_tune)

n = np.array(trace.posterior["η"]).flatten()
l = np.array(trace.posterior["ℓ"]).flatten()
DHz = np.array(trace.posterior["DH_gp"])
DHz = DHz.reshape(-1, DHz.shape[-1])
Hz =np.array(trace.posterior["H_gp"])
Hz = Hz.reshape(-1, Hz.shape[-1])
H0 = np.array(trace.posterior["H0_gp"]).flatten()
h = H0/100
Omega_m = np.array(trace.posterior["wm0"]).flatten()/h**2
Omega_L = np.array(trace.posterior["wL0"]).flatten()/h**2

if get_dM:
    dMz = np.array(trace.posterior["dM_gp"])
    dMz = dMz.reshape(-1, dMz.shape[-1])
else:
    dMz = None

if get_rd:
    Omega_b = np.array(trace.posterior["wb0"]).flatten()/h**2
else:
    Omega_b = None
    
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

os.mkdir(path)
np.savez(os.path.join(path,'samples.npz'), 
         z_arr = z_arr,
         n=n,
         l=l,
         DHz = DHz,
         Hz=Hz,
         dMz=dMz,
         s8z=s8z,
         fs8z=fs8z,
         H0=H0,
         Omega_m=Omega_m,
         Omega_b=Omega_b,
         Omega_L=Omega_L,
         s80=s80,
         S80=S80)
