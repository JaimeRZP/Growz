import numpy as np
from pandas import read_table
    
def read_light_curve_parameters(path):
    with open(path, 'r') as text:
        clean_first_line = text.readline()[1:].strip()
        names = [e.strip().replace('3rd', 'third')
                 for e in clean_first_line.split()]

    lc_parameters = read_table(
        path, sep=' ', names=names, header=0, index_col=False)
    return lc_parameters

def get_signal(z_arr):
    ℓ_true = 0.2
    η_true = 6
    cov_func = η_true ** 2 * pm.gp.cov.Matern52(1, ℓ_true)
    mean_func = pm.gp.mean.Zero()
    return np.random.multivariate_normal(mean_func(z_arr[:, None]).eval(),
           cov_func(z_arr[:, None]).eval()+1e-8*np.eye(len(z_arr[:, None])),
           1).flatten()

def get_preds(z_arr, cosmo):
    c = 299792458.0
    H0 = cosmo.Hubble(0)
    h = H0/100000
    Wm0 = cosmo.Omega_m()
    H_arr = np.array([])
    dA_arr = np.array([])
    dL_arr = np.array([])
    f_arr = np.array([])
    s8_arr = np.array([])
    for z in z_arr: 
        H = cosmo.Hubble(z)
        dA = cosmo.angular_distance(z)
        dL = cosmo.luminosity_distance(z)
        f = cosmo.scale_independent_growth_factor_f(z)
        s8 = cosmo.sigma(8./cosmo.h(),z)
        s8_arr = np.append(s8_arr, s8)
        f_arr = np.append(f_arr, f)
        dL_arr = np.append(dL_arr, dL)
        dA_arr = np.append(dA_arr, dA)
        H_arr = np.append(H_arr, H)

    dM_arr = dA_arr*(1+z_arr)
    Hkms_arr = H_arr*c/1000
    preds ={'H_arr': H_arr, 'dA_arr': dA_arr,
           'dL_arr': dL_arr, 'dM_arr': dM_arr,
           'Hkms_arr': Hkms_arr, 'f_arr': f_arr, 
           's8_arr': s8_arr, 'fs8_arr': s8_arr*f_arr}
    return preds

def make_fs8(H, x_arr, wm0, s80):
    z_arr = np.exp(x_arr)-1
    a_arr = 1./(1+z_arr) 
    dx = np.mean(np.diff(x_arr))
    E = H/H[0]
    Om = wm0*(100/H[0])**2
    xx = x_arr[::-1]
    ee = E[::-1]
    aa = np.exp(-xx)
    dx = np.mean(np.diff(xx))

    nz = len(aa)
    dd = np.zeros(nz)
    yy = np.zeros(nz)
    dd[0] = aa[0]
    yy[0] = aa[0]**3*E[0]

    for i in range(nz-1):
        A0 = -1.5*Om/(aa[i]*ee[i])
        B0 = -1./(aa[i]**2*ee[i])
        A1 = -1.5*Om/(aa[i+1]*ee[i+1])
        B1 = -1./(aa[i+1]**2*ee[i+1])
        yy[i+1]=(1+0.5*dx**2*A0*B0)*yy[i]+0.5*(A0+A1)*dx*dd[i]
        dd[i+1]=0.5*(B0+B1)*dx*yy[i]+(1+0.5*dx**2*A0*B0)*dd[i]

    y = yy[::-1]
    d = dd[::-1]
    fs8 = s80*y/(a_arr**2*E*d[0])
    s8 = s80*d/d[0]

    return s8, fs8

def make_dM(H, x_arr):
    z_arr = np.exp(x_arr)-1
    a_arr = 1./(1+z_arr) 
    dx = np.mean(np.diff(x_arr))
    dM = np.zeros(len(z_arr)+1)
    dM[1:] = dx*np.cumsum((1+z_arr)/H)
    dM = 0.5*(dM[1:]+dM[:-1])-0.5*dM[1]
    return dM

def make_rd(wm0, wb0):
    a1 = 0.00785436
    a2 = 0.177084
    a3 = 0.00912388
    a4 = 0.618711
    a5 = 11.9611
    a6 = 2.81343
    a7 = 0.784719
    rd_gp = 1/(a1*wb0**a2+a3*wm0**a4+a5*(wb0**a6)*(wm0**a7))
    return rd_gp
