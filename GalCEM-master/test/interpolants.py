from numpy.core.numeric import NaN
from numpy.core.shape_base import block
#from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator
import scipy.interpolate as interp
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot,cm

from prep.setup import ZA_sorted

class GalChemInterpolant(object):
    def __init__(self,dfx,dfy):
        self.xnames = list(dfx.columns)
        self.yname = str(dfy.name)
        x = dfx.to_numpy()
        y = dfy.to_numpy() 
        self.linear_nd_interpolator = interp.LinearNDInterpolator(x,y,rescale=True)
        self.nearest_nd_interpolator = interp.NearestNDInterpolator(x,y,rescale=True)
    def __call__(self,dfx):
        x = dfx[self.xnames].to_numpy() if type(dfx)==pd.DataFrame else dfx
        yhat = self.linear_nd_interpolator(x)
        yhat[np.isnan(yhat)] = self.nearest_nd_interpolator(x[np.isnan(yhat)])
        ysave = yhat.copy
        return yhat

def interpolant_func(dfx,dfy):
    yhat = interp.LinearNDInterpolator(dfx,dfy,rescale=True)
    if yhat == NaN:
        yhat = interp.NearestNDInterpolator(dfx,dfy,rescale=True)
    return yhat

#def call_model(x,y):
#    if x.empty:
#        return None
#    else:
#        return interpolant_func(x,y)

def save_model(x,y):
    models = []
    for i,val in enumerate(ZA_sorted):
        print(f'{i=}')
        if x[i].empty:
            print(f'{x[i]=}')
            models.append(None)
        else:
            fit = interpolant_func(x[i],y[i])
            models.append(fit)
    # pickle.dump(model_k10,open('input/yields/lims/k10/processed/models.pkl','wb'))
    return models

def fit_2d_interpolant(dfx,dfy,tag,test,y_log10_scaled,view_angle):
    nticks = 64
    sepfrac = 0.1
    interpolant = model = GalChemInterpolant(dfx,dfy) # https://docs.scipy.org/doc/scipy/reference/interpolate.html
    if not test: return interpolant
    inputs,output = list(dfx.columns),str(dfy.name)
    name = '%s.%s'%(tag,output)
    x,y = dfx.to_numpy(),dfy.to_numpy()
    yhat = interpolant(dfx)
    yhat_tf = 10**yhat if y_log10_scaled else yhat
    ytf = 10**y if y_log10_scaled else y
    eps_abs = np.abs(yhat_tf-ytf)
    print('%s metrics'%name)
    print('\tRMSE: %.1e'%np.sqrt(np.mean(eps_abs**2)))
    print('\tMAE: %.1e'%np.mean(eps_abs))
    print('\tMax Abs Error: %.1e\n'%eps_abs.max())
    x0min,x0max = x[:,0].min(),x[:,0].max()
    x0_sep = x0max-x0min
    x0_ticks = np.linspace(x0min-sepfrac*x0_sep,x0max+sepfrac*x0_sep,nticks)
    x1min,x1max = x[:,1].min(),x[:,1].max()
    x1_sep = x1max-x1min
    x1_ticks = np.linspace(x1min-sepfrac*x1_sep,x1max+sepfrac*x1_sep,nticks)
    x0mesh,x1mesh = np.meshgrid(x0_ticks,x1_ticks)
    xquery = np.hstack([x0mesh.reshape(-1,1),x1mesh.reshape(-1,1)])
    yquery = model(xquery)
    #print(f'{xquery=}')
    #print(f'{yquery=}')
    ymesh = yquery.reshape(x1mesh.shape)
    fig = pyplot.figure(figsize=(15,5))
    ax = fig.add_subplot(1,2,1,projection='3d')
    surf = ax.plot_surface(x0mesh,x1mesh,ymesh,cmap=cm.Greys,alpha=.9,vmin=ymesh.min(),vmax=ymesh.max())
    ax.scatter(x[:,0],x[:,1],y,color='r')
    ax.set_xlabel(inputs[0])
    ax.set_ylabel(inputs[1])
    ax.set_zlabel(output)
    ax.view_init(azim=view_angle)
    ax = fig.add_subplot(1,2,2)
    contour = ax.contourf(x0mesh,x1mesh,ymesh,cmap=cm.Greys,alpha=.95,vmin=ymesh.min(),vmax=ymesh.max())
    fig.colorbar(contour,ax=None,shrink=0.5,aspect=5)
    ax.scatter(x[:,0],x[:,1],color='r')
    ax.set_xlabel(inputs[0])
    ax.set_ylabel(inputs[1])   
    fig.suptitle(tag)
    pyplot.show(block=False)
    fig.savefig('figures/interpolants/%s.pdf'%name,format='pdf',bbox_inches='tight')    

def fit_3d_interpolant(dfx,dfy,tag,test,y_log10_scaled,view_angle):
    nticks = 64
    sepfrac = 0.1
    interpolant = model = GalChemInterpolant(dfx,dfy) # https://docs.scipy.org/doc/scipy/reference/interpolate.html
    if not test: return interpolant
    inputs,output = list(dfx.columns),str(dfy.name)
    name = '%s.%s'%(tag,output)
    x,y = dfx.to_numpy(),dfy.to_numpy()
    yhat = interpolant(dfx)
    yhat_tf = 10**yhat if y_log10_scaled else yhat
    ytf = 10**y if y_log10_scaled else y
    eps_abs = np.abs(yhat_tf-ytf)
    print('%s metrics'%name)
    print('\tRMSE: %.1e'%np.sqrt(np.mean(eps_abs**2)))
    print('\tMAE: %.1e'%np.mean(eps_abs))
    print('\tMax Abs Error: %.1e\n'%eps_abs.max())
    plot_keys = dfx.iloc[:,-1].unique()
    n_keys = len(plot_keys)
    fig = pyplot.figure(figsize=(15,5*n_keys))
    for j,k in enumerate(plot_keys):
        keepers = dfx.iloc[:,-1]==k
        xk = x[keepers,:2]
        yk = y[keepers]
        eps_abs_k = eps_abs[keepers]
        print('\t%s = %d'%(inputs[-1],k))
        print('\t\tRMSE: %.1e'%np.sqrt(np.mean(eps_abs_k**2)))
        print('\t\tMAE: %.1e'%np.mean(eps_abs_k))
        print('\t\tMax Abs Error: %.1e\n'%eps_abs_k.max())
        x0min,x0max = xk[:,0].min(),xk[:,0].max()
        x0_sep = x0max-x0min
        x0_ticks = np.linspace(x0min-sepfrac*x0_sep,x0max+sepfrac*x0_sep,nticks)
        x1min,x1max = xk[:,1].min(),xk[:,1].max()
        x1_sep = x1max-x1min
        x1_ticks = np.linspace(x1min-sepfrac*x1_sep,x1max+sepfrac*x1_sep,nticks)
        x0mesh,x1mesh = np.meshgrid(x0_ticks,x1_ticks)
        xquery = np.hstack([x0mesh.reshape(-1,1),x1mesh.reshape(-1,1),np.tile(k,(x0mesh.size ,1))])
        yquery = model(xquery)
        ymesh = yquery.reshape(x1mesh.shape)
        ax = fig.add_subplot(n_keys,2,1+2*j,projection='3d')
        surf = ax.plot_surface(x0mesh,x1mesh,ymesh,cmap=cm.Greys,alpha=.9,vmin=ymesh.min(),vmax=ymesh.max())
        ax.scatter(xk[:,0],xk[:,1],yk,color='r')
        ax.set_xlabel(inputs[0])
        ax.set_ylabel(inputs[1])
        ax.set_zlabel(output)
        ax.view_init(azim=view_angle)
        ax.set_title('%s = %d'%(inputs[-1],k)) 
        ax = fig.add_subplot(n_keys,2,2+2*j)
        contour = ax.contourf(x0mesh,x1mesh,ymesh,cmap=cm.Greys,alpha=.95,vmin=ymesh.min(),vmax=ymesh.max())
        fig.colorbar(contour,ax=None,shrink=0.5,aspect=5)
        ax.scatter(xk[:,0],xk[:,1],color='r')
        ax.set_xlabel(inputs[0])
        ax.set_ylabel(inputs[1])  
        ax.set_title('%s = %d'%(inputs[-1],k)) 
    fig.suptitle(tag)
    fig.savefig('figures/interpolants/%s.pdf'%name,format='pdf',bbox_inches='tight') 

if __name__ == '__main__':
    '''
    df = pd.read_csv('input/starlifetime/portinari98table14.dat')
    df.columns = [name.replace('#M','mass').replace('Z=','') for name in df.columns]
    df = pd.melt(df,id_vars='mass',value_vars=list(df.columns[1:]),var_name='metallicity',value_name='lifetime_yr')
    df['log10_mass'] = np.log10(df['mass'])
    df['metallicity'] = df['metallicity'].astype(float)
    df['lifetime_Gyr'] = df['lifetime_yr']/1e9
    df['log10_lifetime_Gyr'] = np.log10(df['lifetime_Gyr'])
    lifetimes_interpolant = fit_2d_interpolant(
        dfx = df[['log10_mass','metallicity']],
        dfy = df['log10_lifetime_Gyr'],
        tag = '',
        test = True,
        y_log10_scaled = True,
        view_angle = -45)
    pickle.dump(lifetimes_interpolant,open('output/interpolants/lifetimes_interpolant.pkl','wb'))
    log10_mass_interpolant = fit_2d_interpolant(
        dfx = df[['log10_lifetime_Gyr','metallicity']],
        dfy = df['log10_mass'],
        tag = '',
        test = True,
        y_log10_scaled = True,
        view_angle = -45)
    pickle.dump(log10_mass_interpolant,open('output/interpolants/log10_mass_interpolant.pkl','wb'))
    '''
    dfxs = pickle.load(open('input/yields/lims/k10/processed/X.pkl','rb'))
    dfys = pickle.load(open('input/yields/lims/k10/processed/Y.pkl','rb'))
    idxs = np.arange(len(ZA_sorted)) # [11,19,99,103] # carbon12, oxygen, iron56, iron60
    interpolants = []
    for idx in idxs:
        print(f'{idx=}')
        if dfxs[idx].empty:
            interpolants.append(None)
        else:
            df = dfxs[idx].copy()
            df['log10_y'] = np.log10(dfys[idx])
            df['log10_Z_ini'] = np.log10(df['Z_ini'])
            tag = 'k10_%d, Z=%d, A=%d'%(idx, ZA_sorted[idx,0], ZA_sorted[idx,1])
            log10_yield_lims_interpolant = fit_2d_interpolant(
                dfx = df[['log10_Z_ini','mass_ini']],
                dfy = df['log10_y'],
                tag = tag,
                test = True,
                y_log10_scaled = True,
                view_angle = 135)
            interpolants.append(log10_yield_lims_interpolant)
            #pickle.dump(log10_yield_lims_interpolant,open('output/interpolants/log10_yield_lims_interpolant.%d.pkl'%idx,'wb'))
    interpolants = np.array(interpolants, dtype='object') #, header = "dfx = df[['log10_Z_ini','mass_ini']], dfy = df['log10_mass_frac']"
    #pickle.dump(interpolants,open('output/interpolants/log10_yield_lims_interpolants.pkl','wb'))
    '''
    dfxs = pickle.load(open('input/yields/snii/lc18/tab_R/processed/X.pkl','rb'))
    dfys = pickle.load(open('input/yields/snii/lc18/tab_R/processed/Y.pkl','rb'))
    idxs = np.arange(len(ZA_sorted)) # [11,19,99,103] # carbon12, oxygen, iron56, iron60
    interpolants = []
    for idx in idxs:
        if dfxs[idx].empty:
            interpolants.append(None)
        else:
            df = dfxs[idx].copy()
            df['log10_y'] = np.log10(dfys[idx])
            df['log10_Z_ini'] = np.log10(df['Z_ini'])
            tag = 'lc18_%d, Z=%d, A=%d'%(idx, ZA_sorted[idx,0], ZA_sorted[idx,1])
            log10_yield_snii_interpolant = fit_3d_interpolant(
                dfx = df[['log10_Z_ini','mass_ini','vel_ini']],
                dfy = df['log10_y'],
                tag = tag,
                test = False,
                y_log10_scaled = True,
                view_angle = 135)
            interpolants.append(log10_yield_snii_interpolant)
            #pickle.dump(log10_yield_snii_interpolant,open('output/interpolants/log10_yield_snii_interpolant.%d.pkl'%idx,'wb'))
    interpolants = np.array(interpolants, dtype='object') #, header="dfx = df[['log10_Z_ini','mass_ini','vel_ini']], dfy = df['log10_mass_frac']",
    pickle.dump(interpolants,open('output/interpolants/log10_yield_snii_interpolant.pkl','wb'))
    '''