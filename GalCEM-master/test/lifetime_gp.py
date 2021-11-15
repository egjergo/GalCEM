'''
Better to use their theoretical models directly? 
Can optimize parameters using scipy.optimize.curvefit. 
The GP looks ok and the error metrics aren't terrible,
but it also doesn't recover the points exactly and we cannot enforce monotonicity, continuity, ...
Some of this can be fixed with the RBF kernel, but this gives a terrible fit
so I've stuck with the (significantly more complicated) default kernel. 
'''

import numpy as np
import pandas as pd
from sklearn import gaussian_process
from scipy.interpolate import RBFInterpolator
from matplotlib import pyplot as plt, cm
from util import mplsetup
mplsetup()

# parse data
df = pd.read_csv('input/starlifetime/portinari98table14.dat')
df.columns = [name.replace('#M','mass').replace('Z=','') for name in df.columns]
df = pd.melt(df,id_vars='mass',value_vars=list(df.columns[1:]),var_name='metallicity',value_name='lifetime_yr')
df['log10_mass'] = np.log10(df['mass'])
df['metallicity'] = df['metallicity'].astype(float)
df['log10_lifetime_Gyr'] = np.log10(df['lifetime_yr']/1e9)
df['lifetime_Gyr'] = np.array(df['lifetime_yr']/1e9)
print(df.describe())
X = df[['log10_mass','metallicity']].values
Y = df['log10_lifetime_Gyr'].values

# get GP predictions for plot
xlim = np.array([X.min(0),X.max(0)])
ylim = np.array([Y.min(),Y.max()])
nticks = 64
ticks = [np.linspace(*xlim[:,i],nticks) for i in range(X.shape[1])]
x1mesh,x2mesh = np.meshgrid(*ticks)
xquery = np.hstack([x1mesh.reshape(-1,1),x2mesh.reshape(-1,1)])

Xinv = df[['log10_lifetime_Gyr','metallicity']].values
Yinv = df['log10_mass'].values
xliminv = np.array([Xinv.min(0),Xinv.max(0)])
yliminv = np.array([Yinv.min(),Yinv.max()])
ticksinv = [np.linspace(*xliminv[:,i],nticks) for i in range(Xinv.shape[1])]
x1meshinv,x2meshinv = np.meshgrid(*ticksinv)
xqueryinv = np.hstack([x1meshinv.reshape(-1,1),x2meshinv.reshape(-1,1)])

# fit Model and make predictions
# other models from https://docs.scipy.org/doc/scipy/reference/interpolate.html
modeltype = 'RBFInterpolator' # GaussianProcessRegressor, RBFInterpolator
if modeltype == 'GaussianProcessRegressor':
    gp = gaussian_process.GaussianProcessRegressor()
    gp.fit(X,Y)
    yquery = gp.predict(xquery)
    Yhat = gp.predict(X)
elif modeltype == 'RBFInterpolator':
    model = RBFInterpolator(X,Y)
    yquery = model(xquery)
    Yhat = model(X)
    modelinv = RBFInterpolator(Xinv, Yinv, kernel='linear')
    yqueryinv = modelinv(xqueryinv)
    Yhatinv = modelinv(Xinv)

def interpolation(X,Y):
    return RBFInterpolator(X,Y)

def interpolation_test(X,Y, modelname='tauM', nticks=64):
    xlim = np.array([X.min(0),X.max(0)])
    ylim = np.array([Y.min(),Y.max()])
    ticks = [np.linspace(*xlim[:,i],nticks) for i in range(X.shape[1])]
    x1mesh, x2mesh = np.meshgrid(*ticks)
    xquery = np.hstack([x1mesh.reshape(-1,1),x2mesh.reshape(-1,1)])
    model = interpolation(X,Y)
    yquery = model(xquery)
    Yhat = model(X)
    eps = Y-Yhat
    abs_eps = np.abs(eps)
    print('For the chosen interpolation set (X,Y), the model '+str(modelname)+' performs as follows:')
    print('\nRMSE: %.1e'%np.sqrt(np.mean(abs_eps**2)))
    print('MAE: %.1e'%np.mean(abs_eps))
    print('Max Abs Error: %.1e'%abs_eps.max())
    print(' ')
    ymesh = yquery.reshape(x1mesh.shape)
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,7),subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x1mesh,x2mesh,ymesh,cmap=cm.winter,alpha=.75)
    fig.colorbar(surf,shrink=0.5,aspect=5)
    ax.scatter(X[:,0],X[:,1],Y,color='r')
    ax.set_xlim(xlim[:,0])
    ax.set_ylim(xlim[:,1])
    ax.set_zlim(ylim)
    ax.set_zlabel(r'lifetime', fontsize=20, y=1.1, rotation=90)
    ax.set_ylabel(r'metallicity', fontsize=20, x=-0.1)
    ax.set_xlabel(r'mass', fontsize=20, y=1.1)
    ax.set_title(str(modelname), fontsize=25)
    fig.savefig('figures/test/'+str(modelname)+'_gp.pdf',format='pdf',bbox_inches='tight')
    plt.show(block=False)
    
xquery2 = np.array([[np.log10(120),0.0004],
                    [np.log10(0.6),0.0004]])
yquery2 = model(xquery2)
print(10**yquery2)

# Model predictions at fitting points
eps = Y-Yhat
abs_eps = np.abs(eps)
print('For tau(M)')
print('\nRMSE: %.1e'%np.sqrt(np.mean(abs_eps**2)))
print('MAE: %.1e'%np.mean(abs_eps))
print('Max Abs Error: %.1e'%abs_eps.max())
print(' ')

epsinv = Yinv-Yhatinv
abs_epsinv = np.abs(epsinv)
print('For M(tau)')
print('\nRMSE: %.1e'%np.sqrt(np.mean(abs_epsinv**2)))
print('MAE: %.1e'%np.mean(abs_epsinv))
print('Max Abs Error: %.1e'%abs_epsinv.max())
print(' ')

# plot
ymesh = yquery.reshape(x1mesh.shape)
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,7),subplot_kw={"projection": "3d"})
surf = ax.plot_surface(x1mesh,x2mesh,ymesh,cmap=cm.winter,alpha=.75)
fig.colorbar(surf,shrink=0.5,aspect=5)
ax.scatter(X[:,0],X[:,1],Y,color='r')
ax.set_xlim(xlim[:,0])
ax.set_ylim(xlim[:,1])
ax.set_zlim(ylim)
ax.set_zlabel(r'lifetime', fontsize=20, y=1.1, rotation=90)
ax.set_ylabel(r'metallicity', fontsize=20, x=-0.1)
ax.set_xlabel(r'mass', fontsize=20, y=1.1)
ax.set_title(r'$\tau(M)$', fontsize=25)
fig.savefig('figures/test/lifetimes_gp.pdf',format='pdf',bbox_inches='tight')
plt.show(block=False)


ymeshinv = yqueryinv.reshape(x1meshinv.shape)
fig2,ax2 = plt.subplots(nrows=1,ncols=1,figsize=(10,7),subplot_kw={"projection": "3d"})
surfinv = ax2.plot_surface(x1meshinv,x2meshinv,ymeshinv,cmap=cm.plasma,alpha=.75)
fig2.colorbar(surfinv,shrink=0.5,aspect=5)
ax2.scatter(Xinv[:,0],Xinv[:,1],Yinv,color='k')
ax2.set_xlim(xliminv[:,0])
ax2.set_ylim(xliminv[:,1])
ax2.set_zlim(yliminv)
ax2.set_xlabel(r'lifetime', fontsize=20, y=1.1)
ax2.set_ylabel(r'metallicity', fontsize=20, x=-0.1)
ax2.set_zlabel(r'mass', fontsize=20, y=1.1, rotation=90)
ax2.set_title(r'$M(\tau)$', fontsize=25)
fig2.savefig('figures/test/lifetimesinv_gp.pdf',format='pdf',bbox_inches='tight')
plt.show(block=False)