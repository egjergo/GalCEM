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
from util import mplsetup
from matplotlib import pyplot as plt, cm
mplsetup()

# parse data
df = pd.read_csv('../input/starlifetime/portinari98table14.dat')
df.columns = [name.replace('#M','mass').replace('Z=','') for name in df.columns]
df = pd.melt(df,id_vars='mass',value_vars=list(df.columns[1:]),var_name='metallicity',value_name='lifetime_yr')
df['log10_mass'] = np.log10(df['mass'])
df['metallicity'] = df['metallicity'].astype(float)
df['log_lifetime_Gyr'] = np.log10(df['lifetime_yr']/1e9)
print(df.describe())
X = df[['log10_mass','metallicity']].values
Y = df['log_lifetime_Gyr'].values

# get GP predictions for plot
xlim = np.array([X.min(0),X.max(0)])
ylim = np.array([Y.min(),Y.max()])
nticks = 64
ticks = [np.linspace(*xlim[:,i],nticks) for i in range(X.shape[1])]
x1mesh,x2mesh = np.meshgrid(*ticks)
xquery = np.hstack([x1mesh.reshape(-1,1),x2mesh.reshape(-1,1)])

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

xquery2 = np.array([[np.log10(120),0.0004],
                    [np.log10(0.6),0.0004]])
yquery2 = model(xquery2)
print(10**yquery2)

# Model predictions at fitting points
eps = Y-Yhat
abs_eps = np.abs(eps)
print('\nRMSE: %.1e'%np.sqrt(np.mean(abs_eps**2)))
print('MAE: %.1e'%np.mean(abs_eps))
print('Max Abs Error: %.1e'%abs_eps.max())


# plot
ymesh = yquery.reshape(x1mesh.shape)
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10),subplot_kw={"projection": "3d"})
surf = ax.plot_surface(x1mesh,x2mesh,ymesh,cmap=cm.winter,alpha=.75)
fig.colorbar(surf,shrink=0.5,aspect=5)
ax.scatter(X[:,0],X[:,1],Y,color='r')
ax.set_xlim(xlim[:,0])
ax.set_ylim(xlim[:,1])
ax.set_zlim(ylim)
fig.savefig('GalCEM-master/figures/test/lifetimes_gp.pdf',format='pdf',bbox_inches='tight')
