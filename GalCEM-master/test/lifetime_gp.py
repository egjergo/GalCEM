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
from util import mplsetup
from matplotlib import pyplot as plt, cm
mplsetup()

# parse data
df = pd.read_csv('GalCEM-master/input/starlifetime/portinari98table14.dat')
df.columns = [name.replace('#M','mass').replace('Z=','') for name in df.columns]
df = pd.melt(df,id_vars='mass',value_vars=list(df.columns[1:]),var_name='metalicity',value_name='lifetime_y')
df['log10_mass'] = np.log10(df['mass'])
df['metalicity'] = df['metalicity'].astype(float)
df['log_lifetime_gy'] = np.log10(df['lifetime_y']/1e9)
print(df.describe())
X = df[['log10_mass','metalicity']].values
Y = df['log_lifetime_gy'].values

# fit GP
gp = gaussian_process.GaussianProcessRegressor()
gp.fit(X,Y)

# get GP predictions at fitting points
Yhat = gp.predict(X)
eps = Y-Yhat
abs_eps = np.abs(eps)
print('\nRMSE: %.1e'%np.sqrt(np.mean(abs_eps**2)))
print('MAE: %.1e'%np.mean(abs_eps))
print('Max Abs Error: %.1e'%abs_eps.max())

# get GP predictions for plot
xlim = np.array([X.min(0),X.max(0)])
ylim = np.array([Y.min(),Y.max()])
nticks = 64
ticks = [np.linspace(*xlim[:,i],nticks) for i in range(X.shape[1])]
x1mesh,x2mesh = np.meshgrid(*ticks)
xquery = np.hstack([x1mesh.reshape(-1,1),x2mesh.reshape(-1,1)])
yquery = gp.predict(xquery)
ymesh = yquery.reshape(x1mesh.shape)

# plot
fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,10),subplot_kw={"projection": "3d"})
surf = ax.plot_surface(x1mesh,x2mesh,ymesh,cmap=cm.winter,alpha=.75)
fig.colorbar(surf,shrink=0.5,aspect=5)
ax.scatter(X[:,0],X[:,1],Y,color='r')
ax.set_xlim(xlim[:,0])
ax.set_ylim(xlim[:,1])
ax.set_zlim(ylim)
fig.savefig('GalCEM-master/figures/test/lifetimes_gp.pdf',format='pdf',bbox_inches='tight')
