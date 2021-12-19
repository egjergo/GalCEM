import numpy as np
import pandas as pd
from sklearn import gaussian_process
from scipy.interpolate import RBFInterpolator, LinearNDInterpolator
from matplotlib import pyplot,cm
from util import mplsetup
mplsetup()

def fit_interpolant(X,Y,name,nticks,Ylog10ed):
    Xnpy = X.to_numpy()
    Ynpy = Y.to_numpy()
    model = LinearNDInterpolator(Xnpy,Ynpy,rescale=True,fill_value=np.nan) # https://docs.scipy.org/doc/scipy/reference/interpolate.html
    Yhat = model(Xnpy)
    Yhat_tf = 10**Yhat if Ylog10ed else Yhat
    Y_tf = 10**Y if Ylog10ed else Y
    eps_abs = np.abs(Yhat_tf-Y_tf)
    print('%s metrics'%name)
    print('RMSE: %.1e'%np.sqrt(np.mean(eps_abs**2)))
    print('MAE: %.1e'%np.mean(eps_abs))
    print('Max Abs Error: %.1e\n'%eps_abs.max())
    x1_ticks = np.linspace(Xnpy[:,0].min(),Xnpy[:,0].max(),nticks)
    x2_ticks = np.linspace(Xnpy[:,1].min(),Xnpy[:,1].max(),nticks)
    x1mesh,x2mesh = np.meshgrid(x1_ticks,x2_ticks)
    xquery = np.hstack([x1mesh.reshape(-1,1),x2mesh.reshape(-1,1)])
    yquery = model(xquery)
    ymesh = yquery.reshape(x1mesh.shape)
    fig,ax = pyplot.subplots(nrows=1,ncols=1,figsize=(10,7),subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x1mesh,x2mesh,ymesh,cmap=cm.Greys,alpha=.9)
    fig.colorbar(surf,shrink=0.5,aspect=5)
    ax.scatter(X.iloc[:,0],X.iloc[:,1],Y,color='r')
    ax.set_zlabel(Y.name,fontsize=20,y=1.1,rotation=90)
    ax.set_ylabel(X.columns[0],fontsize=20,x=-0.1)
    ax.set_xlabel(X.columns[1],fontsize=20,y=1.1)
    ax.set_title(name,fontsize=25)
    fig.savefig('figures/test/%s_gp.pdf'%name,format='pdf',bbox_inches='tight')

if __name__ == '__main__':
    # parameters
    df = pd.read_csv('input/starlifetime/portinari98table14.dat')
    df.columns = [name.replace('#M','mass').replace('Z=','') for name in df.columns]
    df = pd.melt(df,id_vars='mass',value_vars=list(df.columns[1:]),var_name='metallicity',value_name='lifetime_yr')
    df['log10_mass'] = np.log10(df['mass'])
    df['metallicity'] = df['metallicity'].astype(float)
    df['lifetime_Gyr'] = df['lifetime_yr']/1e9
    df['log10_lifetime_Gyr'] = np.log10(df['lifetime_Gyr'])
    print(df.describe(),'\n')
    nticks = 64
    # fits
    fit_interpolant(df[['log10_mass','metallicity']],df['log10_lifetime_Gyr'],'lifetimes',nticks,Ylog10ed=True)
    fit_interpolant(df[['log10_lifetime_Gyr','metallicity']],df['log10_mass'],'lifetimesinv',nticks,Ylog10ed=True)
