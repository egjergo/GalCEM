import numpy as np
import pandas as pd
import scipy.interpolate as interp
from matplotlib import pyplot as plt
from matplotlib import cm
from util import mplsetup
mplsetup()

from prep.setup import *

def interpolation(X,Y, kernel='linear'):
    return interp.RBFInterpolator(X,Y, kernel=kernel)

def plot_interpolation(X,Y, xlim, ylim, x1mesh, x2mesh, ymesh, modelname, func):
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(10,7),subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x1mesh,x2mesh,ymesh,cmap=cm.winter,alpha=.75)
    fig.colorbar(surf,shrink=0.5,aspect=5)
    ax.scatter(X[:,0],X[:,1],Y,color='r')
    ax.set_xlim(xlim[:,0])
    ax.set_ylim(xlim[:,1])
    ax.set_zlim(ylim)
    ax.set_zlabel('yields', fontsize=20, y=1.1, rotation=90)
    ax.set_ylabel(X.columns[1], fontsize=20, x=-0.1)
    ax.set_xlabel(X.columns[0], fontsize=20, y=1.1)
    ax.set_title(str(modelname), fontsize=25)
    fig.savefig(f'figures/test/yields_{func.__name__}_{modelname}.pdf', format='pdf', bbox_inches='tight')
    plt.show(block=False)
    return None

def interpolation_test(X,Y, func, modelname=' ', nticks=64):
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
    print(f'For the isotope {modelname} extracted from {func.__name__} the interpolation performs as follows:')
    print('\nRMSE: %.1e'%np.sqrt(np.mean(abs_eps**2)))
    print('MAE: %.1e'%np.mean(abs_eps))
    print('Max Abs Error: %.1e'%abs_eps.max())
    print(' ')
    ymesh = yquery.reshape(x1mesh.shape)
    plot_interpolation(X, Y, xlim, ylim, x1mesh, x2mesh, ymesh, modelname, func)
    return None
    
def lc18_test(i_elemZ, i_elemA, loc='input/yields/snii/lc18/tab_R', filename='lc18_pandas.csv',
              id_vars=['Z_ini', 'vel_ini'], var_name='mass_ini', value_name='yields',
              col_number=9):
    '''
    !!!!!!! will work only if the columns to melt are appended last
    otherwise, change df.columns

        id_vars     existing columns to include in X
        var_name    new column name for melted headers
        value_name  entry descriptor (will always be yields)
        col_number  number of columns to melt 
    '''
    df = pd.read_csv(f'{loc}/{filename}', comment='#')
    df = df.loc[(df['elemZ'] == i_elemZ) & (df['elemA'] == i_elemA)]
    df = pd.melt(df,id_vars=id_vars, value_vars=list(df.columns[-col_number:]),
                 var_name=var_name, value_name=value_name)
    df = df.apply(pd.to_numeric)
    X = df[id_vars+[var_name]]#.values
    Y = df[value_name]#.values
    print(f'X header: {id_vars+[var_name]}')
    print(f'Y header: {[value_name]}')
    return X, Y

def k10_test(i_elemZ, i_elemA, loc='input/yields/lims/k10', filename='k10_pandas.csv',
              id_vars=['Z_ini'], var_name='mass_ini', value_name='yields'):
    df = pd.read_csv(f'{loc}/{filename}', comment='#')
    df = df.loc[(df['elemZ'] == i_elemZ) & (df['elemA'] == i_elemA)]
    X = df[id_vars+[var_name]]#.values
    Y = df[value_name]#.values
    print(f'X header: {id_vars+[var_name]}')
    print(f'Y header: {[value_name]}')
    return X, Y

def test_for_ZA_sorted(func):
    X, Y, models = [], [], []
    for i, val in enumerate(ZA_sorted):
        print(f'i:    {i}')
        Xi, Yi = func(val[0], val[1])
        X.append(Xi)
        Y.append(Yi)
        if Xi.size != 0:
            print(f'X\n{X}')
            print(f'Y\n{Y}')
            models.append(interpolation_test(Xi.values,Yi.values, func, modelname=f' Z={val[0]}, A={val[1]} '))
        else:
            models.append(None)
            print('X is empty')
    return X, Y, models

X_lc18, Y_lc18 = test_for_ZA_sorted(lc18_test)
X_k10, Y_k10 = test_for_ZA_sorted(k10_test)